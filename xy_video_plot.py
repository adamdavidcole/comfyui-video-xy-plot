"""
XY Video Plot Suite - ComfyUI Nodes for Parameter Sweep Video Matrix Generation

This suite provides 4 nodes designed to work with comfyui-easy-use For Loop nodes:

WORKFLOW STRUCTURE:
==================
1. XYPlotSetup (runs ONCE before loop)
   - Parses X/Y axis definitions and values
   - Calculates total iterations
   - Generates unique batch_id for this run
   - Outputs: initial_collection_data (→ For Loop initial_value1),
              total_iterations (→ For Loop iterations),
              xy_config (→ direct pass-through),
              batch_id (→ optional use in filename_prefix, etc.)

2. For Loop Start (comfyui-easy-use)
   - Takes total_iterations as max count
   - Takes initial_collection_data from Setup (no manual typing needed!)
   - Outputs: loop_index, value1 (collection_data state)

3. XYPlotGetValues (runs INSIDE loop, BEFORE generation)
   - Takes loop_index and xy_config
   - Calculates current X/Y values for this iteration
   - Outputs: x_value, y_value (dynamically typed), x_label, y_label

4. [User's workflow: KSampler, image generation, VHS Video Combine, etc.]

5. XYPlotCollectVideo (runs INSIDE loop, AFTER video generation)
   - Takes video_filepath, x_label, y_label, collection_data (from loop)
   - Appends video info to collection_data
   - Outputs: collection_data_out (→ For Loop End)

6. For Loop End (comfyui-easy-use)
   - Receives collection_data_out as value1
   - After all iterations: outputs final_value1

7. XYPlotVideoGrid (runs ONCE after loop completes)
   - Takes collection_data (from For Loop End.final_value1)
   - Assembles all videos into labeled grid
   - Outputs: video_grid tensor, metadata JSON

DATA FLOW PATTERN:
==================
collection_data flows through the loop via For Loop Start/End value* ports:
- Setup: outputs initial "[]" → Loop Start.initial_value1
- Loop Start: value1 → CollectVideo.collection_data
- CollectVideo: appends data → collection_data_out → Loop End.value1
- Loop End: internally feeds back for next iteration
- After loop: Loop End.final_value1 → VideoGrid.collection_data

xy_config flows directly through nodes (NOT through loop value ports):
- Setup → GetValues → CollectVideo → VideoGrid

FUTURE EXPANSION:
=================
- Array inputs: Replace X_Values/Y_Values with IMAGE[], LATENT[] arrays
- Type flexibility: Use wildcard (*) outputs when ComfyUI support improves
- Reference images: Add optional reference_images input to Setup node
- Multi-parameter: Support sweeping 3+ dimensions with nested grids
"""

import os
import json
import torch
import shutil
import time
import numpy as np
import textwrap
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, clips_array
from PIL import ImageFont

import folder_paths
import comfy.utils

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def find_default_font() -> str:
    """Find a suitable default font for text rendering."""
    font_list = ["DejaVu Sans", "Liberation Sans", "Arial", "FreeSans", "Helvetica"]
    for font_name in font_list:
        try:
            ImageFont.truetype(font_name, size=10)
            return font_name
        except IOError:
            continue
    return "serif"

def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filenames."""
    text = str(text)
    return "".join(c for c in text if c.isalnum() or c in ('-', '_')).rstrip()

def parse_csv_values(csv_string: str) -> List[str]:
    """Parse comma-separated values, stripping whitespace."""
    return [v.strip() for v in csv_string.split(',') if v.strip()]

def extract_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract ComfyUI metadata from video file using ffprobe.
    
    Returns dictionary with common workflow metadata (model, steps, cfg, prompt, etc.)
    Returns None if extraction fails or ffprobe is unavailable.
    """
    try:
        # Run ffprobe to get metadata
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        
        if not result.stdout.strip():
            return None
        
        meta = json.loads(result.stdout)
        comment = meta.get("format", {}).get("tags", {}).get("comment")
        
        if not comment:
            return None
        
        # Extract JSON from comment
        start = comment.find('{')
        end = comment.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        
        comment_str = comment[start:end+1]
        comment_json = json.loads(comment_str)
        
        prompt_block = comment_json.get("prompt")
        if isinstance(prompt_block, str):
            prompt_block = json.loads(prompt_block)
        
        # Find the sampler node (typically the first or a KSampler node)
        sampler_node = None
        sampler_key = None
        for key, node in prompt_block.items():
            if node.get("class_type") in ["KSampler", "KSamplerAdvanced"]:
                sampler_node = node
                sampler_key = key
                break
        
        if not sampler_node:
            # Fallback to first node
            sampler_key = next(iter(prompt_block))
            sampler_node = prompt_block[sampler_key]
        
        inputs = sampler_node.get("inputs", {})
        
        # Extract basic sampler parameters
        steps = inputs.get("steps", "")
        cfg = inputs.get("cfg", "")
        seed = inputs.get("seed", "")
        sampler_name = inputs.get("sampler_name", "")
        scheduler = inputs.get("scheduler", "")
        denoise = inputs.get("denoise", "")  # For video-to-video (strength)
        
        # Resolve model name
        model_ptr = inputs.get("model")
        model_name = _resolve_model_name(prompt_block, model_ptr)
        
        # Resolve positive prompt
        positive_ptr = inputs.get("positive")
        positive_prompt = _resolve_prompt_text(prompt_block, positive_ptr)
        
        # Extract LORA information
        lora_names = set()
        for node in prompt_block.values():
            if node.get("class_type") in ["LoraLoader", "LoraLoaderModelOnly"]:
                lora_name = node.get("inputs", {}).get("lora_name")
                if lora_name:
                    lora_names.add(lora_name)
        lora_str = ", ".join(sorted(lora_names)) if lora_names else ""
        
        # Extract latent dimensions
        length = width = height = ""
        latent_ptr = inputs.get("latent_image") or inputs.get("latent")
        if isinstance(latent_ptr, list) and len(latent_ptr) > 0:
            latent_key = str(latent_ptr[0])
            latent_node = prompt_block.get(latent_key, {})
            latent_inputs = latent_node.get("inputs", {})
            length = latent_inputs.get("length", "")
            width = latent_inputs.get("width", "")
            height = latent_inputs.get("height", "")
        
        # Build metadata dictionary
        # Only include values that are actual scalars, not pointer arrays
        meta_dict = {}
        if positive_prompt: meta_dict["prompt"] = positive_prompt
        if model_name: meta_dict["model"] = model_name
        
        # Only include steps/cfg/etc if they're actual values, not pointers
        if steps and not isinstance(steps, list): 
            meta_dict["steps"] = str(steps)
        if cfg and not isinstance(cfg, list): 
            meta_dict["cfg"] = str(cfg)
        
        if sampler_name: meta_dict["sampler"] = sampler_name
        if scheduler: meta_dict["scheduler"] = scheduler
        if lora_str: meta_dict["lora"] = lora_str
        if denoise and denoise != 1.0: meta_dict["strength"] = str(denoise)
        if length: meta_dict["length"] = str(length)
        if width: meta_dict["width"] = str(width)
        if height: meta_dict["height"] = str(height)
        if seed and not isinstance(seed, list): 
            meta_dict["seed"] = str(seed)
        
        return meta_dict if meta_dict else None
        
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Failed to extract metadata from {video_path}: {e}")
        return None

def _resolve_model_name(prompt_block: Dict, model_ptr: Any) -> str:
    """Follow pointer chains to find actual model name."""
    visited = set()
    ptr = model_ptr
    while isinstance(ptr, list) and len(ptr) > 0:
        key = str(ptr[0])
        if key in visited:
            break
        visited.add(key)
        node = prompt_block.get(key, {})
        inputs = node.get("inputs", {})
        model_name = inputs.get("unet_name") or inputs.get("ckpt_name") or inputs.get("model")
        if isinstance(model_name, str):
            return model_name
        ptr = model_name
    return str(model_ptr) if model_ptr else ""

def _resolve_prompt_text(prompt_block: Dict, ptr: Any) -> str:
    """Follow pointer chains to find actual positive prompt text."""
    visited = set()
    while isinstance(ptr, list) and len(ptr) > 0:
        key = str(ptr[0])
        if key in visited:
            break
        visited.add(key)
        node = prompt_block.get(key, {})
        if node.get("class_type") == "CLIPTextEncode":
            return node.get("inputs", {}).get("text", "")
        ptr = node.get("inputs", {}).get("positive")
    return ""

def format_metadata_banner(metadata: Dict[str, Any], batch_id: str, x_axis_name: str, y_axis_name: str, 
                          max_width: int, title_font_size: int, meta_font_size: int) -> Tuple[str, int]:
    """
    Format metadata into multi-line banner text with dynamic height calculation.
    
    Returns:
        banner_text: Formatted text with newlines
        banner_height: Calculated height in pixels
    """
    lines = []
    
    # Line 1: Batch ID as title (emphasized with brackets for visual weight)
    lines.append(f"═══ {batch_id.upper()} ═══")
    
    # Line 2: Prompt (truncated, smaller font)
    prompt = metadata.get("prompt", "")
    if prompt:
        max_prompt_chars = int(max_width / (meta_font_size * 0.6))
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars-3] + "..."
        # Wrap prompt if still too long
        prompt_lines = textwrap.wrap(prompt, width=max_prompt_chars) if len(prompt) > max_prompt_chars else [prompt]
        lines.extend(prompt_lines)
    
    # Line 3+: Technical parameters (smaller font)
    tech_params = []
    
    # Model info
    model = metadata.get("model", "")
    if model:
        # Shorten model name if too long
        if len(model) > 40:
            model = model[:37] + "..."
        tech_params.append(f"Model: {model}")
    
    # Sampler info
    sampler = metadata.get("sampler", "")
    scheduler = metadata.get("scheduler", "")
    if sampler:
        sampler_str = f"Sampler: {sampler}"
        if scheduler:
            sampler_str += f" ({scheduler})"
        tech_params.append(sampler_str)
    
    # Steps and CFG
    steps = metadata.get("steps", "")
    cfg = metadata.get("cfg", "")
    if steps:
        tech_params.append(f"Steps: {steps}")
    if cfg:
        tech_params.append(f"CFG: {cfg}")
    
    # Strength (for video-to-video)
    strength = metadata.get("strength", "")
    if strength:
        tech_params.append(f"Strength: {strength}")
    
    # Dimensions
    length = metadata.get("length", "")
    width = metadata.get("width", "")
    height = metadata.get("height", "")
    if length or width or height:
        dim_str = "Dimensions: "
        if length:
            dim_str += f"{length}f"
        if width and height:
            dim_str += f" {width}×{height}"
        tech_params.append(dim_str)
    
    # LORA
    lora = metadata.get("lora", "")
    if lora:
        tech_params.append(f"LORA: {lora}")
    
    # Join technical parameters with | separator and wrap if needed
    if tech_params:
        tech_line = " | ".join(tech_params)
        max_tech_chars = int(max_width / (meta_font_size * 0.6))
        tech_wrapped = textwrap.wrap(tech_line, width=max_tech_chars) if len(tech_line) > max_tech_chars else [tech_line]
        lines.extend(tech_wrapped)
    
    # Last line: Axis information (smaller font)
    lines.append(f"X-Axis: {x_axis_name} | Y-Axis: {y_axis_name}")
    
    banner_text = "\n".join(lines)
    
    # Calculate height
    title_lines = 1
    meta_lines = len(lines) - 1
    title_line_height = title_font_size * 1.3
    meta_line_height = meta_font_size * 1.2
    top_padding = 15
    bottom_padding = 15
    
    banner_height = int(
        top_padding + 
        (title_lines * title_line_height) + 
        (meta_lines * meta_line_height) + 
        bottom_padding
    )
    
    return banner_text, banner_height

# ================================================================================
# NODE 0: XYPlotDirectorySetup (OPTIONAL - Directory Organization Helper)
# ================================================================================

class XYPlotDirectorySetup:
    """
    Generate organized directory paths for XY plot outputs.
    
    This is an OPTIONAL utility node that helps organize outputs into structured folders.
    Can be placed between XYPlotSetup and the video generation workflow.
    
    Creates directory structure:
      {folder_prefix}/{batch_id}[-{x_axis}-{y_axis}]/videos/vid-
      {folder_prefix}/{batch_id}[-{x_axis}-{y_axis}]/grid-{x_axis}-{y_axis}-
    
    Also saves minimal metadata JSON for future reference.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_id": ("STRING", {"forceInput": True}),
                "xy_config": ("STRING", {"forceInput": True}),
                "folder_prefix": ("STRING", {"default": "grid_", "multiline": False}),
                "include_axis_names": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_filename_prefix", "grid_filename_prefix", "metadata_path")
    FUNCTION = "setup_directories"
    CATEGORY = "VideoPlot"
    
    def setup_directories(self, batch_id: str, xy_config: str, folder_prefix: str,
                         include_axis_names: bool, save_metadata: bool, unique_id=None) -> Tuple[str, str, str]:
        """
        Create organized directory structure and return filename prefixes.
        
        Args:
            batch_id: Unique batch identifier from XYPlotSetup
            xy_config: Configuration JSON from XYPlotSetup
            folder_prefix: Base folder path (can include subdirectories, e.g., "WanGridTest/")
            include_axis_names: Whether to include axis names in directory path
            save_metadata: Whether to save metadata.json file
            unique_id: Node unique ID for progress bar (hidden input)
            
        Returns:
            video_filename_prefix: Path prefix for individual videos (→ VHS Video Combine)
            grid_filename_prefix: Path prefix for final grid video (→ VHS Video Combine)
            metadata_path: Path to metadata.json (empty string if save_metadata=False)
        """
        # Initialize progress bar (4 steps total)
        pbar = comfy.utils.ProgressBar(4, node_id=unique_id)
        
        print(f"[XY Plot Directory Setup] Starting directory setup for batch: {batch_id}")
        pbar.update_absolute(1, 4)
        
        # Parse config
        config = json.loads(xy_config)
        x_axis_name = sanitize_filename(config["x_axis_name"])
        y_axis_name = sanitize_filename(config["y_axis_name"])
        
        print(f"[XY Plot Directory Setup] Axes: X={config['x_axis_name']}, Y={config['y_axis_name']}")
        
        # Build directory path
        if include_axis_names:
            dir_name = f"{batch_id}-{x_axis_name}-{y_axis_name}"
        else:
            dir_name = batch_id
        
        # Determine if folder_prefix should be treated as a directory path or a filename prefix
        # If it ends with '/', treat as directory. Otherwise, treat as filename prefix.
        if folder_prefix:
            if folder_prefix.endswith('/'):
                # Directory path: "WanGridTest/" -> "WanGridTest/20251004_170219/"
                full_dir_name = f"{folder_prefix}{dir_name}"
            else:
                # Filename prefix: "grid_" -> "grid_20251004_170219/"
                full_dir_name = f"{folder_prefix}{dir_name}"
        else:
            # No prefix: just use dir_name
            full_dir_name = dir_name
        
        pbar.update_absolute(2, 4)
        
        base_dir = os.path.join(folder_paths.get_output_directory(), full_dir_name)
        videos_dir = os.path.join(base_dir, "videos")
        
        # Create directories
        print(f"[XY Plot Directory Setup] Creating directories: {base_dir}")
        os.makedirs(videos_dir, exist_ok=True)
        print(f"[XY Plot Directory Setup] ✓ Videos directory: {videos_dir}")
        
        pbar.update_absolute(3, 4)
        
        # Build filename prefixes (relative paths from ComfyUI output directory)
        # VHS Video Combine expects paths relative to output directory
        video_prefix = f"{full_dir_name}/videos/vid-"
        
        if include_axis_names:
            grid_prefix = f"{full_dir_name}/grid-{x_axis_name}-{y_axis_name}-"
        else:
            grid_prefix = f"{full_dir_name}/grid-"
        
        print(f"[XY Plot Directory Setup] Video prefix: {video_prefix}")
        print(f"[XY Plot Directory Setup] Grid prefix: {grid_prefix}")
        
        # Save metadata if requested
        metadata_path = ""
        if save_metadata:
            metadata = {
                "batch_id": batch_id,
                "x_axis": config["x_axis_name"],
                "y_axis": config["y_axis_name"],
                "x_values": config["x_values"],
                "y_values": config["y_values"],
                "created": datetime.now().isoformat(),
                "grid_shape": [config["num_rows"], config["num_cols"]],
                "total_iterations": config["total_iterations"]
            }
            
            metadata_file = os.path.join(base_dir, "grid-metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            metadata_path = metadata_file
            print(f"[XY Plot Directory Setup] ✓ Saved metadata to: {metadata_path}")
        
        pbar.update_absolute(4, 4)
        
        print(f"[XY Plot Directory Setup] ✓ Setup complete!")
        print(f"[XY Plot Directory Setup]   Video prefix: {video_prefix}")
        print(f"[XY Plot Directory Setup]   Grid prefix: {grid_prefix}")
        
        return (video_prefix, grid_prefix, metadata_path)


# ================================================================================
# NODE 1: XYPlotSetup
# ================================================================================

class XYPlotSetup:
    """
    Initialize XY plot parameters and calculate total iterations.
    
    This node runs ONCE before the For Loop starts.
    Outputs total_iterations to feed into For Loop Start node.
    Outputs xy_config JSON containing all plot metadata to pass through the loop.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        TYPES = ["INT", "FLOAT", "STRING"]
        return {
            "required": {
                "x_axis_name": ("STRING", {"default": "seed", "multiline": False}),
                "x_axis_type": (TYPES, {"default": "INT"}),
                "x_values": ("STRING", {"default": "1, 2, 3", "multiline": False}),
                "y_axis_name": ("STRING", {"default": "cfg", "multiline": False}),
                "y_axis_type": (TYPES, {"default": "FLOAT"}),
                "y_values": ("STRING", {"default": "7.0, 8.0, 9.0", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("initial_collection_data", "total_iterations", "xy_config", "batch_id")
    FUNCTION = "setup"
    CATEGORY = "VideoPlot"
    
    def setup(self, x_axis_name: str, y_axis_name: str, x_axis_type: str, y_axis_type: str, 
              x_values: str, y_values: str) -> Tuple[str, int, str, str]:
        """
        Parse axis definitions and create configuration.
        
        Returns:
            initial_collection_data: Empty JSON array "[]" for For Loop Start initial_value1
            total_iterations: Number of loop iterations needed (len(x_vals) * len(y_vals))
            xy_config: JSON string containing all plot metadata
            batch_id: Unique filesystem-safe identifier for this XY plot run (YYYYMMDD_HHMMSS)
        """
        x_vals = parse_csv_values(x_values)
        y_vals = parse_csv_values(y_values)
        
        if not x_vals or not y_vals:
            raise ValueError("X and Y values must not be empty")
        
        total_iterations = len(x_vals) * len(y_vals)
        
        # Generate unique batch ID (filesystem-safe, compact)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create grid layout: iterate Y outer, X inner (row-major order)
        # Grid will be: rows=y_vals, cols=x_vals
        jobs = []
        for y_val in y_vals:
            for x_val in x_vals:
                jobs.append({"x": x_val, "y": y_val})
        
        config = {
            "x_axis_name": x_axis_name,
            "y_axis_name": y_axis_name,
            "x_axis_type": x_axis_type,
            "y_axis_type": y_axis_type,
            "x_values": x_vals,
            "y_values": y_vals,
            "total_iterations": total_iterations,
            "jobs": jobs,
            "num_cols": len(x_vals),
            "num_rows": len(y_vals),
            "batch_id": batch_id
        }
        
        # Return: initial_collection_data, total_iterations, xy_config, batch_id
        return ("[]", total_iterations, json.dumps(config), batch_id)


# ================================================================================
# NODE 2: XYPlotGetValues
# ================================================================================

class XYPlotGetValues:
    """
    Get X/Y values for the current loop iteration.
    
    This node runs INSIDE the For Loop, BEFORE video generation.
    Takes loop_index from For Loop Start and xy_config from Setup.
    Returns current x/y values with DYNAMIC TYPES based on setup configuration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xy_config": ("STRING", {"forceInput": True}),
                "loop_index": ("INT", {"forceInput": True, "default": 0}),
            }
        }
    
    RETURN_TYPES = ("INT,FLOAT,STRING", "INT,FLOAT,STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("x_value", "y_value", "x_label", "y_label", "xy_config")
    FUNCTION = "get_values"
    CATEGORY = "VideoPlot"
    
    def get_values(self, xy_config: str, loop_index: int) -> Tuple[Any, Any, str, str, str]:
        """
        Extract X/Y values for current iteration with proper typing.
        
        Args:
            xy_config: JSON config from XYPlotSetup (includes type info)
            loop_index: Current iteration index from For Loop (0-based)
            
        Returns:
            x_value: Current X value (typed as INT, FLOAT, or STRING)
            y_value: Current Y value (typed as INT, FLOAT, or STRING)
            x_label: Display label for X (e.g., "seed: 42")
            y_label: Display label for Y (e.g., "cfg: 7.5")
            xy_config: Pass-through for next nodes
        """
        config = json.loads(xy_config)
        jobs = config["jobs"]
        
        if loop_index >= len(jobs):
            raise ValueError(f"loop_index {loop_index} out of range (max: {len(jobs)-1})")
        
        job = jobs[loop_index]
        x_str = str(job["x"])
        y_str = str(job["y"])
        
        # Convert to appropriate types based on config
        x_type = config.get("x_axis_type", "STRING")
        y_type = config.get("y_axis_type", "STRING")
        
        # Type conversion for X
        if x_type == "INT":
            try:
                x_value = int(float(x_str))  # Handle "1.0" → 1
            except ValueError:
                print(f"Warning: Could not convert X value '{x_str}' to INT. Using 0.")
                x_value = 0
        elif x_type == "FLOAT":
            try:
                x_value = float(x_str)
            except ValueError:
                print(f"Warning: Could not convert X value '{x_str}' to FLOAT. Using 0.0.")
                x_value = 0.0
        else:  # STRING
            x_value = x_str
        
        # Type conversion for Y
        if y_type == "INT":
            try:
                y_value = int(float(y_str))  # Handle "1.0" → 1
            except ValueError:
                print(f"Warning: Could not convert Y value '{y_str}' to INT. Using 0.")
                y_value = 0
        elif y_type == "FLOAT":
            try:
                y_value = float(y_str)
            except ValueError:
                print(f"Warning: Could not convert Y value '{y_str}' to FLOAT. Using 0.0.")
                y_value = 0.0
        else:  # STRING
            y_value = y_str
        
        x_label = f"{config['x_axis_name']}: {x_str}"
        y_label = f"{config['y_axis_name']}: {y_str}"
        
        return (x_value, y_value, x_label, y_label, xy_config)


# ================================================================================
# NODE 3: XYPlotCollectVideo
# ================================================================================

class XYPlotCollectVideo:
    """
    Collect video filepath with labels for grid assembly.
    
    This node runs INSIDE the For Loop, AFTER video generation.
    Uses pass-through pattern: receives collection_data, appends new entry, outputs updated data.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_filepath": ("STRING", {"forceInput": True}),
                "x_label": ("STRING", {"forceInput": True}),
                "y_label": ("STRING", {"forceInput": True}),
                "xy_config": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "collection_data": ("STRING", {"default": "[]"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("collection_data", "xy_config")
    FUNCTION = "collect"
    CATEGORY = "VideoPlot"
    
    def collect(self, video_filepath: str, x_label: str, y_label: str, 
                xy_config: str, collection_data: str = "[]", unique_id=None) -> Tuple[str, str]:
        """
        Append video info to collection.
        
        Args:
            video_filepath: Path to generated video file
            x_label: X axis label for this video
            y_label: Y axis label for this video
            xy_config: Pass-through config
            collection_data: JSON array from previous iteration (or "[]" if first)
            unique_id: Node unique ID for progress bar (hidden input)
            
        Returns:
            collection_data: Updated JSON array with new entry appended
            xy_config: Pass-through for next iteration
        """
        # Parse existing collection and config
        try:
            data = json.loads(collection_data)
        except json.JSONDecodeError:
            data = []
        
        config = json.loads(xy_config)
        total_iterations = config.get("total_iterations", 0)
        current_iteration = len(data) + 1  # Current video number (1-indexed)
        
        # Validate video file exists and log progress
        if not os.path.exists(video_filepath):
            print(f"[XY Plot Collect] Warning: Video file not found: {video_filepath}")
        else:
            file_size_mb = os.path.getsize(video_filepath) / (1024 * 1024)
            filename = os.path.basename(video_filepath)
            print(f"[XY Plot Collect] Video {current_iteration}/{total_iterations} ({x_label}, {y_label}): {filename} ({file_size_mb:.2f} MB)")
        
        # Append new entry
        data.append({
            "filepath": video_filepath,
            "x_label": x_label,
            "y_label": y_label
        })
        
        return (json.dumps(data), xy_config)


# ================================================================================
# NODE 4: XYPlotVideoGrid
# ================================================================================

class XYPlotVideoGrid:
    """
    Assemble collected videos into a labeled grid.
    
    This node runs ONCE after the For Loop completes.
    Takes final collection_data from For Loop End and generates the video grid.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "collection_data": ("STRING", {"forceInput": True}),
                "xy_config": ("STRING", {"forceInput": True}),
                "max_cell_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "banner_font_size": ("INT", {"default": 20, "min": 1, "max": 128}),
                "label_font_size": ("INT", {"default": 14, "min": 1, "max": 128}),
                "font_color": ("STRING", {"default": "white"}),
                "bg_color": ("STRING", {"default": "black"}),
            },
            "optional": {
                "include_metadata": ("BOOLEAN", {"default": True}),
                "cleanup_files": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video_grid", "metadata")
    FUNCTION = "create_grid"
    CATEGORY = "VideoPlot"
    OUTPUT_NODE = True
    
    def create_grid(self, collection_data: str, xy_config: str, max_cell_width: int,
                   banner_font_size: int, label_font_size: int,
                   font_color: str, bg_color: str, include_metadata: bool = True, 
                   cleanup_files: bool = False) -> Tuple[torch.Tensor, str]:
        """
        Create video grid from collected videos.
        
        Args:
            collection_data: JSON array of collected video entries
            xy_config: Grid configuration from setup
            thumb_width/height: Dimensions for each video cell
            banner_*: Styling for title banner
            cleanup_files: Whether to delete source video files after assembly
            
        Returns:
            video_grid: Assembled video as IMAGE tensor
            metadata: JSON with grid info
        """
        # Parse inputs
        try:
            file_batch = json.loads(collection_data)
            config = json.loads(xy_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON inputs: {e}")
        
        if not file_batch:
            raise ValueError("No videos collected. collection_data is empty.")
        
        # Extract grid dimensions from config
        num_cols = config["num_cols"]
        num_rows = config["num_rows"]
        x_axis_name = config["x_axis_name"]
        y_axis_name = config["y_axis_name"]
        
        # Get labels in original input order (not sorted)
        # Build labels from original x_values and y_values in config
        x_labels = [f"{x_axis_name}: {val}" for val in config["x_values"]]
        y_labels = [f"{y_axis_name}: {val}" for val in config["y_values"]]
        
        # Load first video to detect aspect ratio
        first_video_path = None
        for item in file_batch:
            if os.path.exists(item['filepath']):
                first_video_path = item['filepath']
                break
        
        if not first_video_path:
            raise ValueError("No valid video files found")
        
        # Detect aspect ratio from first video
        with VideoFileClip(first_video_path) as probe_clip:
            original_width = probe_clip.w
            original_height = probe_clip.h
            aspect_ratio = original_width / original_height
            duration = probe_clip.duration
            fps = probe_clip.fps
        
        # Calculate cell dimensions preserving aspect ratio
        cell_width = max_cell_width
        cell_height = int(cell_width / aspect_ratio)
        
        # Extract metadata from first video (only if include_metadata is True)
        batch_id = config.get("batch_id", "XY_Plot")
        metadata = {}
        
        if include_metadata:
            metadata = extract_video_metadata(first_video_path)
            if metadata:
                print(f"Extracted metadata: {metadata}")
            else:
                print("No metadata extracted from video (ffprobe may not be available)")
                metadata = {}
        
        # Layout margins and spacing
        left_margin = 15
        bottom_margin = 15
        top_padding = 15           # Extra space above/below banner (increased)
        y_axis_label_width = 60    # For horizontal "Y-Axis" label (was 50px for vertical)
        y_values_width = 80        # For horizontal Y values (e.g., "2.0", "7.5")
        x_axis_label_height = 30   # For "X-Axis" centered label
        x_values_height = 30       # For X values row
        
        # Create banner text (only if metadata is enabled)
        grid_width = num_cols * cell_width
        banner_height = 0
        banner_text = ""
        
        if include_metadata and metadata:
            # Use smaller font for metadata lines (banner_font_size for title, label_font_size for details)
            banner_text, banner_height = format_metadata_banner(
                metadata, 
                batch_id, 
                x_axis_name, 
                y_axis_name,
                grid_width,
                title_font_size=banner_font_size,
                meta_font_size=label_font_size
            )
        # If include_metadata is False, banner_height stays 0 (no banner)
        
        # Build grid mapping: [row][col] -> filepath
        grid_map: List[List[Optional[str]]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        x_map = {label: i for i, label in enumerate(x_labels)}
        y_map = {label: i for i, label in enumerate(y_labels)}
        
        for item in file_batch:
            if item['y_label'] in y_map and item['x_label'] in x_map:
                row = y_map[item['y_label']]
                col = x_map[item['x_label']]
                grid_map[row][col] = item['filepath']
        
        # Load and resize all video clips to calculated dimensions
        all_clips = []
        for item in file_batch:
            if os.path.exists(item['filepath']):
                clip = VideoFileClip(item['filepath']).resized(width=cell_width, height=cell_height)
                all_clips.append(clip)
            else:
                print(f"Warning: Video file not found: {item['filepath']}")
        
        if not all_clips:
            raise ValueError("No valid video files found")
        
        # Build rows of clips for grid
        rows_of_clips = []
        for r_idx in range(num_rows):
            row_clips = []
            for c_idx in range(num_cols):
                fpath = grid_map[r_idx][c_idx]
                if fpath and os.path.exists(fpath):
                    # Find the clip with matching filepath
                    clip_to_add = next((c for c in all_clips if c.filename == fpath), None)
                    if clip_to_add:
                        row_clips.append(clip_to_add)
                    else:
                        # Create black placeholder if clip not found
                        placeholder = ColorClip(size=(cell_width, cell_height), color=(0, 0, 0), 
                                              duration=duration)
                        row_clips.append(placeholder)
                else:
                    # Create black placeholder for missing videos
                    placeholder = ColorClip(size=(cell_width, cell_height), color=(0, 0, 0), 
                                          duration=duration)
                    row_clips.append(placeholder)
            rows_of_clips.append(row_clips)
        
        # Pad last row if incomplete
        if len(rows_of_clips[-1]) < num_cols:
            pad_clip = ColorClip(size=(cell_width, cell_height), color=(0, 0, 0), 
                               duration=duration).with_opacity(0)
            rows_of_clips[-1].extend([pad_clip] * (num_cols - len(rows_of_clips[-1])))
        else:
            pad_clip = None  # Ensure pad_clip is defined
        
        # Create grid
        grid_clip = clips_array(rows_of_clips)
        
        # Get font for labels
        font_to_use = find_default_font()
        
        # Calculate total dimensions with margins and labels
        total_label_width = left_margin + y_axis_label_width + y_values_width
        total_header_height = banner_height + x_axis_label_height + x_values_height
        grid_content_width = total_label_width + grid_clip.w + left_margin
        grid_content_height = total_header_height + grid_clip.h + bottom_margin
        
        # Create title banner (only if include_metadata is enabled)
        banner_clip = None
        if include_metadata and metadata and banner_height > 0:
            # Calculate max width for banner text (grid width + some extra room)
            banner_max_width = grid_clip.w + y_values_width + y_axis_label_width
            
            banner_clip = TextClip(
                text=banner_text,
                font=font_to_use,
                font_size=label_font_size,  # Use smaller font for metadata banners
                color=font_color,
                bg_color=bg_color,
                size=(banner_max_width, banner_height),
                method='caption'
            ).with_duration(duration)
            # Position at actual top-left corner
            banner_clip = banner_clip.with_position((left_margin, top_padding))
        
        # Create X-axis label header (centered above all columns)
        x_axis_header = TextClip(
            text=x_axis_name,
            font=font_to_use,
            font_size=label_font_size,
            color=font_color,
            bg_color=bg_color,
            size=(grid_clip.w, x_axis_label_height),
            method='caption'
        ).with_duration(duration)
        x_axis_header = x_axis_header.with_position((total_label_width, banner_height))
        
        # Create X-axis values (one per column, centered)
        x_value_clips = []
        for i, x_label in enumerate(x_labels):
            # Extract just the value part (e.g., "seed: 1" -> "1")
            label_text = x_label.split(": ", 1)[1] if ": " in x_label else x_label
            x_val_clip = TextClip(
                text=label_text,
                font=font_to_use,
                font_size=label_font_size,
                color=font_color,
                bg_color=bg_color,
                size=(cell_width, x_values_height),
                method='caption'
            ).with_duration(duration)
            # Position: after labels, below x-axis header
            x_pos = total_label_width + (i * cell_width)
            y_pos = banner_height + x_axis_label_height
            x_val_clip = x_val_clip.with_position((x_pos, y_pos))
            x_value_clips.append(x_val_clip)
        
        # Create Y-axis label (horizontal, far left, centered vertically on grid)
        y_axis_label = TextClip(
            text=y_axis_name,
            font=font_to_use,
            font_size=label_font_size,
            color=font_color,
            bg_color=bg_color,
            size=(y_axis_label_width, grid_clip.h),
            method='caption'
        ).with_duration(duration)
        
        # Position: far left, aligned with grid vertically
        y_label_x = left_margin
        y_label_y = total_header_height
        y_axis_label = y_axis_label.with_position((y_label_x, y_label_y))
        
        # Create Y-axis values (horizontal, one per row)
        y_value_clips = []
        for i, y_label in enumerate(y_labels):
            # Extract just the value part
            label_text = y_label.split(": ", 1)[1] if ": " in y_label else y_label
            y_val_clip = TextClip(
                text=label_text,
                font=font_to_use,
                font_size=label_font_size,
                color=font_color,
                bg_color=bg_color,
                size=(y_values_width, cell_height),
                method='caption'
            ).with_duration(duration)
            # Position: after y-axis label, aligned with row
            x_pos = left_margin + y_axis_label_width
            y_pos = total_header_height + (i * cell_height)
            y_val_clip = y_val_clip.with_position((x_pos, y_pos))
            y_value_clips.append(y_val_clip)
        
        # Create background
        background = ColorClip(size=(grid_content_width, grid_content_height), color=(0, 0, 0), duration=duration)
        
        # Position the main grid
        grid_x_pos = total_label_width
        grid_y_pos = total_header_height
        grid_positioned = grid_clip.with_position((grid_x_pos, grid_y_pos))
        
        # Composite everything (conditionally include banner if it exists)
        all_clips = [
            background,
            x_axis_header,
            y_axis_label,
            grid_positioned
        ] + x_value_clips + y_value_clips
        
        # Add banner clip only if it was created (when include_metadata=True)
        if banner_clip is not None:
            all_clips.insert(1, banner_clip)  # Insert after background
        
        final_video = CompositeVideoClip(all_clips)
        
        # Render to temporary file and load as tensor
        temp_filepath = os.path.join(
            folder_paths.get_temp_directory(),
            f"xy_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        
        output_tensor = None
        try:
            final_video.write_videofile(
                temp_filepath,
                fps=fps,
                codec="libx264",
                preset="fast",
                logger=None
            )
            
            # Load video as tensor
            with VideoFileClip(temp_filepath) as result_clip:
                frames = [torch.from_numpy(frame).float() / 255.0 for frame in result_clip.iter_frames()]
            output_tensor = torch.stack(frames)
            
        finally:
            # Cleanup video clips first (important: close before deleting files)
            final_video.close()
            for clip in all_clips:
                try:
                    clip.close()
                except Exception as e:
                    print(f"Warning: Failed to close clip: {e}")
            if pad_clip is not None:
                try:
                    pad_clip.close()
                except Exception as e:
                    print(f"Warning: Failed to close pad clip: {e}")
            
            # Delete temporary grid file
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file {temp_filepath}: {e}")
        
        # Optional: cleanup source video files (after all clips are closed)
        if cleanup_files:
            import time
            time.sleep(0.1)  # Give OS time to release file handles
            for item in file_batch:
                video_path = item['filepath']
                try:
                    # Delete video file
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"Deleted source video: {video_path}")
                    
                    # Delete corresponding PNG file (same name, .png extension)
                    png_path = os.path.splitext(video_path)[0] + '.png'
                    if os.path.exists(png_path):
                        os.remove(png_path)
                        print(f"Deleted thumbnail: {png_path}")
                        
                except Exception as e:
                    print(f"Warning: Failed to delete {video_path} or its thumbnail: {e}")
        
        # Generate metadata
        metadata = {
            "grid_shape": [num_rows, num_cols],
            "x_axis": x_axis_name,
            "y_axis": y_axis_name,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "total_videos": len(file_batch),
            "fps": fps,
            "duration": duration
        }
        
        return (output_tensor, json.dumps(metadata, indent=2))


# ================================================================================
# NODE REGISTRATION
# ================================================================================

NODE_CLASS_MAPPINGS = {
    "XYPlotSetup": XYPlotSetup,
    "XYPlotDirectorySetup": XYPlotDirectorySetup,
    "XYPlotGetValues": XYPlotGetValues,
    "XYPlotCollectVideo": XYPlotCollectVideo,
    "XYPlotVideoGrid": XYPlotVideoGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XYPlotSetup": "XY Plot Setup",
    "XYPlotDirectorySetup": "XY Plot Directory Setup",
    "XYPlotGetValues": "XY Plot Get Values",
    "XYPlotCollectVideo": "XY Plot Collect Video",
    "XYPlotVideoGrid": "XY Plot Video Grid",
}