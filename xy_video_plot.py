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
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, clips_array
from PIL import ImageFont

import folder_paths

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
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("collection_data", "xy_config")
    FUNCTION = "collect"
    CATEGORY = "VideoPlot"
    
    def collect(self, video_filepath: str, x_label: str, y_label: str, 
                xy_config: str, collection_data: str = "[]") -> Tuple[str, str]:
        """
        Append video info to collection.
        
        Args:
            video_filepath: Path to generated video file
            x_label: X axis label for this video
            y_label: Y axis label for this video
            xy_config: Pass-through config
            collection_data: JSON array from previous iteration (or "[]" if first)
            
        Returns:
            collection_data: Updated JSON array with new entry appended
            xy_config: Pass-through for next iteration
        """
        # Parse existing collection
        try:
            data = json.loads(collection_data)
        except json.JSONDecodeError:
            data = []
        
        # Validate video file exists
        if not os.path.exists(video_filepath):
            print(f"Warning: Video file not found: {video_filepath}")
        
        # Append new entry
        data.append({
            "filepath": video_filepath,
            "x_label": x_label,
            "y_label": y_label
        })
        
        return (json.dumps(data), xy_config)


# ================================================================================
# NODE 3B: XYPlotGetValuesTyped (Fallback with Multiple Typed Outputs)
# ================================================================================

class XYPlotGetValuesTyped:
    """
    Alternative to XYPlotGetValues with explicit typed outputs.
    
    Use this node if KSampler doesn't accept the dynamic typing from XYPlotGetValues.
    Outputs all type variations - connect the appropriate one to your parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xy_config": ("STRING", {"forceInput": True}),
                "loop_index": ("INT", {"forceInput": True, "default": 0}),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "STRING", "INT", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("x_int", "x_float", "x_str", "y_int", "y_float", "y_str", "x_label", "y_label", "xy_config")
    FUNCTION = "get_values_typed"
    CATEGORY = "VideoPlot"
    
    def get_values_typed(self, xy_config: str, loop_index: int) -> Tuple[int, float, str, int, float, str, str, str, str]:
        """
        Extract X/Y values with all type variations.
        
        Returns:
            x_int: X as INT
            x_float: X as FLOAT
            x_str: X as STRING
            y_int: Y as INT
            y_float: Y as FLOAT
            y_str: Y as STRING
            x_label, y_label, xy_config: Same as XYPlotGetValues
        """
        config = json.loads(xy_config)
        jobs = config["jobs"]
        
        if loop_index >= len(jobs):
            raise ValueError(f"loop_index {loop_index} out of range (max: {len(jobs)-1})")
        
        job = jobs[loop_index]
        x_str = str(job["x"])
        y_str = str(job["y"])
        
        # Convert X to all types
        try:
            x_int = int(float(x_str))
        except ValueError:
            x_int = 0
            print(f"Warning: Could not convert X '{x_str}' to INT")
        
        try:
            x_float = float(x_str)
        except ValueError:
            x_float = 0.0
            print(f"Warning: Could not convert X '{x_str}' to FLOAT")
        
        # Convert Y to all types
        try:
            y_int = int(float(y_str))
        except ValueError:
            y_int = 0
            print(f"Warning: Could not convert Y '{y_str}' to INT")
        
        try:
            y_float = float(y_str)
        except ValueError:
            y_float = 0.0
            print(f"Warning: Could not convert Y '{y_str}' to FLOAT")
        
        x_label = f"{config['x_axis_name']}: {x_str}"
        y_label = f"{config['y_axis_name']}: {y_str}"
        
        return (x_int, x_float, x_str, y_int, y_float, y_str, x_label, y_label, xy_config)


# ================================================================================
# NODE 4: XYPlotCollectVideo
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
                "cleanup_files": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video_grid", "metadata")
    FUNCTION = "create_grid"
    CATEGORY = "VideoPlot"
    OUTPUT_NODE = True
    
    def create_grid(self, collection_data: str, xy_config: str, max_cell_width: int,
                   banner_font_size: int, label_font_size: int,
                   font_color: str, bg_color: str, cleanup_files: bool = True) -> Tuple[torch.Tensor, str]:
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
        
        # Get unique sorted labels
        x_labels = sorted(list(set(item['x_label'] for item in file_batch)))
        y_labels = sorted(list(set(item['y_label'] for item in file_batch)))
        
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
        
        # Layout margins and spacing
        left_margin = 15
        bottom_margin = 15
        top_padding = 15           # Extra space above/below banner (increased)
        y_axis_label_width = 60    # For horizontal "Y-Axis" label (was 50px for vertical)
        y_values_width = 80        # For horizontal Y values (e.g., "2.0", "7.5")
        x_axis_label_height = 30   # For "X-Axis" centered label
        x_values_height = 30       # For X values row
        
        # Create banner text
        banner_text = f"X-Axis: {x_axis_name} | Y-Axis: {y_axis_name}"
        grid_width = num_cols * cell_width
        max_chars = int(grid_width / (banner_font_size * 0.6))
        wrapped_text = "\n".join(textwrap.wrap(banner_text, width=max_chars if max_chars > 0 else 50))
        
        line_count = wrapped_text.count('\n') + 1
        line_height = banner_font_size * 1.2
        banner_height = int(line_count * line_height + top_padding * 2)  # Add padding top and bottom
        
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
        
        # Create title banner (positioned at top-left corner, spans grid width)
        # Calculate max width for banner text (grid width + some extra room)
        banner_max_width = grid_clip.w + y_values_width + y_axis_label_width
        banner_clip = TextClip(
            text=wrapped_text,
            font=font_to_use,
            font_size=banner_font_size,
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
        
        # Composite everything
        all_clips = [
            background,
            banner_clip,  # Now positioned at top-left instead of centered
            x_axis_header,
            y_axis_label,
            grid_positioned
        ] + x_value_clips + y_value_clips
        
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
            # Cleanup
            final_video.close()
            for clip in all_clips:
                clip.close()
            if pad_clip is not None:
                pad_clip.close()
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        
        # Optional: cleanup source video files
        if cleanup_files:
            for item in file_batch:
                try:
                    if os.path.exists(item['filepath']):
                        os.remove(item['filepath'])
                except Exception as e:
                    print(f"Warning: Failed to delete {item['filepath']}: {e}")
        
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
    "XYPlotGetValues": XYPlotGetValues,
    "XYPlotGetValuesTyped": XYPlotGetValuesTyped,
    "XYPlotCollectVideo": XYPlotCollectVideo,
    "XYPlotVideoGrid": XYPlotVideoGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XYPlotSetup": "XY Plot Setup",
    "XYPlotGetValues": "XY Plot Get Values",
    "XYPlotGetValuesTyped": "XY Plot Get Values (Typed)",
    "XYPlotCollectVideo": "XY Plot Collect Video",
    "XYPlotVideoGrid": "XY Plot Video Grid",
}