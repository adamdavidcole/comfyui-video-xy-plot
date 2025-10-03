# ComfyUI XY Video Plot Suite

A suite of ComfyUI nodes for generating parameter sweep video matrices with labeled grids. Create comprehensive visual comparisons of how different parameter combinations affect your video generation results.

## Overview

This package provides 4 nodes that work together with `comfyui-easy-use` For Loop nodes to:
1. Define X/Y parameter sweep ranges
2. Iterate through all combinations
3. Collect generated videos
4. Assemble them into a labeled comparison grid

## Installation

1. Install moviepy:
```bash
pip install moviepy
```

2. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-xyvideoplot2
```

3. Restart ComfyUI

## Nodes

### 1. XY Plot Setup
**Purpose**: Initialize the parameter sweep configuration

**Inputs**:
- `x_axis_name`: Name for X axis (e.g., "seed")
- `y_axis_name`: Name for Y axis (e.g., "cfg")
- `x_values`: Comma-separated X values (e.g., "1, 2, 3")
- `y_values`: Comma-separated Y values (e.g., "7.0, 8.0, 9.0")

**Outputs**:
- `total_iterations`: Total number of iterations (feeds into For Loop)
- `xy_config`: Configuration JSON (pass through workflow)

### 2. XY Plot Get Values
**Purpose**: Get current X/Y values for the loop iteration

**Inputs**:
- `xy_config`: From XY Plot Setup
- `loop_index`: From For Loop Start

**Outputs**:
- `x_value`: Current X value (STRING, auto-converts to INT/FLOAT)
- `y_value`: Current Y value (STRING, auto-converts to INT/FLOAT)
- `x_label`: Display label for X
- `y_label`: Display label for Y
- `xy_config`: Pass-through

### 3. XY Plot Collect Video
**Purpose**: Collect video filepath with labels during loop

**Inputs**:
- `video_filepath`: Path to generated video (from VHS Video Combine)
- `x_label`: From XY Plot Get Values
- `y_label`: From XY Plot Get Values
- `xy_config`: Pass-through
- `collection_data`: From For Loop Start value1 (optional)

**Outputs**:
- `collection_data`: Updated collection (to For Loop End value1)
- `xy_config`: Pass-through

### 4. XY Plot Video Grid
**Purpose**: Assemble final video grid after loop completes

**Inputs**:
- `collection_data`: From For Loop End final_value1
- `xy_config`: Pass-through
- `thumb_width`: Width of each video cell (default: 480)
- `thumb_height`: Height of each video cell (default: 270)
- `banner_font_size`: Title banner font size (default: 20)
- `banner_font_color`: Font color (default: "white")
- `banner_bg_color`: Background color (default: "black")
- `cleanup_files`: Delete source videos after assembly (default: True)

**Outputs**:
- `video_grid`: Assembled grid as IMAGE tensor
- `metadata`: Grid information as JSON

## Workflow Structure

```
┌─────────────────────┐
│  XY Plot Setup      │
│  x: "1,2,3"         │
│  y: "7.0,8.0"       │
└─────┬───────────┬───┘
      │           │
      │ total     │ xy_config
      │ iter: 6   │
      ↓           ↓
┌─────────────────────────────────┐
│  For Loop Start                 │
│  initial_value1 = "[]"          │
└────┬────────────────┬───────────┘
     │ loop_index     │ value1
     ↓                │
┌─────────────────────┴──────┐
│  XY Plot Get Values        │
│  (loop_index, xy_config)   │
└─┬──┬──────┬────────────────┘
  │  │      │
  │  │      └─ x_label, y_label
  │  └──────── y_value → KSampler cfg
  └─────────── x_value → KSampler seed
              │
         ┌────▼──────────┐
         │  Your Workflow │
         │  (KSampler,    │
         │   AnimateDiff, │
         │   etc.)        │
         └────┬───────────┘
              │
         ┌────▼──────────────┐
         │  VHS Video Combine │
         │  (save video)      │
         └────┬───────────────┘
              │ video_filepath
              ↓
┌──────────────────────────────┐
│  XY Plot Collect Video       │
│  (accumulate data)           │
└────┬─────────────────────────┘
     │ collection_data
     ↓
┌──────────────────────────────┐
│  For Loop End                │
│  value1 → loops back         │
└────┬─────────────────────────┘
     │ final_value1 (after loop)
     ↓
┌──────────────────────────────┐
│  XY Plot Video Grid          │
│  (assemble final grid)       │
└────┬─────────────────────────┘
     │
     └─→ video_grid (IMAGE)
```

## Example Workflow

**Goal**: Compare different seeds (X) vs cfg values (Y)

1. **XY Plot Setup**:
   - x_axis_name: "seed"
   - y_axis_name: "cfg"
   - x_values: "123, 456, 789"
   - y_values: "7.0, 8.5, 10.0"
   - Output: 9 total iterations

2. **For Loop Start**:
   - Connect `total_iterations` to loop count
   - Set `initial_value1` = "[]"

3. **XY Plot Get Values**:
   - Connect `loop_index` from For Loop
   - Connect `x_value` to KSampler seed
   - Connect `y_value` to KSampler cfg

4. **Your Generation Workflow**:
   - KSampler → AnimateDiff → VHS Video Combine
   - Save video to file

5. **XY Plot Collect Video**:
   - Connect `video_filepath` from VHS
   - Connect `x_label`, `y_label` from Get Values
   - Connect `collection_data` from For Loop Start `value1`

6. **For Loop End**:
   - Connect `collection_data` to `value1`

7. **XY Plot Video Grid** (after loop):
   - Connect `final_value1` from For Loop End
   - Adjust styling parameters
   - Output: 3x3 grid of videos with labels

## Tips

### Type Conversion
- X/Y values are output as STRINGs
- ComfyUI automatically converts "42" → 42 (INT) and "7.5" → 7.5 (FLOAT)
- Works seamlessly with most node inputs

### Performance
- Each video cell is resized to `thumb_width × thumb_height`
- Smaller thumbnails = faster processing
- Consider using `cleanup_files=True` to save disk space

### Label Formatting
- Labels automatically include axis name: "seed: 123"
- Banner shows axis names: "X-Axis: seed | Y-Axis: cfg"

### Future Expansion
The architecture is designed for easy extension:
- Array inputs (IMAGE[], LATENT[])
- Multi-dimensional sweeps (3+ parameters)
- Reference image comparisons
- Custom label templates

## Troubleshooting

**"No videos collected" error**:
- Ensure VHS Video Combine is saving files correctly
- Check that `video_filepath` is connected

**Grid positioning incorrect**:
- Values are sorted alphabetically for consistent layout
- Check that X/Y values are unique

**Missing videos in grid**:
- Black placeholders show where videos are missing
- Check console for file-not-found warnings

## Requirements

- ComfyUI
- moviepy
- comfyui-easy-use (for For Loop nodes)
- VHS Video Combine (for video generation)

## License

MIT License - See LICENSE file for details

## Credits

Designed for the ComfyUI community to make parameter exploration more visual and intuitive.
