# Quick Start Guide - XY Video Plot Suite

## Minimal Working Example

### Step-by-Step Connection Guide

#### 1. XY Plot Setup Node
```
Inputs:
  x_axis_name: "seed"
  y_axis_name: "cfg" 
  x_values: "123, 456, 789"
  y_values: "7.0, 8.5"

Outputs:
  total_iterations → For Loop Start.iterations
  xy_config → save for later
```

#### 2. For Loop Start (comfyui-easy-use)
```
Inputs:
  iterations: [from XY Plot Setup.total_iterations]
  initial_value1: "[]"  ← Type this manually!

Outputs:
  loop_index → XY Plot Get Values.loop_index
  value1 → XY Plot Collect Video.collection_data
```

#### 3. XY Plot Get Values Node
```
Inputs:
  xy_config: [from XY Plot Setup]
  loop_index: [from For Loop Start]

Outputs:
  x_value → KSampler.seed (or your parameter)
  y_value → KSampler.cfg (or your parameter)
  x_label → XY Plot Collect Video.x_label
  y_label → XY Plot Collect Video.y_label
  xy_config → XY Plot Collect Video.xy_config
```

#### 4. Your Generation Workflow
```
Example chain:
  KSampler 
    → VAE Decode 
    → AnimateDiff 
    → VHS Video Combine

VHS Video Combine outputs:
  filenames → XY Plot Collect Video.video_filepath
```

#### 5. XY Plot Collect Video Node
```
Inputs:
  video_filepath: [from VHS Video Combine.filenames]
  x_label: [from XY Plot Get Values]
  y_label: [from XY Plot Get Values]
  xy_config: [from XY Plot Get Values]
  collection_data: [from For Loop Start.value1]

Outputs:
  collection_data → For Loop End.value1
  xy_config → save for grid node
```

#### 6. For Loop End (comfyui-easy-use)
```
Inputs:
  value1: [from XY Plot Collect Video.collection_data]

Outputs:
  final_value1 → XY Plot Video Grid.collection_data
```

#### 7. XY Plot Video Grid Node
```
Inputs:
  collection_data: [from For Loop End.final_value1]
  xy_config: [from XY Plot Collect Video]
  thumb_width: 480 (adjust as needed)
  thumb_height: 270 (adjust as needed)
  banner_font_size: 20
  banner_font_color: "white"
  banner_bg_color: "black"
  cleanup_files: true

Outputs:
  video_grid → Preview Video / Save Video
  metadata → (optional) View Text
```

## Critical Connection Points

### ✅ MUST Connect:
1. `total_iterations` → For Loop count
2. `loop_index` → Get Values
3. `x_value` / `y_value` → Your parameters (KSampler, etc.)
4. VHS output → `video_filepath`
5. Loop Start `value1` → Collect Video `collection_data`
6. Collect Video `collection_data` → Loop End `value1`
7. Loop End `final_value1` → Video Grid `collection_data`

### ⚠️ Common Mistakes:
- Forgetting to set `initial_value1 = "[]"` in For Loop Start
- Not connecting both `x_label` and `y_label`
- Missing the `xy_config` pass-through chain
- Connecting to wrong For Loop ports (use value1, not value2)

## Testing Your Setup

### Test with Small Grid First:
```
x_values: "1, 2"
y_values: "7.0, 8.0"
Total iterations: 4 (2x2 grid)
```

### Verify Loop is Working:
- Check console for video generation messages
- Ensure 4 videos are created before grid assembly
- Watch for "Video file not found" warnings

### Expected Timeline:
```
1. Setup runs once
2. Loop executes 4 times:
   - Get Values (iteration 0): x=1, y=7.0
   - Generate video
   - Collect video
   - Get Values (iteration 1): x=2, y=7.0
   - Generate video
   - Collect video
   - Get Values (iteration 2): x=1, y=8.0
   - Generate video
   - Collect video
   - Get Values (iteration 3): x=2, y=8.0
   - Generate video
   - Collect video
3. Loop ends
4. Video Grid assembles final output
```

## Parameter Type Examples

### Integer Parameters (seed, steps, etc.):
```
x_values: "1, 2, 3, 4"
→ ComfyUI auto-converts to: 1, 2, 3, 4
```

### Float Parameters (cfg, denoise, etc.):
```
y_values: "7.0, 7.5, 8.0, 8.5"
→ ComfyUI auto-converts to: 7.0, 7.5, 8.0, 8.5
```

### String Parameters (sampler_name, scheduler):
```
x_values: "euler, dpmpp_2m, uni_pc"
→ Stays as strings
```

## Grid Layout

Grid is organized as:
- **Rows** = Y values (top to bottom)
- **Columns** = X values (left to right)

Example 3x2 grid:
```
                X: 1    X: 2    X: 3
Y: 7.0      [video] [video] [video]
Y: 8.0      [video] [video] [video]
```

## Advanced: Multiple Parameter Types

You can sweep different parameter types simultaneously:

```
x_axis_name: "seed"
x_values: "123, 456, 789"  ← INT

y_axis_name: "cfg"
y_values: "7.0, 8.5, 10.0"  ← FLOAT

Connect:
  x_value → KSampler.seed (auto-converts to INT)
  y_value → KSampler.cfg (auto-converts to FLOAT)
```

## Troubleshooting Checklist

- [ ] moviepy installed (`pip install moviepy`)
- [ ] comfyui-easy-use For Loop nodes installed
- [ ] VHS Video Combine node available
- [ ] For Loop Start `initial_value1` set to `"[]"`
- [ ] All connections match the guide above
- [ ] Test with 2x2 grid first
- [ ] Check console for error messages
- [ ] Verify video files are being created during loop

## Need Help?

Common issues:
1. **"No videos collected"** → Check VHS Video Combine is saving files
2. **"loop_index out of range"** → Verify For Loop iterations = total_iterations
3. **Empty grid** → Ensure collection_data flows through Loop Start value1
4. **Missing labels** → Connect x_label and y_label from Get Values

For more details, see README.md
