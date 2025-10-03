# XY Video Plot - Complete Connection Guide

## Two Data Flows

### 1. xy_config (Metadata - Pass-Through)
**Purpose**: Grid configuration (axis names, types, dimensions)
**Pattern**: Direct connections, NOT through For Loop ports

```
XYPlotSetup.xy_config 
  → XYPlotGetValues.xy_config (input)
  → XYPlotGetValues.xy_config (output) 
  → XYPlotCollectVideo.xy_config (input)
  → XYPlotCollectVideo.xy_config (output)
  → XYPlotVideoGrid.xy_config (input)
```

**Visual:**
```
Setup → GetValues → CollectVideo → VideoGrid
        (xy_config passes through each node)
```

---

### 2. collection_data (Accumulator - Loop Feedback)
**Purpose**: Accumulate video filepaths across iterations
**Pattern**: Through For Loop Start/End value1 ports

```
Step 1: For Loop Start
  - Set initial_value1 = "[]"
  - Output value1 connects to...

Step 2: XYPlotCollectVideo
  - Input: collection_data (from For Loop Start.value1)
  - Appends new video info
  - Output: collection_data (to For Loop End.value1)

Step 3: For Loop End
  - Input: value1 (from CollectVideo.collection_data)
  - Loops back internally
  - After loop completes: final_value1 connects to...

Step 4: XYPlotVideoGrid
  - Input: collection_data (from For Loop End.final_value1)
```

**Visual:**
```
Loop Start (initial_value1="[]")
    ↓ value1
CollectVideo (receives, appends, outputs)
    ↓ collection_data
Loop End (value1 input, loops back)
    ↓ final_value1 (after loop)
VideoGrid
```

---

## Complete Connection Diagram

```
┌──────────────────┐
│  XYPlotSetup     │
└────┬─────┬───┬───┘
     │     │   │
     │total│xy │initial_collection_data
     │iter │cfg│
     ↓     │   ↓
┌─────────┴───────────────┐
│  For Loop Start         │
│  initial_value1: ───────┘ (from Setup)
└───┬──────────────┬──────┘
    │loop_index    │value1
    │              │
    ↓              ↓                 
┌──────────────────────────────────────┐
│  XYPlotGetValues                     │
│  (loop_index, xy_config)             │
└─┬──┬───────┬───────────────┬────────┘
  │  │       │               │
  x  y    labels           xy_config (passes through)
  │  │       │               │
  ↓  ↓       ↓               │
┌──────────────┐             │
│ Your Workflow │             │
│ (KSampler,    │             │
│  VHS, etc.)   │             │
└──────┬────────┘             │
       │video_path            │
       ↓                      ↓
┌──────────────────────────────────────┐
│  XYPlotCollectVideo                  │
│  (video_path, labels, xy_config,     │
│   collection_data from loop)         │
└───────────────┬───────────┬──────────┘
                │           │
     collection_data     xy_config (passes through)
                │           │
                ↓           │
┌─────────────────────────┐ │
│  For Loop End           │ │
│  value1 (loops back)    │ │
└──────────┬──────────────┘ │
           │final_value1    │
           │(after loop)    │
           ↓                ↓
┌──────────────────────────────────────┐
│  XYPlotVideoGrid                     │
│  (collection_data, xy_config)        │
└──────────────────────────────────────┘
```

---

## Step-by-Step Connections

### Part 1: Setup & Loop Initialization

**1. XYPlotSetup Node**
- Configure axes and values
- **Input Order (Organized):**
  - `x_axis_name`: e.g., "seed"
  - `x_axis_type`: INT, FLOAT, or STRING
  - `x_values`: e.g., "1, 2, 3"
  - `y_axis_name`: e.g., "cfg"
  - `y_axis_type`: INT, FLOAT, or STRING
  - `y_values`: e.g., "7.0, 8.0, 9.0"
- **Outputs:**
  - `total_iterations` → **For Loop Start.iterations**
  - `xy_config` → **Save this wire** (connect through workflow)
  - `initial_collection_data` → **For Loop Start.initial_value1**

**2. For Loop Start Node**
- `iterations`: Connect from XYPlotSetup.total_iterations
- `initial_value1`: Connect from XYPlotSetup.initial_collection_data ✨ (No manual typing needed!)
- **Outputs:**
  - `loop_index` → XYPlotGetValues.loop_index
  - `value1` → XYPlotCollectVideo.collection_data

---

### Part 2: Inside Loop

**3. XYPlotGetValues Node**
- `xy_config`: Connect from XYPlotSetup.xy_config
- `loop_index`: Connect from For Loop Start.loop_index
- **Outputs:**
  - `x_value` → Your parameter (e.g., KSampler.seed)
  - `y_value` → Your parameter (e.g., KSampler.cfg)
  - `x_label` → XYPlotCollectVideo.x_label
  - `y_label` → XYPlotCollectVideo.y_label
  - `xy_config` → XYPlotCollectVideo.xy_config

**4. Your Workflow** (KSampler, AnimateDiff, etc.)
- Use x_value and y_value as parameters
- Generate video → VHS Video Combine
- **Output:** `filenames` → XYPlotCollectVideo.video_filepath

**5. XYPlotCollectVideo Node**
- `video_filepath`: From VHS Video Combine.filenames
- `x_label`: From XYPlotGetValues.x_label
- `y_label`: From XYPlotGetValues.y_label
- `xy_config`: From XYPlotGetValues.xy_config
- `collection_data`: From For Loop Start.value1
- **Outputs:**
  - `collection_data` → **For Loop End.value1**
  - `xy_config` → Save for VideoGrid

**6. For Loop End Node**
- `value1`: Connect from XYPlotCollectVideo.collection_data
- **Output (after loop completes):**
  - `final_value1` → XYPlotVideoGrid.collection_data

---

### Part 3: After Loop

**7. XYPlotVideoGrid Node**
- `collection_data`: From For Loop End.final_value1
- `xy_config`: From XYPlotCollectVideo.xy_config
- Configure styling parameters
- **Outputs:**
  - `video_grid` → Preview or Save
  - `metadata` → Optional text display

---

## Key Points

### ✅ xy_config:
- Direct wire through nodes
- Does NOT go through For Loop value ports
- Just metadata, doesn't change

### ✅ collection_data:
- Goes through For Loop value1 ports
- Accumulates data each iteration
- Empty at start: `"[]"`
- Grows each loop: `[{video1}, {video2}, ...]`
- Complete at end → VideoGrid

### ⚠️ Common Mistakes:
1. ~~**Forgetting** `initial_value1 = "[]"` in For Loop Start~~ ✨ **FIXED! Now automatic from Setup**
2. **Connecting** xy_config to loop value ports (don't do this!)
3. **Not connecting** collection_data through value1 ports
4. **Wrong port:** Using value2 instead of value1

---

## Testing Checklist

- [ ] XYPlotSetup.total_iterations → For Loop Start.iterations
- [ ] XYPlotSetup.initial_collection_data → For Loop Start.initial_value1 ✨
- [ ] For Loop Start.value1 → XYPlotCollectVideo.collection_data
- [ ] XYPlotCollectVideo.collection_data → For Loop End.value1
- [ ] For Loop End.final_value1 → XYPlotVideoGrid.collection_data
- [ ] xy_config wire: Setup → GetValues → CollectVideo → VideoGrid
- [ ] All labels connected: x_label, y_label
- [ ] Video path connected from VHS

---

## If Dynamic Typing Doesn't Work

Replace `XYPlotGetValues` with `XYPlotGetValuesTyped`:
- Same connections
- But use `x_int` or `x_float` instead of `x_value`
- More outputs but guaranteed type safety

