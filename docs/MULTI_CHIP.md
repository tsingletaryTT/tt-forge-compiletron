# Multi-Chip Architecture

How TT-Forge Compiletron scales from 1 to 32+ chips.

## Overview

Compiletron automatically detects hardware and distributes models across all available chips using round-robin scheduling. This ensures even load balancing and maximum throughput.

## Hardware Detection

### Automatic Detection

```bash
python3 compiletron.py detect
```

Uses `tt-smi` to detect:
- **Number of chips**: 1 to 32+ supported
- **Board type**: P300C, P150, N300, P100, etc.
- **Architecture**: Blackhole, Wormhole B0, Grayskull
- **PCI bus locations**: For device mapping

### Manual Override

Set environment variable:
```bash
export TT_METAL_DEVICE_COUNT=4
```

## Round-Robin Distribution

### Algorithm

With **K chips** and **N models**, chip `i` gets models:
```
i, i+K, i+2K, i+3K, ...
```

This ensures:
- **Even distribution**: Each chip gets ~N/K models
- **Load balancing**: Slow models spread across chips
- **No synchronization**: Chips work independently

### Examples

**4 chips, 108 models:**
```
Chip 0: models 0, 4, 8, 12, 16, ...  (27 models)
Chip 1: models 1, 5, 9, 13, 17, ...  (27 models)
Chip 2: models 2, 6, 10, 14, 18, ... (27 models)
Chip 3: models 3, 7, 11, 15, 19, ... (27 models)
```

**8 chips, 108 models:**
```
Chip 0: models 0, 8, 16, 24, ...  (14 models)
Chip 1: models 1, 9, 17, 25, ...  (14 models)
Chip 2: models 2, 10, 18, 26, ... (13 models)
...
Chip 7: models 7, 15, 23, 31, ... (13 models)
```

**16 chips, 108 models:**
```
Chip 0: models 0, 16, 32, 48, ... (7 models)
Chip 1: models 1, 17, 33, 49, ... (7 models)
...
Chip 15: models 15, 31, 47, ...   (7 models)
```

### Time Estimation

Sequential time (1 chip):
```
Total = sum(compile_time for all models)
```

Parallel time (K chips):
```
Total = max(sum(compile_time for chip_i) for i in 0..K-1)
```

**Example**: 50 models averaging 20s each
- 1 chip: 50 × 20s = 1000s (16.7 minutes)
- 4 chips: ~1000s / 4 = 250s (4.2 minutes)
- 8 chips: ~1000s / 8 = 125s (2.1 minutes)

## Chip Isolation

Each chip runs in isolated environment using `TT_VISIBLE_DEVICES`.

### Environment Variables Per Chip

```bash
# Chip 0
TT_VISIBLE_DEVICES=0
TT_METAL_ARCH_NAME=blackhole
TT_MESH_GRAPH_DESC_PATH=/path/to/p100_mesh_graph_descriptor.textproto

# Chip 1
TT_VISIBLE_DEVICES=1
TT_METAL_ARCH_NAME=blackhole
TT_MESH_GRAPH_DESC_PATH=/path/to/p100_mesh_graph_descriptor.textproto

# And so on...
```

### Why TT_VISIBLE_DEVICES?

- **Process isolation**: Each worker sees only its chip
- **No resource conflicts**: Exclusive chip access
- **Independent operation**: No inter-chip communication needed
- **Fault tolerance**: One chip failure doesn't affect others

### Why TT_MESH_GRAPH_DESC_PATH?

Required for **P300C** boards when running single-chip mode:
- P300C has complex mesh topology
- UMD needs descriptor to understand connectivity
- Without it: "Custom cluster type" errors
- Location: `$TT_METAL_HOME/build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/`

Not required for:
- P150 (simple single chip)
- N300 (dual chip, but different config)
- P100 (different topology)

## Parallel Execution

### Worker Processes

Each chip runs a separate Python process:
```bash
# Chip 0
TT_VISIBLE_DEVICES=0 python3 lib/worker.py 0 &

# Chip 1
TT_VISIBLE_DEVICES=1 python3 lib/worker.py 1 &

# Chip 2
TT_VISIBLE_DEVICES=2 python3 lib/worker.py 2 &

# Chip 3
TT_VISIBLE_DEVICES=3 python3 lib/worker.py 3 &

# Wait for all
wait
```

### Orchestrator Script

`scripts/run_parallel.sh` handles:
1. Hardware detection (how many chips?)
2. Model distribution calculation
3. Worker process spawning with proper env vars
4. Result aggregation
5. Cleanup on interrupt (Ctrl+C)

### Tmux Visualization

`scripts/view_logs.sh` creates auto-scaling layout:

**1 chip**: Full screen
```
┌──────────────────────┐
│                      │
│     Chip 0 log       │
│                      │
└──────────────────────┘
```

**4 chips**: 2×2 grid
```
┌──────────┬──────────┐
│ Chip 0   │ Chip 2   │
├──────────┼──────────┤
│ Chip 1   │ Chip 3   │
└──────────┴──────────┘
```

**8 chips**: 3×3 grid
```
┌──────┬──────┬──────┐
│ Ch 0 │ Ch 1 │ Ch 2 │
├──────┼──────┼──────┤
│ Ch 3 │ Ch 4 │ Ch 5 │
├──────┼──────┼──────┤
│ Ch 6 │ Ch 7 │(empty│
└──────┴──────┴──────┘
```

**16 chips**: 4×4 grid
```
┌────┬────┬────┬────┐
│ C0 │ C1 │ C2 │ C3 │
├────┼────┼────┼────┤
│ C4 │ C5 │ C6 │ C7 │
├────┼────┼────┼────┤
│ C8 │ C9 │C10 │C11 │
├────┼────┼────┼────┤
│C12 │C13 │C14 │C15 │
└────┴────┴────┴────┘
```

**32+ chips**: Multiple tmux windows (16 per window)

## Hardware Configurations

### Workstation (4 chips)
- **Board**: 2x P300C (dual-chip boards)
- **Total**: 4x Blackhole chips
- **Typical use**: Model development, testing
- **Compile time**: 50 models in ~5 minutes

### Server (8 chips)
- **Board**: 4x P300C or 8x P150
- **Total**: 8x Blackhole chips
- **Typical use**: Production inference
- **Compile time**: 100 models in ~5 minutes

### Cluster (16+ chips)
- **Board**: 8x P300C or custom configs
- **Total**: 16+ chips
- **Typical use**: Large-scale deployment
- **Compile time**: 100 models in ~3 minutes

## Best Practices

### 1. Model Ordering

Models are **pre-shuffled** with fixed seed (789) to distribute families evenly:
```python
# Without shuffle: Chip 0 gets all ResNets, Chip 1 gets all VGGs
# With shuffle: Each chip gets mix of families
```

This prevents one chip getting all slow models.

### 2. Staggered Startup

Workers use random 0-5s delay to avoid:
- Simultaneous Forge initialization
- Concurrent file access
- Race conditions in UMD

### 3. Results Aggregation

Each worker writes to shared CSV:
```csv
chip_id,model_name,success,compile_time
0,ResNet-18,True,8.2
1,VGG-16,True,2.4
2,EfficientNet-b0,True,8.5
3,DenseNet-121,True,42.3
```

Final report combines all chips.

### 4. Error Handling

- Chip failure → other chips continue
- Model failure → chip moves to next model
- Timeout → retry with exponential backoff
- SIGTERM → graceful shutdown, save partial results

## Troubleshooting

### Not All Chips Detected

**Check with tt-smi:**
```bash
tt-smi -s | jq '.devices | length'
```

**Common causes:**
- Firmware version mismatch
- PCIe initialization failed
- Board not properly seated

**Fix:**
```bash
# Reset all devices
tt-smi -r

# Check firmware
tt-smi -s | jq '.devices[].firmware_version'

# Reboot if needed
sudo reboot
```

### Chip Assignment Errors

**Error**: `TT_VISIBLE_DEVICES=4 but only 4 devices`

**Cause**: 0-indexed, so valid IDs are 0-3 for 4 chips

**Fix**: Use `0..N-1` not `1..N`

### Mesh Descriptor Not Found

**Error**: `TT_MESH_GRAPH_DESC_PATH not found`

**Check locations:**
```bash
find $TT_METAL_HOME -name "*.textproto"
```

**Set manually:**
```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
```

### Uneven Load Distribution

**Symptoms**: Some chips finish much earlier than others

**Cause**: Model ordering not shuffled

**Fix**: Ensure MODEL_LIST uses `random.seed(789)` shuffle

## Performance Metrics

### Scaling Efficiency

Ideal speedup: `S = N` (N chips = N× faster)

Real speedup: `S = T₁ / Tₙ`

Efficiency: `E = S / N = (T₁ / Tₙ) / N`

**Typical results**:
- 4 chips: ~90% efficiency (3.6× speedup)
- 8 chips: ~85% efficiency (6.8× speedup)
- 16 chips: ~80% efficiency (12.8× speedup)

Efficiency decreases due to:
- Uneven model distribution (108 models not divisible by all N)
- Varying model complexity
- Startup/shutdown overhead

### Optimization Tips

1. **Choose model count divisible by chip count**
   - 100 models ÷ 4 chips = 25 each (perfect)
   - 108 models ÷ 4 chips = 27, 27, 27, 27 (good)
   - 100 models ÷ 6 chips = 17, 17, 17, 17, 16, 16 (okay)

2. **Sort by complexity before distribution**
   - Slow models first, then fast models
   - Or: alternate slow/fast for balance

3. **Use tmux for monitoring**
   - Spot idle chips
   - Identify bottlenecks
   - Real-time progress tracking

## See Also

- [Forge Setup](FORGE_SETUP.md) - Installation guide
- [Model Library](MODEL_LIBRARY.md) - Available models
- [Main README](../README.md) - Usage guide
