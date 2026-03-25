# Example Workflows

This directory contains example scripts demonstrating common TT-Forge Compiletron workflows.

## Prerequisites

All examples require:
1. **Forge environment activated:**
   ```bash
   source ~/tt-forge-fe/env/activate
   ```

2. **Tenstorrent hardware available** (1+ devices)

3. **tt-forge-compiletron in your current directory**

## Quick Start

```bash
cd ~/code/tt-forge-compiletron

# Run quickest example (5 models, ~20 seconds)
./scripts/examples/example-quick-test.sh
```

## Examples

### 1. Quick Test (`example-quick-test.sh`)

**What it does:** Compiles 5 fastest models on a single chip

**Use case:** Validate Forge installation and hardware setup

**Time:** ~15-20 seconds
**Hardware:** Any (1+ chip)

```bash
./scripts/examples/example-quick-test.sh
```

**Output:**
- Compilation results for 5 models
- Success/failure stats
- Performance metrics

---

### 2. Parallel Sweep (`example-parallel-sweep.sh`)

**What it does:** Compiles 100 models distributed across all available chips

**Use case:** Large-scale testing with parallel execution

**Time:** ~30 min (4 chips), ~60 min (2 chips)
**Hardware:** 2+ chips recommended

```bash
./scripts/examples/example-parallel-sweep.sh
```

**Output:**
- Round-robin distribution across chips
- Real-time progress from all chips
- Aggregated results

**Features:**
- Automatic hardware detection
- Time estimation
- Confirmation prompt
- Results aggregation

---

### 3. Family Compilation (`example-family-compilation.sh`)

**What it does:** Compiles all models in a specific family (e.g., all ResNets)

**Use case:** Systematic testing of architecture variants

**Time:** Varies by family
- ResNet: ~2 minutes (6 models)
- EfficientNet: ~15 minutes (8 models)
- VGG: ~10 minutes (12 models)

**Hardware:** Any (1+ chip)

```bash
# Compile all ResNet models
./scripts/examples/example-family-compilation.sh resnet

# Compile all EfficientNet models
./scripts/examples/example-family-compilation.sh efficientnet

# Compile all VGG models
./scripts/examples/example-family-compilation.sh vgg
```

**Available families:**
```bash
python3 compiletron.py models families
```

---

### 4. Resume Interrupted Compilation (`example-resume.sh`)

**What it does:** Shows how to resume after interruption (Ctrl+C, crash, etc.)

**Use case:** Recovering from failed or interrupted compilation runs

**Time:** Depends on remaining models

**Hardware:** Any (1+ chip)

```bash
./scripts/examples/example-resume.sh
```

**Features:**
- Analyzes previous results
- Shows failed models
- Suggests retry strategies
- Provides manual resume instructions

**Resume strategies:**
1. **Re-run specific families:** `python3 compiletron.py run --family resnet`
2. **Re-run complexity level:** `python3 compiletron.py run --complexity low`
3. **Manual model selection:** Edit model list and re-run

---

## Workflow Patterns

### Pattern 1: Development Testing
```bash
# 1. Quick validation
./scripts/examples/example-quick-test.sh

# 2. Single family test
./scripts/examples/example-family-compilation.sh resnet

# 3. View results
python3 compiletron.py results
```

### Pattern 2: Comprehensive Testing
```bash
# 1. Parallel sweep (100 models)
./scripts/examples/example-parallel-sweep.sh

# 2. Generate report
python3 compiletron.py results report --output full_report.md

# 3. Export CSV
python3 compiletron.py results export --output full_results.csv
```

### Pattern 3: Debugging Failed Models
```bash
# 1. Run quick test to establish baseline
./scripts/examples/example-quick-test.sh

# 2. Run family that's failing
./scripts/examples/example-family-compilation.sh efficientnet

# 3. Check specific model
python3 compiletron.py models info EfficientNet-B0

# 4. Retry with different settings
python3 compiletron.py run --family efficientnet --chip 1
```

## Customization

### Create Your Own Examples

1. Copy an existing script:
```bash
cp scripts/examples/example-quick-test.sh scripts/examples/my-workflow.sh
```

2. Modify the `python3 compiletron.py run` command:
```bash
# Example: Compile 20 low-complexity models
python3 compiletron.py run --complexity low --count 20 --chip 0
```

3. Make executable:
```bash
chmod +x scripts/examples/my-workflow.sh
```

### Common Customizations

**Filter by complexity:**
```bash
python3 compiletron.py run --complexity low --count 30
```

**Specific chip:**
```bash
python3 compiletron.py run --quick --chip 2
```

**Multiple families:**
```bash
# Run ResNets, then VGGs
python3 compiletron.py run --family resnet --chip 0
python3 compiletron.py run --family vgg --chip 0
```

## Tips

1. **Always activate Forge first:** The examples check for this and will fail gracefully if not activated

2. **Monitor progress:** Use `watch` to monitor results:
   ```bash
   watch -n 5 'python3 compiletron.py results'
   ```

3. **Parallel execution:** For multi-chip systems, parallel mode is much faster:
   ```bash
   # 100 models, 4 chips → ~30 minutes
   # 100 models, 1 chip → ~120 minutes
   ```

4. **Tmux for parallel viewing:** Use tmux to watch all chips simultaneously:
   ```bash
   bash scripts/view_logs.sh
   ```

5. **Results persistence:** Results are saved to `results/results_YYYYMMDD_HHMMSS.csv` and persist between runs

## Troubleshooting

**"Forge environment not activated"**
```bash
source ~/tt-forge-fe/env/activate
```

**"No devices found"**
```bash
# Check hardware
tt-smi

# Reset devices
tt-smi -r
```

**"Cannot import forge module"**
```bash
# Check Forge installation
python3 compiletron.py setup check
```

**"Compilation failed"**
- Try `tt-smi -r` to reset devices
- Check specific model: `python3 compiletron.py models info <model>`
- Try different chip: `--chip 1` instead of `--chip 0`

## Next Steps

After running examples:
1. **View detailed results:** `python3 compiletron.py results view -v`
2. **Generate report:** `python3 compiletron.py results report`
3. **Explore model library:** `python3 compiletron.py models families`
4. **Create custom workflows:** Copy and modify example scripts

## Support

- **Main README:** `../../README.md`
- **Documentation:** `../../docs/`
- **Command help:** `python3 compiletron.py --help`
