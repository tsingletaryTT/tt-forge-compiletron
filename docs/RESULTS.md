# TT-Forge Compilation Results Template

This template provides a standard format for documenting TT-Forge compilation results. Copy and fill in your specific results.

## Test Run Information

**Date:** YYYY-MM-DD HH:MM:SS
**Duration:** X hours Y minutes
**Operator:** [Your Name/Team]
**Environment:** [Development/Testing/Production]

## Hardware Configuration

**Board Type:** [P300C/P150/N300/P100]
**Architecture:** [Blackhole/Wormhole/Grayskull]
**Number of Chips:** [1/2/4/8/16+]
**Total Memory:** XXX GB
**Firmware Version:** X.X.X.X
**KMD Version:** X.X.X

**Hardware Details:**
```bash
# Output from: tt-smi -s
[Paste tt-smi output here]
```

## Software Versions

**TT-Metal:**
- **Repository:** [GitHub URL or local path]
- **Branch/Commit:** [commit hash or tag]
- **Build Date:** YYYY-MM-DD

**TT-Forge:**
- **Repository:** [GitHub URL or local path]
- **Branch/Commit:** [commit hash or tag]
- **Build Date:** YYYY-MM-DD

**Python Environment:**
- **Python Version:** 3.XX
- **PyTorch Version:** X.XX.X
- **TorchVision Version:** X.XX.X

**Key Dependencies:**
```bash
# Output from: pip list | grep -E "(torch|forge|metal)"
[Paste dependency versions here]
```

## Compilation Summary

### Overall Statistics

- **Total Models Tested:** XXX
- **Successful Compilations:** XXX (XX.X%)
- **Failed Compilations:** XXX (XX.X%)
- **Total Compilation Time:** XX hours XX minutes (XXXXX seconds)
- **Average Compilation Time:** XX.X seconds per model
- **Min Compilation Time:** X.X seconds ([Model Name])
- **Max Compilation Time:** XXX.X seconds ([Model Name])

### By Complexity

| Complexity | Total | Success | Failed | Success Rate | Avg Time |
|------------|-------|---------|--------|--------------|----------|
| Low        | XX    | XX      | X      | XX.X%        | XX.X s   |
| Medium     | XX    | XX      | X      | XX.X%        | XX.X s   |
| High       | XX    | XX      | X      | XX.X%        | XX.X s   |

### By Model Family

| Family          | Models | Success | Failed | Success Rate | Total Time |
|-----------------|--------|---------|--------|--------------|------------|
| ResNet          | X      | X       | X      | XX.X%        | XXX s      |
| VGG             | X      | X       | X      | XX.X%        | XXX s      |
| EfficientNet    | X      | X       | X      | XX.X%        | XXX s      |
| DenseNet        | X      | X       | X      | XX.X%        | XXX s      |
| MobileNet       | X      | X       | X      | XX.X%        | XXX s      |
| [Add more families as needed]                                           |

## Successful Compilations

### Top 10 Fastest Models

| Rank | Model | Time (s) | Complexity | Parameters |
|------|-------|----------|------------|------------|
| 1    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 2    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 3    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 4    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 5    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 6    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 7    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 8    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 9    | [Model Name] | XX.X | [low/medium/high] | XX.XM |
| 10   | [Model Name] | XX.X | [low/medium/high] | XX.XM |

### Top 10 Slowest Models

| Rank | Model | Time (s) | Complexity | Parameters |
|------|-------|----------|------------|------------|
| 1    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 2    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 3    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 4    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 5    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 6    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 7    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 8    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 9    | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |
| 10   | [Model Name] | XXX.X | [low/medium/high] | XXX.XM |

### Complete Success List

[Optional: Include full list of successful models or reference CSV file]

```
See attached: results_YYYYMMDD_HHMMSS.csv
```

## Failed Compilations

### Failure Summary

**Total Failures:** XX models (XX.X%)

### Failed Models by Family

| Family | Failed Count | Models |
|--------|--------------|--------|
| [Family Name] | X | [Model1, Model2, ...] |
| [Family Name] | X | [Model1, Model2, ...] |

### Failure Analysis

#### Common Error Types

1. **[Error Type 1]** (XX models)
   - Models: [Model1, Model2, ...]
   - Possible cause: [Description]
   - Workaround: [If any]

2. **[Error Type 2]** (XX models)
   - Models: [Model1, Model2, ...]
   - Possible cause: [Description]
   - Workaround: [If any]

#### Detailed Failure List

| Model | Family | Error Type | Notes |
|-------|--------|------------|-------|
| [Model Name] | [Family] | [Error Type] | [Any relevant notes] |
| [Model Name] | [Family] | [Error Type] | [Any relevant notes] |

## Parallel Execution Details

[If using multi-chip parallel execution]

**Execution Mode:** Parallel (X chips)
**Distribution:** Round-robin

### Per-Chip Statistics

| Chip | Models | Success | Failed | Total Time | Avg Time |
|------|--------|---------|--------|------------|----------|
| 0    | XX     | XX      | X      | XXX s      | XX.X s   |
| 1    | XX     | XX      | X      | XXX s      | XX.X s   |
| 2    | XX     | XX      | X      | XXX s      | XX.X s   |
| 3    | XX     | XX      | X      | XXX s      | XX.X s   |

**Load Balancing:**
- Well balanced: ✅/❌
- Max difference: XX models between chips
- Notes: [Any observations about distribution]

## Performance Analysis

### Compilation Time Distribution

```
0-5s:     XXXX models  (XX.X%)  ████████████████
5-10s:    XX models    (XX.X%)  ████████
10-20s:   XX models    (XX.X%)  ██████
20-50s:   XX models    (XX.X%)  ████
50-100s:  X models     (XX.X%)  ██
100s+:    X models     (XX.X%)  █
```

### Model Size vs Compilation Time

| Size Range | Count | Avg Time | Notes |
|------------|-------|----------|-------|
| < 10M params    | XX | XX.X s | Small models |
| 10-25M params   | XX | XX.X s | Medium models |
| 25-50M params   | XX | XX.X s | Large models |
| 50-100M params  | XX | XX.X s | Very large models |
| 100M+ params    | X  | XXX.X s | Huge models |

### Expected vs Actual Times

**Models faster than expected:** XX (XX.X%)
**Models within expected range (±20%):** XX (XX.X%)
**Models slower than expected:** XX (XX.X%)

**Significant deviations:**
- [Model Name]: Expected XX.Xs, Actual XXX.Xs (XXX% slower/faster)
- [Model Name]: Expected XX.Xs, Actual XXX.Xs (XXX% slower/faster)

## Observations and Notes

### Positive Findings

- [Notable success or performance achievement]
- [Another positive observation]
- [Any improvements over previous runs]

### Issues Encountered

- [Issue description and impact]
- [Another issue]
- [Workarounds applied]

### Recommendations

- [Recommended action based on results]
- [Configuration suggestion]
- [Follow-up work needed]

## Comparison with Previous Runs

[If applicable]

| Metric | Previous Run | This Run | Change |
|--------|--------------|----------|--------|
| Success Rate | XX.X% | XX.X% | +/-X.X% |
| Avg Time | XX.Xs | XX.Xs | +/-X.Xs |
| Total Models | XXX | XXX | +/-XX |

**Notable Changes:**
- [Description of significant changes]
- [Explanation of improvements or regressions]

## Reproducibility

### Commands Used

```bash
# Hardware detection
python3 compiletron.py detect

# Compilation command
[Exact command used]

# Results generation
python3 compiletron.py results report --output this_report.md
```

### Configuration Files

[If any custom configurations were used]

```yaml
# config.yaml or relevant settings
[Configuration content]
```

## Attachments

- **Raw Results CSV:** `results_YYYYMMDD_HHMMSS.csv`
- **Log Files:** [Path to logs if saved]
- **Screenshots:** [If any tmux screenshots or visualizations]

## Conclusion

[Summary paragraph of overall results, key takeaways, and next steps]

---

**Report Generated:** YYYY-MM-DD HH:MM:SS
**Generated By:** [Name/Tool]
**Template Version:** 1.0

## Usage Instructions

To generate a report from compiletron results:

```bash
# Run compilation
python3 compiletron.py run --quick

# Generate automated report
python3 compiletron.py results report --output auto_report.md

# Then fill in this template with additional details
```

This template can be customized based on your specific needs. Remove sections that aren't applicable and add any additional sections relevant to your testing.
