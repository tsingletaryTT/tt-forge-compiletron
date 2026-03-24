# TT-Forge Compiletron Tests

Comprehensive test suite for hardware detection and model distribution.

## Running Tests

### Quick Start

```bash
./run_tests.sh
```

### Run Specific Test File

```bash
python3 -m pytest tests/test_hardware.py -v -p no:asyncio
```

### Run Specific Test Class

```bash
python3 -m pytest tests/test_hardware.py::TestDetectHardware -v -p no:asyncio
```

### Run Specific Test

```bash
python3 -m pytest tests/test_hardware.py::TestDetectHardware::test_4_chips_p300c_detected -v -p no:asyncio
```

## Test Coverage

### Hardware Detection Tests (9 tests)

Tests for `detect_hardware()` function:

- ✅ **test_no_devices_detected** - No Tenstorrent devices present
- ✅ **test_single_p150_detected** - Single P150 chip
- ✅ **test_4_chips_p300c_detected** - 4 chips (2x P300C boards)
- ✅ **test_8_chips_n300_detected** - 8 chips (N300 configuration)
- ✅ **test_16_chips_detected** - 16-chip configuration
- ✅ **test_32_chips_detected** - 32-chip configuration (large cluster)
- ✅ **test_tt_smi_not_found** - tt-smi command not available
- ✅ **test_tt_smi_timeout** - tt-smi command times out
- ✅ **test_invalid_json_output** - tt-smi returns invalid JSON

### Model Distribution Tests (7 tests)

Tests for `calculate_model_distribution()` function:

- ✅ **test_4_chips_108_models** - Round-robin distribution (27 models per chip)
- ✅ **test_8_chips_108_models** - Distribution across 8 chips (13-14 per chip)
- ✅ **test_16_chips_108_models** - Distribution across 16 chips (6-7 per chip)
- ✅ **test_single_chip** - Single chip gets all models
- ✅ **test_more_chips_than_models** - More chips than models (some chips idle)
- ✅ **test_perfect_divisibility** - Models divide evenly (25 each for 100÷4)
- ✅ **test_chip_id_range** - Chip IDs are in correct range (0 to N-1)

### Chip Configuration Tests (4 tests)

Tests for `get_chip_config()` function:

- ✅ **test_chip_0_p300c** - Environment vars for chip 0 on P300C
- ✅ **test_chip_3_p300c** - Environment vars for chip 3 on P300C
- ✅ **test_chip_out_of_range** - Error when requesting invalid chip ID
- ✅ **test_architecture_names** - Blackhole, Wormhole, Grayskull architectures

### Mesh Descriptor Tests (5 tests)

Tests for `validate_mesh_descriptor()` function:

- ✅ **test_p300c_descriptor_exists** - P300C descriptor found
- ✅ **test_p300c_descriptor_missing** - P300C descriptor not found
- ✅ **test_p150_no_descriptor_needed** - P150 works without descriptor
- ✅ **test_n300_no_descriptor_needed** - N300 works without descriptor
- ✅ **test_no_tt_metal_home** - Falls back to default path

### Display Tests (2 tests)

Tests for `print_hardware_info()` function:

- ✅ **test_print_4_chips** - Display info for 4-chip system
- ✅ **test_print_no_devices** - Display when no devices found

### Integration Tests (2 tests)

End-to-end workflow tests:

- ✅ **test_full_workflow_4_chips_100_models** - Complete workflow:
  - Detect 4 chips
  - Calculate round-robin distribution
  - Generate environment config for each chip
  - Validate mesh descriptor

- ✅ **test_scaling_from_1_to_32_chips** - Scaling validation:
  - Tests 1, 4, 8, 16, and 32 chip configurations
  - Verifies distribution works for all chip counts
  - Confirms all models assigned exactly once

## Test Statistics

- **Total Tests**: 29
- **Pass Rate**: 100%
- **Hardware Scenarios**: 7 (0, 1, 4, 8, 16, 32 chips)
- **Board Types**: 4 (P150, P300C, N300, P100)
- **Architectures**: 3 (Blackhole, Wormhole B0, Grayskull)

## Mock Data

Tests use realistic mock data for `tt-smi -s` output:

```python
MOCK_TT_SMI_4_CHIPS_P300C = {
    "devices": [
        {
            "board_type": "P300C",
            "arch": "blackhole",
            "pci_bus": "0000:01:00.0",
            "firmware_version": "19.4.2.0"
        },
        # ... 3 more devices
    ]
}
```

Mock data covers:
- Empty systems (0 devices)
- Single chip systems (1 device)
- Small workstations (4 devices)
- Medium servers (8 devices)
- Large clusters (16-32 devices)

## Key Testing Patterns

### 1. Mocking subprocess.run()

```python
@patch('subprocess.run')
def test_example(self, mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps(mock_data),
        stderr=""
    )
    # Test uses mocked tt-smi output
```

### 2. Mocking Path.exists()

```python
@patch('pathlib.Path.exists')
def test_example(self, mock_exists):
    mock_exists.return_value = True
    # Test uses mocked file existence
```

### 3. Mocking Environment Variables

```python
@patch.dict('os.environ', {'TT_METAL_HOME': '/test/path'})
def test_example(self):
    # Test uses custom environment
```

## Why These Tests Matter

1. **No Hardware Required** - Tests run on any system, no Tenstorrent chips needed
2. **Comprehensive Coverage** - Tests all code paths in hardware.py
3. **Realistic Scenarios** - Mock data matches real tt-smi output format
4. **Scalability Validation** - Confirms round-robin works for 1-32+ chips
5. **Error Handling** - Validates graceful degradation (missing tt-smi, timeouts, etc.)
6. **Regression Prevention** - Catches bugs before they reach production

## Future Test Additions

Potential areas for expansion:

- [ ] Worker process tests (compilation with timeouts/retries)
- [ ] Forge setup tests (installation validation)
- [ ] Model library tests (filtering, sorting, metadata)
- [ ] Cache management tests (PyTorch cache handling)
- [ ] CLI tests (command parsing, help text)
- [ ] Orchestrator tests (parallel execution logic)
- [ ] Tmux layout tests (grid calculation for N chips)

## Running Tests Without pytest

The test file can also be run directly:

```bash
cd ~/code/tt-forge-compiletron
python3 tests/test_hardware.py
```

This uses pytest's `main()` function at the bottom of the file.

## Troubleshooting

### pytest-asyncio Errors

If you see:
```
ImportError: cannot import name 'Config' from 'pytest'
```

Use the `-p no:asyncio` flag:
```bash
python3 -m pytest tests/ -v -p no:asyncio
```

This disables the pytest-asyncio plugin (not needed for these tests).

### Import Errors

If you see:
```
ModuleNotFoundError: No module named 'lib'
```

Make sure you're running from the project root:
```bash
cd ~/code/tt-forge-compiletron
python3 -m pytest tests/
```

## Contributing Tests

When adding new tests:

1. Use descriptive test names: `test_<what>_<scenario>`
2. Include docstrings explaining what's being tested
3. Use appropriate mocking (don't rely on external state)
4. Verify tests pass in isolation: `pytest tests/test_file.py::TestClass::test_name`
5. Update this README with new test descriptions

## See Also

- [Hardware Detection Documentation](../docs/MULTI_CHIP.md)
- [Main README](../README.md)
- [Implementation Complete](../IMPLEMENTATION_COMPLETE.md)
