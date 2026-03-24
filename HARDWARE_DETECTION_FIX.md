# Hardware Detection Fix - Real tt-smi Integration

## Issue
Running `python3 compiletron.py detect` on a machine with 4 Tenstorrent chips was returning "No hardware detected" even though `tt-smi` showed 4 devices.

## Root Cause
The code was using incorrect JSON structure that didn't match actual `tt-smi -s` output:

### Expected (incorrect):
```json
{
  "devices": [
    {"board_type": "P300C", "arch": "blackhole", "pci_bus": "..."}
  ]
}
```

### Actual tt-smi output:
```json
{
  "device_info": [
    {
      "board_info": {
        "board_type": "p300c",
        "bus_id": "0000:04:00.0"
      },
      "firmwares": {...},
      "telemetry": {...}
    }
  ]
}
```

## Key Differences

1. **Array name**: `device_info` not `devices`
2. **Nested structure**: `board_info.board_type` not direct `board_type`
3. **Case**: lowercase `p300c` not uppercase `P300C`
4. **Bus ID field**: `bus_id` not `pci_bus`
5. **No arch field**: Architecture must be inferred from board_type

## Fix Applied

### 1. Updated `lib/hardware.py` (lines 45-66)

**Before:**
```python
devices = data.get('devices', [])
first_device = devices[0]
board_type = first_device.get('board_type', 'unknown')
arch = first_device.get('arch', 'unknown')
```

**After:**
```python
# tt-smi uses 'device_info' not 'devices'
devices = data.get('device_info', [])

# Get board type from nested structure
first_device = devices[0]
board_info = first_device.get('board_info', {})
board_type = board_info.get('board_type', 'unknown').upper()

# Map board type to architecture (tt-smi doesn't include arch)
board_to_arch = {
    'P300C': 'blackhole',
    'P150': 'blackhole',
    'N300': 'wormhole_b0',
    'P100': 'grayskull',
}
arch = board_to_arch.get(board_type, 'unknown')
```

### 2. Improved device info extraction (lines 72-89)

Added simplified device structure for easy access:
```python
simplified_devices = []
for i, dev in enumerate(devices):
    board_info = dev.get('board_info', {})
    simplified_devices.append({
        'device_index': i,
        'board_type': board_type,
        'bus_id': board_info.get('bus_id', 'unknown'),
        'raw': dev  # Keep full device data for advanced use
    })
```

### 3. Updated print function (line 267)

Changed from `pci_bus` to `bus_id`:
```python
bus_id = dev.get('bus_id', '?')
print(f"    • Device {dev_id}: {hw_info['board_type']} at {bus_id}")
```

### 4. Updated test mock data

Changed all mock tt-smi outputs to match actual structure:
```python
MOCK_TT_SMI_4_CHIPS_P300C = {
    "device_info": [  # Changed from "devices"
        {
            "board_info": {  # Nested structure
                "board_type": "p300c",  # Lowercase
                "bus_id": "0000:01:00.0"  # Changed from "pci_bus"
            },
            "firmwares": {
                "fw_bundle_version": "19.4.2.0"
            }
        }
        # ... for each device
    ]
}
```

## Verification

### Before Fix:
```bash
$ python3 compiletron.py detect
Loaded 101 models
❌ Error: No devices detected
```

### After Fix:
```bash
$ python3 compiletron.py detect
Loaded 101 models
✓ Detected 4 Tenstorrent chip(s)
  Board type: P300C
  Architecture: blackhole

  Devices:
    • Device 0: P300C at 0000:04:00.0
    • Device 1: P300C at 0000:03:00.0
    • Device 2: P300C at 0000:02:00.0
    • Device 3: P300C at 0000:01:00.0

Summary: 4x P300C (Blackhole)

Round-robin distribution (108 models):
  Chip 0: 27 models
  Chip 1: 27 models
  Chip 2: 27 models
  Chip 3: 27 models
```

### Tests Still Pass:
```bash
$ ./run_tests.sh
============================== 29 passed in 0.04s ==============================
✅ All tests passed!
```

## Board Type to Architecture Mapping

The fix includes automatic mapping since tt-smi doesn't provide architecture directly:

| Board Type | Architecture | Notes |
|------------|--------------|-------|
| P300C | blackhole | Dual-chip Blackhole board |
| P150 | blackhole | Single-chip Blackhole |
| N300 | wormhole_b0 | Wormhole B0 architecture |
| P100 | grayskull | Grayskull architecture |

## Files Modified

1. **`lib/hardware.py`** (3 sections updated)
   - JSON parsing (lines 45-66)
   - Device info extraction (lines 72-89)
   - Print function (line 267)

2. **`tests/test_hardware.py`** (6 mock data structures updated)
   - MOCK_TT_SMI_NO_DEVICES
   - MOCK_TT_SMI_SINGLE_P150
   - MOCK_TT_SMI_4_CHIPS_P300C
   - MOCK_TT_SMI_8_CHIPS_N300
   - MOCK_TT_SMI_16_CHIPS_MIXED
   - MOCK_TT_SMI_32_CHIPS

## Impact

✅ **Hardware detection now works on real hardware**
✅ **All 29 tests still pass**
✅ **More accurate device information displayed**
✅ **Correct architecture inference from board type**
✅ **No changes needed to CLI or API**

## Lesson Learned

When building tools that integrate with external commands:
1. ✅ **Test with real data early** - Mock data should match actual output
2. ✅ **Document actual format** - Don't assume structure
3. ✅ **Handle missing fields gracefully** - Architecture not always present
4. ✅ **Normalize data** - Convert lowercase to uppercase for consistency

## Next Steps

Hardware detection is now fully functional! You can:

1. **Detect your hardware**:
   ```bash
   python3 compiletron.py detect
   ```

2. **List models by family**:
   ```bash
   python3 compiletron.py models list --family resnet
   ```

3. **Estimate compilation time**:
   ```bash
   python3 compiletron.py models estimate --count 50 --chips 4
   ```

4. **Run compilations** (when Forge is ready):
   ```bash
   source ~/tt-forge-fe/env/activate
   python3 compiletron.py run --quick
   ```
