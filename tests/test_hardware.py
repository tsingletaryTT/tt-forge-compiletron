"""
Tests for hardware detection module.

Tests hardware detection, chip distribution, and environment configuration
using mocked tt-smi output for various hardware scenarios.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path so we can import lib modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.hardware import (
    detect_hardware,
    calculate_model_distribution,
    get_chip_config,
    validate_mesh_descriptor,
    print_hardware_info
)


# Mock tt-smi outputs for different hardware scenarios
# Matches actual tt-smi -s JSON structure
MOCK_TT_SMI_NO_DEVICES = {
    "device_info": []
}

MOCK_TT_SMI_SINGLE_P150 = {
    "device_info": [
        {
            "board_info": {
                "board_type": "p150",
                "bus_id": "0000:01:00.0"
            },
            "firmwares": {
                "fw_bundle_version": "19.4.2.0"
            }
        }
    ]
}

MOCK_TT_SMI_4_CHIPS_P300C = {
    "device_info": [
        {
            "board_info": {
                "board_type": "p300c",
                "bus_id": f"0000:0{i}:00.0"
            },
            "firmwares": {
                "fw_bundle_version": "19.4.2.0"
            }
        }
        for i in range(1, 5)
    ]
}

MOCK_TT_SMI_8_CHIPS_N300 = {
    "device_info": [
        {
            "board_info": {
                "board_type": "n300",
                "bus_id": f"0000:0{i}:00.0"
            },
            "firmwares": {
                "fw_bundle_version": "19.3.0.0"
            }
        }
        for i in range(1, 9)
    ]
}

MOCK_TT_SMI_16_CHIPS_MIXED = {
    "device_info": [
        {
            "board_info": {
                "board_type": "p300c",
                "bus_id": f"0000:{i:02d}:00.0"
            },
            "firmwares": {
                "fw_bundle_version": "19.4.2.0"
            }
        }
        for i in range(1, 17)
    ]
}

MOCK_TT_SMI_32_CHIPS = {
    "device_info": [
        {
            "board_info": {
                "board_type": "p100",
                "bus_id": f"0000:{i:02d}:00.0"
            },
            "firmwares": {
                "fw_bundle_version": "19.0.0.0"
            }
        }
        for i in range(1, 33)
    ]
}


class TestDetectHardware:
    """Test hardware detection with various tt-smi outputs."""

    @patch('subprocess.run')
    def test_no_devices_detected(self, mock_run):
        """Test when no Tenstorrent devices are present."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_NO_DEVICES),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 0
        assert hw['board_type'] == 'unknown'
        assert hw['arch'] == 'unknown'

    @patch('subprocess.run')
    def test_single_p150_detected(self, mock_run):
        """Test single P150 chip detection."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_SINGLE_P150),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 1
        assert hw['board_type'] == 'P150'
        assert hw['arch'] == 'blackhole'
        assert 'devices' in hw
        assert len(hw['devices']) == 1

    @patch('subprocess.run')
    def test_4_chips_p300c_detected(self, mock_run):
        """Test 4-chip P300C configuration (2x dual-chip boards)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 4
        assert hw['board_type'] == 'P300C'
        assert hw['arch'] == 'blackhole'
        assert len(hw['devices']) == 4

    @patch('subprocess.run')
    def test_8_chips_n300_detected(self, mock_run):
        """Test 8-chip N300 configuration."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_8_CHIPS_N300),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 8
        assert hw['board_type'] == 'N300'
        assert hw['arch'] == 'wormhole_b0'
        assert len(hw['devices']) == 8

    @patch('subprocess.run')
    def test_16_chips_detected(self, mock_run):
        """Test 16-chip configuration."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_16_CHIPS_MIXED),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 16
        assert len(hw['devices']) == 16

    @patch('subprocess.run')
    def test_32_chips_detected(self, mock_run):
        """Test 32-chip configuration (large cluster)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_32_CHIPS),
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 32
        assert hw['board_type'] == 'P100'
        assert hw['arch'] == 'grayskull'

    @patch('subprocess.run')
    def test_tt_smi_not_found(self, mock_run):
        """Test when tt-smi command is not available."""
        mock_run.side_effect = FileNotFoundError("tt-smi not found")

        hw = detect_hardware()
        assert hw['num_chips'] == 0
        assert hw.get('error') is not None

    @patch('subprocess.run')
    def test_tt_smi_timeout(self, mock_run):
        """Test when tt-smi times out."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('tt-smi', 10)

        hw = detect_hardware()
        assert hw['num_chips'] == 0
        assert hw.get('error') is not None

    @patch('subprocess.run')
    def test_invalid_json_output(self, mock_run):
        """Test when tt-smi returns invalid JSON."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="invalid json {{{",
            stderr=""
        )

        hw = detect_hardware()
        assert hw['num_chips'] == 0
        assert hw.get('error') is not None


class TestCalculateModelDistribution:
    """Test round-robin model distribution across chips."""

    def test_4_chips_108_models(self):
        """Test distribution of 108 models across 4 chips."""
        distribution = calculate_model_distribution(108, 4)

        # Should have 4 entries (one per chip)
        assert len(distribution) == 4

        # Each chip should get 27 models
        for chip_id, model_indices in distribution:
            assert len(model_indices) == 27

        # Verify round-robin: chip 0 gets 0,4,8,12,...
        chip0_models = distribution[0][1]
        assert chip0_models[0] == 0
        assert chip0_models[1] == 4
        assert chip0_models[2] == 8

        # Verify all models assigned exactly once
        all_models = []
        for _, models in distribution:
            all_models.extend(models)
        assert sorted(all_models) == list(range(108))

    def test_8_chips_108_models(self):
        """Test distribution of 108 models across 8 chips."""
        distribution = calculate_model_distribution(108, 8)

        assert len(distribution) == 8

        # 108 / 8 = 13.5, so some chips get 14, others 13
        model_counts = [len(models) for _, models in distribution]
        assert sum(model_counts) == 108
        assert min(model_counts) == 13
        assert max(model_counts) == 14

    def test_16_chips_108_models(self):
        """Test distribution of 108 models across 16 chips."""
        distribution = calculate_model_distribution(108, 16)

        assert len(distribution) == 16

        # 108 / 16 = 6.75, so some chips get 7, others 6
        model_counts = [len(models) for _, models in distribution]
        assert sum(model_counts) == 108
        assert min(model_counts) == 6
        assert max(model_counts) == 7

    def test_single_chip(self):
        """Test single chip gets all models."""
        distribution = calculate_model_distribution(101, 1)

        assert len(distribution) == 1
        assert len(distribution[0][1]) == 101
        assert distribution[0][1] == list(range(101))

    def test_more_chips_than_models(self):
        """Test when there are more chips than models."""
        distribution = calculate_model_distribution(5, 10)

        # Only first 5 chips get models
        assert len(distribution) == 10

        # First 5 chips get 1 model each
        for i in range(5):
            assert len(distribution[i][1]) == 1
            assert distribution[i][1][0] == i

        # Last 5 chips get 0 models
        for i in range(5, 10):
            assert len(distribution[i][1]) == 0

    def test_perfect_divisibility(self):
        """Test when models divide evenly across chips."""
        distribution = calculate_model_distribution(100, 4)

        # Each chip should get exactly 25 models
        for chip_id, model_indices in distribution:
            assert len(model_indices) == 25

    def test_chip_id_range(self):
        """Test chip IDs are in correct range."""
        distribution = calculate_model_distribution(50, 8)

        chip_ids = [chip_id for chip_id, _ in distribution]
        assert chip_ids == list(range(8))


class TestGetChipConfig:
    """Test per-chip environment configuration."""

    @patch('subprocess.run')
    def test_chip_0_p300c(self, mock_run):
        """Test environment config for chip 0 on P300C."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        hw = detect_hardware()
        config = get_chip_config(0, hw['num_chips'], hw['arch'], hw['board_type'])

        assert config['TT_VISIBLE_DEVICES'] == '0'
        assert config['TT_METAL_ARCH_NAME'] == 'blackhole'
        assert 'TT_MESH_GRAPH_DESC_PATH' in config

    @patch('subprocess.run')
    def test_chip_3_p300c(self, mock_run):
        """Test environment config for chip 3 on P300C."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        hw = detect_hardware()
        config = get_chip_config(3, hw['num_chips'], hw['arch'], hw['board_type'])

        assert config['TT_VISIBLE_DEVICES'] == '3'
        assert config['TT_METAL_ARCH_NAME'] == 'blackhole'

    @patch('subprocess.run')
    def test_chip_out_of_range(self, mock_run):
        """Test requesting chip ID beyond available chips."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        hw = detect_hardware()

        # Should raise ValueError for chip_id >= num_chips
        with pytest.raises(ValueError):
            get_chip_config(4, hw['num_chips'], hw['arch'], hw['board_type'])

    @patch('subprocess.run')
    def test_architecture_names(self, mock_run):
        """Test different architecture names."""
        # Test Blackhole
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )
        hw = detect_hardware()
        config = get_chip_config(0, hw['num_chips'], hw['arch'], hw['board_type'])
        assert config['TT_METAL_ARCH_NAME'] == 'blackhole'

        # Test Wormhole B0
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_8_CHIPS_N300),
            stderr=""
        )
        hw = detect_hardware()
        config = get_chip_config(0, hw['num_chips'], hw['arch'], hw['board_type'])
        assert config['TT_METAL_ARCH_NAME'] == 'wormhole_b0'

        # Test Grayskull
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_32_CHIPS),
            stderr=""
        )
        hw = detect_hardware()
        config = get_chip_config(0, hw['num_chips'], hw['arch'], hw['board_type'])
        assert config['TT_METAL_ARCH_NAME'] == 'grayskull'


class TestValidateMeshDescriptor:
    """Test mesh descriptor path validation."""

    @patch('pathlib.Path.exists')
    @patch.dict('os.environ', {'TT_METAL_HOME': '/home/test/tt-metal'})
    def test_p300c_descriptor_exists(self, mock_exists):
        """Test P300C mesh descriptor validation when file exists."""
        mock_exists.return_value = True

        result = validate_mesh_descriptor('P300C', 'blackhole')
        assert result is not None

    @patch('pathlib.Path.exists')
    @patch.dict('os.environ', {'TT_METAL_HOME': '/home/test/tt-metal'})
    def test_p300c_descriptor_missing(self, mock_exists):
        """Test P300C mesh descriptor validation when file missing."""
        mock_exists.return_value = False

        result = validate_mesh_descriptor('P300C', 'blackhole')
        assert result is None

    @patch('os.path.exists')
    def test_p150_no_descriptor_needed(self, mock_exists):
        """Test P150 doesn't require mesh descriptor."""
        # P150 may check for descriptor but finding it is optional
        result = validate_mesh_descriptor('P150', 'blackhole')
        # Result can be None or a path, just checking it doesn't crash
        assert result is None or isinstance(result, Path)

    @patch('os.path.exists')
    def test_n300_no_descriptor_needed(self, mock_exists):
        """Test N300 doesn't require mesh descriptor."""
        result = validate_mesh_descriptor('N300', 'wormhole_b0')
        # N300 doesn't need descriptor, should return None
        assert result is None
        mock_exists.assert_not_called()

    @patch('pathlib.Path.exists')
    @patch.dict('os.environ', {}, clear=True)
    def test_no_tt_metal_home(self, mock_exists):
        """Test validation when TT_METAL_HOME not set."""
        # Should handle missing TT_METAL_HOME gracefully by using default path
        mock_exists.return_value = False  # Simulate no descriptor found

        result = validate_mesh_descriptor('P300C', 'blackhole')
        # When TT_METAL_HOME not set, falls back to ~/tt-metal
        # If descriptor not found there, returns None
        assert result is None


class TestPrintHardwareInfo:
    """Test hardware info display function."""

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_print_4_chips(self, mock_print, mock_run):
        """Test printing info for 4-chip system."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        hw = detect_hardware()
        print_hardware_info(hw)

        # Should print something about chips and architecture
        assert mock_print.called

    @patch('subprocess.run')
    @patch('builtins.print')
    def test_print_no_devices(self, mock_print, mock_run):
        """Test printing info when no devices found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_NO_DEVICES),
            stderr=""
        )

        hw = detect_hardware()
        print_hardware_info(hw)

        # Should print error or warning message
        assert mock_print.called


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    @patch('subprocess.run')
    def test_full_workflow_4_chips_100_models(self, mock_run):
        """Test complete workflow: detect hardware, distribute models, configure chips."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(MOCK_TT_SMI_4_CHIPS_P300C),
            stderr=""
        )

        # Step 1: Detect hardware
        hw = detect_hardware()
        assert hw['num_chips'] == 4

        # Step 2: Calculate distribution
        distribution = calculate_model_distribution(100, hw['num_chips'])
        assert len(distribution) == 4

        # Step 3: Get config for each chip
        for chip_id in range(hw['num_chips']):
            config = get_chip_config(chip_id, hw['num_chips'], hw['arch'], hw['board_type'])
            assert config['TT_VISIBLE_DEVICES'] == str(chip_id)
            assert 'TT_METAL_ARCH_NAME' in config

        # Step 4: Validate mesh descriptor (can be None or Path, just check it doesn't crash)
        result = validate_mesh_descriptor(hw['board_type'], hw['arch'])
        assert result is None or isinstance(result, Path)

    @patch('subprocess.run')
    def test_scaling_from_1_to_32_chips(self, mock_run):
        """Test that distribution works correctly as chip count increases."""
        test_cases = [
            (MOCK_TT_SMI_SINGLE_P150, 1),
            (MOCK_TT_SMI_4_CHIPS_P300C, 4),
            (MOCK_TT_SMI_8_CHIPS_N300, 8),
            (MOCK_TT_SMI_16_CHIPS_MIXED, 16),
            (MOCK_TT_SMI_32_CHIPS, 32)
        ]

        for mock_data, expected_chips in test_cases:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_data),
                stderr=""
            )

            hw = detect_hardware()
            assert hw['num_chips'] == expected_chips

            # Distribute 108 models
            distribution = calculate_model_distribution(108, expected_chips)

            # Verify all models assigned
            all_models = []
            for _, models in distribution:
                all_models.extend(models)
            assert sorted(all_models) == list(range(108))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
