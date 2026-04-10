"""Tests for RealsenseEnv using a mocked Open3D backend.

Run with:  pytest tests/test_realsense_env.py -v
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-mock open3d so the module can be imported without hardware / open3d.
# ---------------------------------------------------------------------------

_mock_o3d = MagicMock()

# Simulate list_devices
_mock_o3d.t.io.RealSenseSensor.list_devices.return_value = []

# Simulate metadata
_mock_metadata = MagicMock()
_mock_metadata.intrinsics.intrinsic_matrix = np.eye(3)
_mock_metadata.depth_scale = 0.001

_mock_sensor_instance = MagicMock()
_mock_sensor_instance.get_metadata.return_value = _mock_metadata

# capture_frame returns a mock RGBDImage with .color attribute
_mock_rgbd = MagicMock()
_mock_color = MagicMock()
_mock_rgbd.color = _mock_color
_mock_sensor_instance.capture_frame.return_value = _mock_rgbd

# create_from_rgbd_image returns a mock PointCloud
_mock_pcd = MagicMock()
_mock_o3d.t.geometry.PointCloud.create_from_rgbd_image.return_value = _mock_pcd

# RealSenseSensor() constructor returns our mock instance
_mock_o3d.t.io.RealSenseSensor.return_value = _mock_sensor_instance

# Provide Dtype / Tensor so config construction works
_mock_o3d.core.Dtype.Float32 = "float32"
_mock_o3d.core.Tensor.return_value = MagicMock()

# Inject mock into sys.modules before importing the module under test
sys.modules.setdefault("open3d", _mock_o3d)

from xarm_toolkit.env.realsense_env import (  # noqa: E402
    CAMERA_CONFIGS,
    RealsenseEnv,
    _DEFAULT_CONFIG,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset call counts between tests."""
    _mock_sensor_instance.reset_mock()
    _mock_o3d.t.io.RealSenseSensor.reset_mock()
    _mock_o3d.t.io.RealSenseSensor.list_devices.return_value = []
    _mock_o3d.t.io.RealSenseSensor.return_value = _mock_sensor_instance
    _mock_o3d.t.geometry.PointCloud.create_from_rgbd_image.reset_mock()
    _mock_sensor_instance.get_metadata.return_value = _mock_metadata
    _mock_sensor_instance.capture_frame.return_value = _mock_rgbd
    yield


def _make_env(serial: str = "327122075644", mode: str = "rgbd", record: bool = False):
    with patch("xarm_toolkit.env.realsense_env.atexit"):
        return RealsenseEnv(serial=serial, mode=mode, record=record)


# =========================================================================
# Tests: configuration registry
# =========================================================================

class TestCameraConfigs:
    def test_known_serials_present(self):
        assert "327122075644" in CAMERA_CONFIGS
        assert "f1271506" in CAMERA_CONFIGS

    def test_l515_has_visual_preset(self):
        assert "visual_preset" in CAMERA_CONFIGS["f1271506"]

    def test_d435i_has_no_visual_preset(self):
        assert "visual_preset" not in CAMERA_CONFIGS["327122075644"]

    def test_default_config_has_required_keys(self):
        required = {"color_format", "color_resolution", "depth_format", "depth_resolution", "fps"}
        assert required <= set(_DEFAULT_CONFIG.keys())


# =========================================================================
# Tests: construction
# =========================================================================

class TestConstruction:
    def test_known_serial_no_warning(self, caplog):
        _make_env(serial="327122075644")
        assert "Unknown serial" not in caplog.text

    def test_unknown_serial_logs_warning(self, caplog):
        _make_env(serial="unknown_serial_123")
        assert "Unknown serial" in caplog.text

    def test_intrinsic_matrix_stored(self):
        env = _make_env()
        np.testing.assert_array_equal(env.intrinsic_matrix, np.eye(3))

    def test_depth_scale_stored(self):
        env = _make_env()
        assert env.depth_scale == pytest.approx(0.001)

    def test_record_mode(self):
        _make_env(record=True)
        _mock_sensor_instance.init_sensor.assert_called_once()
        _mock_sensor_instance.start_capture.assert_called_once_with(True)

    def test_non_record_mode(self):
        _make_env(record=False)
        _mock_sensor_instance.start_capture.assert_called_once_with()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _make_env(mode="invalid")

    def test_mode_stored(self):
        for m in ("rgb", "rgbd", "pcd"):
            env = _make_env(mode=m)
            assert env.mode == m


# =========================================================================
# Tests: observation — rgb mode
# =========================================================================

class TestRgbMode:
    def test_obs_keys(self):
        env = _make_env(mode="rgb")
        obs = env.step()
        assert "rgb" in obs
        assert "rgbd" not in obs
        assert "pcd" not in obs

    def test_rgb_is_rgbd_color(self):
        env = _make_env(mode="rgb")
        obs = env.step()
        assert obs["rgb"] is _mock_color

    def test_no_pcd_computed(self):
        env = _make_env(mode="rgb")
        env.step()
        _mock_o3d.t.geometry.PointCloud.create_from_rgbd_image.assert_not_called()


# =========================================================================
# Tests: observation — rgbd mode
# =========================================================================

class TestRgbdMode:
    def test_obs_keys(self):
        env = _make_env(mode="rgbd")
        obs = env.step()
        assert "rgbd" in obs
        assert "pcd" not in obs
        assert "rgb" not in obs

    def test_rgbd_is_captured_frame(self):
        env = _make_env(mode="rgbd")
        obs = env.step()
        assert obs["rgbd"] is _mock_rgbd

    def test_no_pcd_computed(self):
        env = _make_env(mode="rgbd")
        env.step()
        _mock_o3d.t.geometry.PointCloud.create_from_rgbd_image.assert_not_called()

    def test_intrinsic_in_obs(self):
        env = _make_env(mode="rgbd")
        obs = env.step()
        np.testing.assert_array_equal(obs["intrinsic_matrix"], np.eye(3))

    def test_depth_scale_in_obs(self):
        env = _make_env(mode="rgbd")
        obs = env.step()
        assert obs["depth_scale"] == pytest.approx(0.001)


# =========================================================================
# Tests: observation — pcd mode
# =========================================================================

class TestPcdMode:
    def test_obs_keys(self):
        env = _make_env(mode="pcd")
        obs = env.step()
        assert "rgbd" in obs
        assert "pcd" in obs

    def test_pcd_created(self):
        env = _make_env(mode="pcd")
        obs = env.step()
        assert obs["pcd"] is _mock_pcd
        _mock_o3d.t.geometry.PointCloud.create_from_rgbd_image.assert_called_once()


# =========================================================================
# Tests: reset / step
# =========================================================================

class TestResetStep:
    def test_reset_returns_obs(self):
        env = _make_env(mode="rgbd")
        obs = env.reset()
        assert "rgbd" in obs

    def test_step_returns_obs(self):
        env = _make_env(mode="rgbd")
        obs = env.step()
        assert "rgbd" in obs


# =========================================================================
# Tests: cleanup
# =========================================================================

class TestCleanup:
    def test_cleanup_stops_capture(self):
        env = _make_env()
        env.cleanup()
        _mock_sensor_instance.stop_capture.assert_called_once()
