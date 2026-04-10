"""Tests for XArmEnv using a mocked XArmAPI.

Run with:  pytest tests/test_xarm_env.py -v
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy / hardware-only dependencies so the module can be imported
# even in CI environments where pytransform3d / xarm SDK are not installed.
# ---------------------------------------------------------------------------

_mock_pr = MagicMock()
_mock_pr.quaternion_dist.return_value = 0.0
_mock_pr.quaternion_from_extrinsic_euler_xyz.return_value = np.array([1, 0, 0, 0])

for mod_name in (
    "pytransform3d",
    "pytransform3d.rotations",
    "pytransform3d.transformations",
    "xarm",
    "xarm.wrapper",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Patch the specific sub-module so `from pytransform3d import rotations as pr`
# returns our configured mock.
sys.modules["pytransform3d.rotations"] = _mock_pr
# Also set it as an attribute on the parent mock so `from pytransform3d import rotations`
# resolves correctly.
sys.modules["pytransform3d"].rotations = _mock_pr

# ---------------------------------------------------------------------------
# Mock XArmAPI factory
# ---------------------------------------------------------------------------

def _make_mock_arm(mode: int = 7):
    """Return a MagicMock that behaves like XArmAPI for happy-path calls."""
    arm = MagicMock()

    # Properties
    type(arm).mode = PropertyMock(return_value=mode)
    type(arm).ft_ext_force = PropertyMock(return_value=[0.0] * 6)
    type(arm).ft_raw_force = PropertyMock(return_value=[0.1] * 6)

    # Position / angle getters — return (code=0, payload)
    arm.get_position.return_value = (0, [470.0, 0.0, 530.0, math.pi, 0.0, -math.pi / 2])
    arm.get_servo_angle.return_value = (0, [0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2, math.pi / 2, 0.0])
    arm.get_gripper_position.return_value = (0, 0.0)

    # Setters — return code=0
    arm.set_position.return_value = 0
    arm.set_servo_angle.return_value = 0
    arm.set_gripper_position.return_value = 0
    arm.ft_sensor_enable.return_value = 0
    arm.ft_sensor_set_zero.return_value = 0

    # No-ops
    arm.clean_error.return_value = None
    arm.clean_warn.return_value = None
    arm.motion_enable.return_value = None
    arm.set_mode.return_value = None
    arm.set_state.return_value = None
    arm.set_gripper_mode.return_value = None
    arm.set_gripper_enable.return_value = None
    arm.set_gripper_speed.return_value = None
    arm.disconnect.return_value = None

    return arm


@pytest.fixture()
def mock_arm():
    return _make_mock_arm()


@pytest.fixture()
def env_force(mock_arm):
    """XArmEnv with use_force=True (default)."""
    with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
        from xarm_toolkit.env.xarm_env import XArmEnv
        env = XArmEnv(use_force=True)
    return env


@pytest.fixture()
def env_no_force(mock_arm):
    """XArmEnv with use_force=False."""
    with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
        from xarm_toolkit.env.xarm_env import XArmEnv
        env = XArmEnv(use_force=False)
    return env


# =========================================================================
# Tests: construction
# =========================================================================

class TestConstruction:
    def test_default_action_mode(self, env_force):
        assert env_force.action_mode == "delta_eef"

    def test_use_force_flag(self, env_force, env_no_force):
        assert env_force.use_force is True
        assert env_no_force.use_force is False

    def test_invalid_action_mode(self, mock_arm):
        with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
            from xarm_toolkit.env.xarm_env import XArmEnv
            with pytest.raises(ValueError, match="Invalid action_mode"):
                XArmEnv(action_mode="invalid")

    def test_force_sensor_enabled_on_init(self, mock_arm):
        """When use_force=True, ft_sensor_enable should be called."""
        with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
            from xarm_toolkit.env.xarm_env import XArmEnv
            XArmEnv(use_force=True)
        mock_arm.ft_sensor_enable.assert_called()

    def test_force_sensor_not_enabled_when_disabled(self, mock_arm):
        """When use_force=False, ft_sensor_enable should NOT be called."""
        with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
            from xarm_toolkit.env.xarm_env import XArmEnv
            XArmEnv(use_force=False)
        mock_arm.ft_sensor_enable.assert_not_called()


# =========================================================================
# Tests: observation
# =========================================================================

class TestObservation:
    def test_obs_keys_with_force(self, env_force):
        obs = env_force._get_observation()
        assert set(obs.keys()) == {
            "cart_pos", "servo_angle", "ext_force", "raw_force",
            "goal_pos", "gripper_position",
        }
        assert obs["ext_force"] is not None
        assert obs["raw_force"] is not None

    def test_obs_keys_without_force(self, env_no_force):
        obs = env_no_force._get_observation()
        assert obs["ext_force"] is None
        assert obs["raw_force"] is None

    def test_servo_angle_shape(self, env_force):
        obs = env_force._get_observation()
        assert obs["servo_angle"].shape == (6,)


# =========================================================================
# Tests: action modes
# =========================================================================

class TestDeltaEef:
    def test_delta_accumulates(self, env_force):
        initial = env_force.goal_pos.copy()
        delta = np.array([10, 0, 0, 0, 0, 0], dtype=np.float64)
        env_force.step(delta)
        np.testing.assert_allclose(env_force.goal_pos, initial + delta)

    def test_calls_set_position(self, env_force):
        env_force.step([1, 2, 3, 0, 0, 0])
        env_force.arm.set_position.assert_called()


class TestAbsoluteEef:
    def test_absolute_sets_goal(self, mock_arm):
        with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
            from xarm_toolkit.env.xarm_env import XArmEnv
            env = XArmEnv(action_mode="absolute_eef", use_force=False)
        target = np.array([400, 10, 500, math.pi, 0, 0], dtype=np.float64)
        env.step(target)
        np.testing.assert_allclose(env.goal_pos, target)
        mock_arm.set_position.assert_called()


class TestAbsoluteJoint:
    def test_calls_set_servo_angle(self, mock_arm):
        type(mock_arm).mode = PropertyMock(return_value=6)
        with patch("xarm_toolkit.env.xarm_env.XArmAPI", return_value=mock_arm):
            from xarm_toolkit.env.xarm_env import XArmEnv
            env = XArmEnv(action_mode="absolute_joint", use_force=False)
        joints = [0, 0, -math.pi / 2, 0, math.pi / 2, 0]
        env.step(joints)
        mock_arm.set_servo_angle.assert_called()

# =========================================================================
# Tests: retry & error handling
# =========================================================================

class TestRetry:
    def test_retry_succeeds_after_failures(self):
        from xarm_toolkit.env.xarm_env import _retry

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return (1, None)
            return (0, "ok")

        result = _retry(flaky, max_retries=5, label="test")
        assert result == (0, "ok")
        assert call_count == 3

    def test_retry_exhausted_raises(self):
        from xarm_toolkit.env.xarm_env import _retry, XArmCommandError

        def always_fail():
            return (99, None)

        with pytest.raises(XArmCommandError, match="99"):
            _retry(always_fail, max_retries=2, label="fail_test")

    def test_retry_plain_int_code(self):
        from xarm_toolkit.env.xarm_env import _retry

        def ok():
            return 0

        result = _retry(ok, max_retries=1, label="int_test")
        assert result == 0


# =========================================================================
# Tests: reset
# =========================================================================

class TestReset:
    @patch("xarm_toolkit.env.xarm_env.time")
    def test_reset_returns_obs(self, mock_time, env_no_force):
        obs = env_no_force.reset(close_gripper=True)
        assert "cart_pos" in obs
        assert "servo_angle" in obs

    @patch("xarm_toolkit.env.xarm_env.time")
    def test_reset_restores_goal_pos(self, mock_time, env_no_force):
        env_no_force.step([10, 0, 0, 0, 0, 0])
        env_no_force.reset()
        np.testing.assert_allclose(env_no_force.goal_pos, env_no_force.reset_pose)


# =========================================================================
# Tests: cleanup
# =========================================================================

class TestCleanup:
    def test_cleanup_disconnects(self, env_force):
        env_force.cleanup()
        env_force.arm.disconnect.assert_called_once()

    def test_cleanup_disables_force_sensor(self, env_force):
        env_force.cleanup()
        # ft_sensor_enable(0) should be called during cleanup
        env_force.arm.ft_sensor_enable.assert_called_with(0)

    def test_cleanup_no_force_sensor(self, env_no_force):
        """Cleanup should not try to disable force sensor when use_force=False."""
        env_no_force.arm.ft_sensor_enable.reset_mock()
        env_no_force.cleanup()
        env_no_force.arm.ft_sensor_enable.assert_not_called()
