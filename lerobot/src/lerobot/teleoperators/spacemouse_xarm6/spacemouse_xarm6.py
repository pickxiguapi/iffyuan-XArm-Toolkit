"""LeRobot SpaceMouse teleoperator for XArm6 - implementation.

Wraps :class:`xarm_toolkit.teleop.spacemouse.SpacemouseAgent` behind the
standard LeRobot :class:`Teleoperator` interface so that
``lerobot-record --teleop.type=spacemouse_xarm6`` works out-of-the-box.

**No files in xarm_toolkit/ are modified** - this is a pure adapter layer.

Action output format (per-joint floats, same keys as Robot's action_features):
    dx, dy, dz, droll, dpitch, dyaw : float  — 6-DOF delta
    gripper_action                   : float  — 0.0 (close) / 1.0 (open)
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from .config_spacemouse_xarm6 import SpacemouseXarm6Config

logger = logging.getLogger(__name__)

# Must match Robot's _ACTION_KEYS exactly
_ACTION_KEYS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_action"]

# Gripper threshold for 0/1 mapping
_GRIPPER_THRESHOLD = 420


class SpacemouseXarm6(Teleoperator):
    """LeRobot-compatible SpaceMouse teleoperator for XArm6.

    Reads 6-DOF deltas from a 3DConnexion SpaceMouse and returns them
    as per-joint floats matching the :class:`Xarm6` robot's ``action_features``.
    """

    config_class = SpacemouseXarm6Config
    name = "spacemouse_xarm6"

    def __init__(self, config: SpacemouseXarm6Config):
        super().__init__(config)
        self.config = config
        self._agent = None

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, Any]:
        return {key: float for key in _ACTION_KEYS}

    @property
    def feedback_features(self) -> dict[str, Any]:
        # SpaceMouse has no haptic feedback
        return {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._agent is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.warning("SpaceMouse already connected - skipping.")
            return

        from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig

        sm_config = SpacemouseConfig(
            translation_scale=self.config.translation_scale,
            z_scale=self.config.z_scale,
            rotation_scale=self.config.rotation_scale,
            deadzone=self.config.deadzone,
            gripper_open_pos=self.config.gripper_open_pos,
            gripper_close_pos=self.config.gripper_close_pos,
        )
        self._agent = SpacemouseAgent(config=sm_config)
        logger.info("SpaceMouse teleoperator connected.")

    def disconnect(self) -> None:
        self._agent = None
        logger.info("SpaceMouse teleoperator disconnected.")

    # ------------------------------------------------------------------
    # Calibration (no-op)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Configuration (no-op)
    # ------------------------------------------------------------------

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Read SpaceMouse and return per-joint action dict.

        Returns
        -------
        dict with keys: dx, dy, dz, droll, dpitch, dyaw, gripper_action
            Each value is a float.
            gripper_action: 0.0 = closed, 1.0 = open
        """
        if not self.is_connected:
            raise ConnectionError("SpaceMouse is not connected. Call connect() first.")

        action_6d, gripper_pos = self._agent.act()

        # Map gripper position to 0/1
        gripper_action = 1.0 if gripper_pos > _GRIPPER_THRESHOLD else 0.0

        # Return per-joint floats (pipeline aggregates into action (7,))
        result: dict[str, float] = {}
        for i, key in enumerate(_ACTION_KEYS[:-1]):
            result[key] = float(action_6d[i])
        result["gripper_action"] = gripper_action

        return result

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No-op - SpaceMouse has no haptic feedback."""
        pass
