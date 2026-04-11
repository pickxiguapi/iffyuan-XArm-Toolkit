"""LeRobot XArm6 robot implementation.

Wraps :class:`xarm_toolkit.env.xarm_env.XArmEnv` and
:class:`xarm_toolkit.env.realsense_env.RealsenseEnv` behind the standard
LeRobot :class:`Robot` interface, so ``lerobot-record / lerobot-teleoperate``
work out-of-the-box with the XArm 6-DOF arm + dual RealSense cameras.

**No files in xarm_toolkit/ are modified** - this is a pure adapter layer.

Feature mapping (follows SO100/Koch convention — short keys, per-joint floats):
    observation.state  <- [x, y, z, roll, pitch, yaw, gripper] each as float
    action             <- [dx, dy, dz, droll, dpitch, dyaw, gripper_action] each as float
    observation.images.image       <- rgb_fix (fixed camera)  -> (H, W, 3) uint8
    observation.images.wrist_image <- rgb_arm (arm camera)    -> (H, W, 3) uint8
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation

from .config_xarm6 import Xarm6Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names — short keys (no "observation." / "action." prefix)
# hw_to_dataset_features adds the prefix automatically.
# ---------------------------------------------------------------------------

# State features: each declared as float, aggregated into observation.state (7,)
_STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# Action features: each declared as float, aggregated into action (7,)
_ACTION_KEYS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_action"]

# Camera feature keys — become observation.images.image / observation.images.wrist_image
_CAM_FIX = "image"             # fixed camera - main view
_CAM_ARM = "wrist_image"       # arm camera - wrist view

# Gripper threshold: <= 420 -> closed (0), > 420 -> open (1)
_GRIPPER_THRESHOLD = 420


class Xarm6(Robot):
    """LeRobot-compatible adapter for the XArm 6 robot.

    Connects to the real hardware via ``xarm_toolkit``'s :class:`XArmEnv` and
    :class:`RealsenseEnv`.  Uses the same feature mapping as the Zarr->LeRobot
    converter (see README):

    - ``observation.state``: [x, y, z, roll, pitch, yaw, gripper] -> (7,) float32
    - ``action``: [dx, dy, dz, droll, dpitch, dyaw, gripper_action] -> (7,) float32
    - ``observation.images.image``: fixed camera RGB
    - ``observation.images.wrist_image``: arm-mounted camera RGB
    """

    config_class = Xarm6Config
    name = "xarm6"

    def __init__(self, config: Xarm6Config):
        super().__init__(config)
        self.config = config

        # Lazily created on connect()
        self._env = None
        self._cam_arm = None
        self._cam_fix = None

        # XArm6 manages cameras internally (not via LeRobot's camera system),
        # but lerobot-record uses len(robot.cameras) for thread pool sizing.
        # Expose a dict keyed by camera name so the count is correct.
        self.cameras = {_CAM_ARM: None, _CAM_FIX: None}

    # ------------------------------------------------------------------
    # Feature declarations (follows SO100/Koch convention)
    # ------------------------------------------------------------------

    @property
    def observation_features(self) -> dict[str, Any]:
        h, w = self.config.image_height, self.config.image_width
        features: dict[str, Any] = {}
        # Per-joint state features -> aggregated into observation.state by pipeline
        for key in _STATE_KEYS:
            features[key] = float
        # Camera features -> become observation.images.{key}
        features[_CAM_FIX] = (h, w, 3)
        features[_CAM_ARM] = (h, w, 3)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {key: float for key in _ACTION_KEYS}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._env is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.warning("XArm6 already connected - skipping.")
            return

        # Import xarm_toolkit components (they live in the parent project)
        from xarm_toolkit.env.xarm_env import XArmEnv
        from xarm_toolkit.env.realsense_env import RealsenseEnv

        logger.info("Connecting to XArm6 at %s ...", self.config.ip_address)
        self._env = XArmEnv(
            addr=self.config.ip_address,
            use_force=False,
            action_mode=self.config.action_mode,
            initial_gripper_position=self.config.initial_gripper_position,
        )

        logger.info("Connecting cameras ...")
        self._cam_arm = RealsenseEnv(
            serial=self.config.cam_arm_serial,
            mode="rgb",
        )
        self._cam_fix = RealsenseEnv(
            serial=self.config.cam_fix_serial,
            mode="rgb",
        )

        # Reset arm to home pose
        self._env.reset(close_gripper=False)
        logger.info("XArm6 connected and ready.")

    def disconnect(self) -> None:
        if self._cam_arm is not None:
            self._cam_arm.cleanup()
            self._cam_arm = None
        if self._cam_fix is not None:
            self._cam_fix.cleanup()
            self._cam_fix = None
        if self._env is not None:
            self._env.cleanup()
            self._env = None
        logger.info("XArm6 disconnected.")

    # ------------------------------------------------------------------
    # Calibration (no-op for XArm - factory-calibrated)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Configuration (no-op - handled in connect)
    # ------------------------------------------------------------------

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def get_observation(self) -> RobotObservation:
        """Read all sensors and return a dict matching observation_features.

        Returns dict with short keys (no "observation." prefix):
            x, y, z, roll, pitch, yaw : float   — end-effector pose
            gripper                    : float   — 0.0 (closed) or 1.0 (open)
            image                      : np.ndarray (H, W, 3) uint8 — fixed cam
            wrist_image                : np.ndarray (H, W, 3) uint8 — arm cam
        """
        if not self.is_connected:
            raise ConnectionError("XArm6 is not connected. Call connect() first.")

        # Arm state
        env_obs = self._env._get_observation()

        pos = env_obs["cart_pos"]  # list of 6 floats: [x, y, z, roll, pitch, yaw]
        gripper_raw = float(env_obs["gripper_position"])
        gripper_state = 1.0 if gripper_raw > _GRIPPER_THRESHOLD else 0.0

        # Camera images
        h, w = self.config.image_height, self.config.image_width

        obs: dict[str, Any] = {}
        # State: per-joint floats (pipeline aggregates into observation.state)
        for i, key in enumerate(_STATE_KEYS[:-1]):  # x, y, z, roll, pitch, yaw
            obs[key] = float(pos[i])
        obs["gripper"] = gripper_state

        # Images: short keys (pipeline maps to observation.images.{key})
        obs[_CAM_FIX] = self._capture_rgb(self._cam_fix, w, h)
        obs[_CAM_ARM] = self._capture_rgb(self._cam_arm, w, h)

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send motor commands to the XArm.

        Args:
            action: Dict with per-joint keys (dx, dy, dz, droll, dpitch, dyaw, gripper_action),
                    each a float value. This matches the action_features format.

        Returns:
            The action dict that was actually sent.
        """
        if not self.is_connected:
            raise ConnectionError("XArm6 is not connected. Call connect() first.")

        # Extract 6-DOF delta from per-joint dict
        action_6d = np.array(
            [float(action[k]) for k in _ACTION_KEYS[:6]],
            dtype=np.float64,
        )
        gripper_action_val = float(action["gripper_action"])

        # gripper_action: 0 -> close (0), 1 -> open (840)
        gripper_pos = self.config.initial_gripper_position if gripper_action_val > 0.5 else 0

        self._env.step(action_6d, gripper_action=gripper_pos)
        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_rgb(cam, width: int, height: int, max_retries: int = 5) -> np.ndarray:
        """Capture one RGB frame, convert to numpy uint8, and resize.

        The first few frames after camera startup may be empty — retry
        up to *max_retries* times before falling back to a black image.

        Returns (H, W, 3) uint8 array.
        """
        import time

        for _ in range(max_retries):
            cam_obs = cam.step()
            rgb = cam_obs["rgb"]
            img = np.asarray(rgb)  # (H_orig, W_orig, 3) uint8

            if img.size > 0 and img.ndim == 3:
                if img.shape[0] != height or img.shape[1] != width:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                return img

            time.sleep(0.05)

        logger.warning("Camera returned empty frame after %d retries, using black image.", max_retries)
        return np.zeros((height, width, 3), dtype=np.uint8)
