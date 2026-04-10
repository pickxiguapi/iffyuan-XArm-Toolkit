"""LeRobot XArm6 robot plugin — robot implementation.

Wraps :class:`xarm_toolkit.env.xarm_env.XArmEnv` and
:class:`xarm_toolkit.env.realsense_env.RealsenseEnv` behind the standard
LeRobot :class:`Robot` interface, so ``lerobot record / teleoperate / replay``
work out-of-the-box with the XArm 6-DOF arm + dual RealSense cameras.

**No files in xarm_toolkit/ are modified** — this is a pure adapter layer.

Feature mapping (aligned with Pi0.5 / VLA convention):
    observation.state       ← pos(6) + gripper_state(1)  → (7,) float32
    action                  ← action(6) + gripper_action(1) → (7,) float32
    observation.image       ← rgb_fix (固定相机)           → (H, W, 3) uint8
    observation.wrist_image ← rgb_arm (臂上相机)           → (H, W, 3) uint8
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from lerobot.robots.robot import Robot

from lerobot_robot_xarm6.config_xarm6 import Xarm6Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants — aligned with Pi0.5 / VLA convention
# ---------------------------------------------------------------------------

# observation.state = pos(6) + gripper_state(1) = (7,)
_OBS_STATE = "observation.state"
_STATE_DIM = 7

# action = action_delta(6) + gripper_action(1) = (7,)
_ACTION = "action"
_ACTION_DIM = 7

# Camera keys
_CAM_FIX = "observation.image"         # 固定相机 — 主视角
_CAM_ARM = "observation.wrist_image"   # 臂上相机 — 手腕视角

# Gripper threshold: ≤ 420 → closed (0), > 420 → open (1)
_GRIPPER_THRESHOLD = 420


class Xarm6(Robot):
    """LeRobot-compatible adapter for the XArm 6 robot.

    Connects to the real hardware via ``xarm_toolkit``'s :class:`XArmEnv` and
    :class:`RealsenseEnv`.  Uses the same feature mapping as the Zarr→LeRobot
    converter (see README):

    - ``observation.state``: pos(6) + gripper_state(1) → (7,) float32
    - ``action``: action_delta(6) + gripper_action(1) → (7,) float32
    - ``observation.image``: fixed camera RGB
    - ``observation.wrist_image``: arm-mounted camera RGB
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

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------

    @property
    def observation_features(self) -> dict[str, Any]:
        h, w = self.config.image_height, self.config.image_width
        return {
            _OBS_STATE: (_STATE_DIM,),           # pos(6) + gripper_state(1)
            _CAM_FIX: (h, w, 3),                 # 固定相机 RGB
            _CAM_ARM: (h, w, 3),                 # 臂上相机 RGB
        }

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            _ACTION: (_ACTION_DIM,),              # action(6) + gripper_action(1)
        }

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._env is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.warning("XArm6 already connected — skipping.")
            return

        # Import xarm_toolkit components (they live in the parent project)
        from xarm_toolkit.env.xarm_env import XArmEnv
        from xarm_toolkit.env.realsense_env import RealsenseEnv

        logger.info("Connecting to XArm6 at %s ...", self.config.ip_address)
        self._env = XArmEnv(
            addr=self.config.ip_address,
            use_force=False,  # LeRobot 采集不需要力控
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
    # Calibration (no-op for XArm — factory-calibrated)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Read all sensors and return a dict matching observation_features.

        Returns
        -------
        dict with keys:
            observation.state       : np.ndarray (7,) float32
                                      pos(6) + gripper_state(1)
            observation.image       : np.ndarray (H, W, 3) uint8
                                      fixed camera RGB
            observation.wrist_image : np.ndarray (H, W, 3) uint8
                                      arm-mounted camera RGB
        """
        if not self.is_connected:
            raise ConnectionError("XArm6 is not connected. Call connect() first.")

        # Arm state
        env_obs = self._env._get_observation()

        # observation.state = pos(6) + gripper_state(1)
        pos = np.array(env_obs["cart_pos"], dtype=np.float32)       # (6,)
        gripper_raw = float(env_obs["gripper_position"])
        gripper_state = np.array(
            [0.0 if gripper_raw <= _GRIPPER_THRESHOLD else 1.0],
            dtype=np.float32,
        )  # (1,)
        state = np.concatenate([pos, gripper_state])                # (7,)

        # Camera images — capture, convert to numpy, resize
        h, w = self.config.image_height, self.config.image_width

        return {
            _OBS_STATE: state,
            _CAM_FIX: self._capture_rgb(self._cam_fix, w, h),
            _CAM_ARM: self._capture_rgb(self._cam_arm, w, h),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send motor commands to the XArm.

        Args:
            action: Dict with key ``"action"`` → np.ndarray (7,)
                    = [action_delta(6), gripper_action(1)].

        Returns:
            The action dict that was actually sent.
        """
        if not self.is_connected:
            raise ConnectionError("XArm6 is not connected. Call connect() first.")

        act = np.asarray(action[_ACTION], dtype=np.float64)

        # Split: first 6 = eef delta/absolute, last 1 = gripper
        action_6d = act[:6]
        gripper_action_val = float(act[6])

        # gripper_action: 0 → close (0), 1 → open (840)
        gripper_pos = self.config.initial_gripper_position if gripper_action_val > 0.5 else 0

        self._env.step(action_6d, gripper_action=gripper_pos)
        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_rgb(cam, width: int, height: int) -> np.ndarray:
        """Capture one RGB frame, convert to numpy uint8, and resize.

        Returns (H, W, 3) uint8 array.
        """
        cam_obs = cam.step()

        # RealsenseEnv returns open3d Tensor for "rgb" mode
        rgb = cam_obs["rgb"]
        img = np.asarray(rgb)  # (H_orig, W_orig, 3) uint8

        # Resize if needed
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        return img
