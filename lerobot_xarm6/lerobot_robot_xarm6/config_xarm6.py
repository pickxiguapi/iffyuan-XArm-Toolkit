"""LeRobot XArm6 robot plugin — configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("xarm6")
@dataclass
class Xarm6Config(RobotConfig):
    """Configuration for an XArm 6-DOF robot with dual RealSense cameras.

    This config is automatically discovered by LeRobot when the
    ``lerobot_robot_xarm6`` package is installed (pip install -e .).
    """

    # --- XArm connection ---
    ip_address: str = "192.168.31.232"
    action_mode: str = "delta_eef"  # delta_eef | absolute_eef | absolute_joint
    initial_gripper_position: int = 840  # 0 = closed, 840 = fully open

    # --- Cameras (managed by us, not LeRobot's camera system) ---
    cam_arm_serial: str = "327122075644"   # D435i (arm-mounted)
    cam_fix_serial: str = "f1271506"       # L515 (fixed)
    image_width: int = 320
    image_height: int = 240

    # --- LeRobot camera dict (empty — we handle cameras ourselves) ---
    cameras: dict = field(default_factory=dict)
