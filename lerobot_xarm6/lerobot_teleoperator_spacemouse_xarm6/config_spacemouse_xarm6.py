"""LeRobot SpaceMouse teleoperator plugin for XArm6 — configuration."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse_xarm6")
@dataclass
class SpacemouseXarm6Config(TeleoperatorConfig):
    """Configuration for SpaceMouse 6-DOF teleoperator adapted for XArm6.

    Wraps :class:`xarm_toolkit.teleop.spacemouse.SpacemouseAgent` with
    the standard LeRobot Teleoperator interface.
    """

    # --- SpaceMouse scaling ---
    translation_scale: float = 5.0    # mm per raw unit
    z_scale: float | None = None      # None → same as translation_scale
    rotation_scale: float = 0.004     # rad per raw unit
    deadzone: float = 0.0             # ignore raw values below this

    # --- Gripper range ---
    gripper_open_pos: int = 840
    gripper_close_pos: int = 0
