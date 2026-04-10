"""xarm_toolkit — utilities for xArm robot and RealSense camera control."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarm_toolkit.env.realsense_env import RealsenseEnv
    from xarm_toolkit.env.xarm_env import XArmEnv

__all__ = ["RealsenseEnv", "XArmEnv"]

_LAZY_IMPORTS: dict[str, str] = {
    "RealsenseEnv": "xarm_toolkit.env.realsense_env",
    "XArmEnv": "xarm_toolkit.env.xarm_env",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
