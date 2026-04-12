"""xarm_toolkit.deploy — VLA model server/client for robot deployment."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarm_toolkit.deploy.client import VLAClient
    from xarm_toolkit.deploy.openpi_server import Pi05XArmPolicy
    from xarm_toolkit.deploy.robot_deploy import XArmDeployer
    from xarm_toolkit.deploy.server import VLAServer

__all__ = ["VLAClient", "VLAServer", "Pi05XArmPolicy", "XArmDeployer"]

_LAZY_IMPORTS: dict[str, str] = {
    "VLAClient": "xarm_toolkit.deploy.client",
    "VLAServer": "xarm_toolkit.deploy.server",
    "Pi05XArmPolicy": "xarm_toolkit.deploy.openpi_server",
    "XArmDeployer": "xarm_toolkit.deploy.robot_deploy",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
