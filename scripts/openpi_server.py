#!/usr/bin/env python3
"""GPU-side OpenPI Pi0.5 inference server for XArm6.

Thin entry point — all logic lives in :mod:`xarm_toolkit.deploy`.

Usage::

    python scripts/openpi_server.py \\
        --config-file /path/to/pi05_xarm_v2_stage1.py \\
        --ckpt-dir /path/to/checkpoint \\
        --port 10093
"""

from __future__ import annotations

import argparse

from xarm_toolkit.deploy.openpi_server import Pi05XArmPolicy
from xarm_toolkit.deploy.server import VLAServer
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("openpi_server")


def main():
    parser = argparse.ArgumentParser(description="Start Pi0.5 XArm6 inference server")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to openpi config Python file (must define build_config())")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Local checkpoint directory")
    parser.add_argument("--port", type=int, default=10093,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server bind address")
    parser.add_argument("--idle-timeout", type=float, default=0,
                        help="Connection idle timeout in seconds (0 = disabled)")
    args = parser.parse_args()

    policy = Pi05XArmPolicy(
        config_file=args.config_file,
        ckpt_dir=args.ckpt_dir,
    )

    server = VLAServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout
    )

    logger.info("Starting OpenPI server on %s:%d", args.host, args.port)
    server.run()


if __name__ == "__main__":
    main()
