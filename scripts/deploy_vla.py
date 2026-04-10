#!/usr/bin/env python3
"""Robot-side VLA deployment script.

Connects to a remote VLA server, captures observations from the robot and
cameras, sends them for inference, and executes the returned actions.

Usage::

    python scripts/deploy_vla.py \\
        --server-host 192.168.1.100 \\
        --server-port 10093 \\
        --instruction "pick up the red cup" \\
        --ip 192.168.31.232
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from xarm_toolkit.deploy.client import VLAClient
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("deploy_vla")

# Camera serial numbers
CAM_ARM_SERIAL = "327122075644"   # D435i (arm-mounted)
CAM_FIX_SERIAL = "f1271506"       # L515 (fixed)


def main():
    parser = argparse.ArgumentParser(description="VLA robot-side deployment")
    parser.add_argument("--server-host", type=str, default="localhost",
                        help="VLA server hostname / IP")
    parser.add_argument("--server-port", type=int, default=10093,
                        help="VLA server port")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Language instruction for the task")
    parser.add_argument("--ip", type=str, default="192.168.31.232",
                        help="XArm IP address")
    parser.add_argument("--action-mode", type=str, default="absolute_joint",
                        choices=["absolute_joint", "absolute_eef", "delta_eef"],
                        help="Action mode for XArmEnv")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control frequency (Hz)")
    args = parser.parse_args()

    dt = 1.0 / args.hz

    # ---- Initialize hardware ----
    logger.info("Initializing XArm environment ...")
    env = XArmEnv(
        addr=args.ip,
        action_mode=args.action_mode,
        use_force=False,
    )

    logger.info("Initializing cameras ...")
    cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode="rgb")
    cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode="rgb")

    # ---- Connect to VLA server ----
    logger.info("Connecting to VLA server at %s:%d ...", args.server_host, args.server_port)
    client = VLAClient(host=args.server_host, port=args.server_port)

    if not client.ping():
        logger.error("Server ping failed!")
        return

    logger.info("Server connected. Metadata: %s", client.metadata)

    # ---- Reset ----
    obs = env.reset(close_gripper=False)
    logger.info("Robot reset. Starting deployment loop ...")
    logger.info("Instruction: %s", args.instruction)

    try:
        for step_i in range(args.max_steps):
            t0 = time.monotonic()

            # Capture images
            rgb_arm = cam_arm.step()["rgb"]
            rgb_fix = cam_fix.step()["rgb"]

            # Robot state
            state = obs["servo_angle"]

            # Inference
            result = client.predict(
                images=[rgb_arm, rgb_fix],
                instruction=args.instruction,
                state=state,
            )

            actions = result.get("actions")
            if actions is None:
                logger.warning("No actions returned at step %d", step_i)
                continue

            # Execute action (first action in the chunk)
            action = actions[0] if actions.ndim > 1 else actions
            # Split into arm action and gripper
            arm_action = action[:6]
            gripper_action = float(action[6]) if len(action) > 6 else None

            obs = env.step(
                action=arm_action,
                gripper_action=gripper_action,
            )

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if step_i % 10 == 0:
                logger.info(
                    "Step %d/%d  elapsed=%.3fs  pos=%s",
                    step_i, args.max_steps, elapsed,
                    np.array2string(obs["cart_pos"][:3], precision=1),
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        client.close()
        env.cleanup()
        cam_arm.cleanup()
        cam_fix.cleanup()
        logger.info("Done.")


if __name__ == "__main__":
    main()
