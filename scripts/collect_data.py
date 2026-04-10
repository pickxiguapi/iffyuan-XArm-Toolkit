#!/usr/bin/env python3
"""Data collection entry point.

Usage:
    # 默认采集 rgbd，3 个 episode
    python scripts/collect_data.py --dataset datasets/demo.zarr

    # 指定任务初始偏移
    python scripts/collect_data.py --start-bias 0 0 -200 --dataset datasets/plug.zarr

    # 带随机偏移 + 夹爪始终闭合（stamp 类任务）
    python scripts/collect_data.py --start-bias 0 0 -270 --gripper-closed --dataset datasets/stamp.zarr

键盘控制:
    Space   — 开始录制当前 episode
    Enter   — 结束当前 episode
    Ctrl+C  — 中止采集

SpaceMouse:
    6D 移动    — 控制机械臂末端
    左键(btn0) — 等待阶段: 张开夹爪 / 录制阶段: 张开夹爪
    右键(btn1) — 等待阶段: 关闭夹爪 / 录制阶段: 关闭夹爪
"""

from __future__ import annotations

import argparse

from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig
from xarm_toolkit.collect.collector import Collector
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("collect_data")

# 硬件常量
XARM_IP = "192.168.31.232"
CAM_ARM_SERIAL = "327122075644"  # D435i
CAM_FIX_SERIAL = "f1271506"      # L515


def parse_args():
    p = argparse.ArgumentParser(description="XArm6 data collection")

    # 数据集
    p.add_argument("--dataset", type=str, default="datasets/demo.zarr",
                   help="Zarr dataset path")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to collect")

    # 任务参数（替代 configs/tasks/*.yaml）
    p.add_argument("--start-bias", type=float, nargs=3, default=[0, 0, 0],
                   metavar=("X", "Y", "Z"),
                   help="Initial position offset from Home [x, y, z] in mm (default: 0 0 0)")
    p.add_argument("--random-bias-x", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Random X offset range per episode, e.g. --random-bias-x -50 70")
    p.add_argument("--random-bias-y", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Random Y offset range per episode, e.g. --random-bias-y -20 10")
    p.add_argument("--gripper-closed", action="store_true",
                   help="Gripper always closed (stamp-like tasks)")

    # 硬件
    p.add_argument("--ip", type=str, default=XARM_IP,
                   help="XArm IP address")
    p.add_argument("--force", action="store_true",
                   help="Enable force sensor (default: off)")

    # 相机
    p.add_argument("--cam-mode", type=str, default="rgbd",
                   choices=["rgb", "rgbd", "pcd"],
                   help="Camera mode: rgb, rgbd (default), pcd")
    p.add_argument("--image-size", type=int, nargs=2, default=[320, 240],
                   metavar=("W", "H"),
                   help="Image size for saved data (default: 320 240)")

    # 视频
    p.add_argument("--save-video", action="store_true",
                   help="Save per-episode MP4 videos for review")
    p.add_argument("--video-fps", type=float, default=15.0,
                   help="FPS for saved videos (default 15)")

    # SpaceMouse
    p.add_argument("--trans-scale", type=float, default=5.0,
                   help="SpaceMouse translation sensitivity")
    p.add_argument("--rot-scale", type=float, default=0.004,
                   help="SpaceMouse rotation sensitivity")

    return p.parse_args()


def main():
    args = parse_args()

    # --- Build task config from CLI args ---
    task_cfg = {
        "start_bias": args.start_bias,
        "gripper_always_closed": args.gripper_closed,
    }
    if args.random_bias_x is not None:
        task_cfg.setdefault("random_bias", {})["x"] = args.random_bias_x
    if args.random_bias_y is not None:
        task_cfg.setdefault("random_bias", {})["y"] = args.random_bias_y

    # --- Init env ---
    env = XArmEnv(
        addr=args.ip,
        use_force=args.force,
        action_mode="delta_eef",
        initial_gripper_position=840,
    )

    # --- Init cameras ---
    cam_mode = args.cam_mode
    cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode=cam_mode)
    cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode=cam_mode)

    # --- Init SpaceMouse ---
    sm_cfg = SpacemouseConfig(
        translation_scale=args.trans_scale,
        rotation_scale=args.rot_scale,
    )
    agent = SpacemouseAgent(config=sm_cfg)

    # --- Run collector ---
    logger.info(
        "Episodes: %d | Force: %s | Cam: %s | Bias: %s | Dataset: %s",
        args.episodes, args.force, cam_mode, args.start_bias, args.dataset,
    )

    collector = Collector(
        env=env,
        cam_arm=cam_arm,
        cam_fix=cam_fix,
        agent=agent,
        dataset_path=args.dataset,
        task_config=task_cfg,
        num_episodes=args.episodes,
        cam_mode=cam_mode,
        image_size=tuple(args.image_size),
        save_video=args.save_video,
        video_fps=args.video_fps,
    )

    collector.run()


if __name__ == "__main__":
    main()
