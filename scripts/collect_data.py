#!/usr/bin/env python3
"""Data collection entry point.

Usage:
    # 默认任务，采集 3 个 episode
    python scripts/collect_data.py --dataset datasets/demo.zarr

    # 指定任务配置 + episode 数量
    python scripts/collect_data.py --task plug --episodes 10 --dataset datasets/plug.zarr

    # 无力传感器模式
    python scripts/collect_data.py --no-force --dataset datasets/test.zarr

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
from pathlib import Path

import yaml

from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig
from xarm_toolkit.collect.collector import Collector
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("collect_data")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_task_config(task_name: str) -> dict:
    """Load task config from configs/tasks/<task_name>.yaml."""
    task_file = CONFIGS_DIR / "tasks" / f"{task_name}.yaml"
    if not task_file.exists():
        logger.warning("Task config '%s' not found, using default.", task_name)
        task_file = CONFIGS_DIR / "tasks" / "default.yaml"
    with open(task_file) as f:
        return yaml.safe_load(f) or {}


def load_hardware_config() -> dict:
    """Load hardware config from configs/hardware.yaml."""
    hw_file = CONFIGS_DIR / "hardware.yaml"
    if not hw_file.exists():
        logger.warning("hardware.yaml not found, using defaults.")
        return {}
    with open(hw_file) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    p = argparse.ArgumentParser(description="XArm6 data collection")
    p.add_argument("--dataset", type=str, default="datasets/demo.zarr",
                   help="Zarr dataset path")
    p.add_argument("--task", type=str, default="default",
                   help="Task name (loads configs/tasks/<task>.yaml)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to collect")
    p.add_argument("--force", action="store_true",
                   help="Enable force sensor (default: off)")
    p.add_argument("--cam-mode", type=str, default="rgbd",
                   choices=["rgb", "rgbd", "pcd"],
                   help="Camera mode: rgb, rgbd (default), pcd")
    p.add_argument("--trans-scale", type=float, default=5.0,
                   help="SpaceMouse translation sensitivity")
    p.add_argument("--rot-scale", type=float, default=0.004,
                   help="SpaceMouse rotation sensitivity")
    return p.parse_args()


def main():
    args = parse_args()

    hw_cfg = load_hardware_config()
    task_cfg = load_task_config(args.task)

    robot_cfg = hw_cfg.get("robot", {})
    cam_cfg = hw_cfg.get("cameras", {})
    collect_cfg = hw_cfg.get("collect", {})

    # --- Init env ---
    use_force = args.force
    env = XArmEnv(
        addr=robot_cfg.get("addr", "192.168.31.232"),
        use_force=use_force,
        action_mode="delta_eef",
        initial_gripper_position=840,
    )

    # --- Init cameras (mode matches --cam-mode) ---
    cam_mode = args.cam_mode
    arm_serial = cam_cfg.get("arm", {}).get("serial", "327122075644")
    fix_serial = cam_cfg.get("fix", {}).get("serial", "f1271506")
    cam_arm = RealsenseEnv(serial=arm_serial, mode=cam_mode)
    cam_fix = RealsenseEnv(serial=fix_serial, mode=cam_mode)

    # --- Init SpaceMouse ---
    sm_cfg = SpacemouseConfig(
        translation_scale=args.trans_scale,
        rotation_scale=args.rot_scale,
    )
    agent = SpacemouseAgent(config=sm_cfg)

    # --- Image size ---
    img_size = tuple(collect_cfg.get("image_size", [320, 240]))

    # --- Run collector ---
    logger.info("Task: %s | Episodes: %d | Force: %s | Cam: %s | Dataset: %s",
                args.task, args.episodes, use_force, cam_mode, args.dataset)

    collector = Collector(
        env=env,
        cam_arm=cam_arm,
        cam_fix=cam_fix,
        agent=agent,
        dataset_path=args.dataset,
        task_config=task_cfg,
        num_episodes=args.episodes,
        cam_mode=cam_mode,
        image_size=img_size,
        warmup_time=collect_cfg.get("warmup_time", 1.0),
    )

    collector.run()


if __name__ == "__main__":
    main()
