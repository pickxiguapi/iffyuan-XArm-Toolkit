#!/usr/bin/env python3
"""验证转换后的 LeRobot 数据集.

加载 LeRobotDataset 做端到端检查：帧数、episode、维度、图像 shape。

用法::

    python scripts/verify_lerobot.py \
        --path lerobot_datasets/demo \
        --repo-id demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def verify(lerobot_path: str, repo_id: str):
    """验证 LeRobot 数据集."""
    # 兼容 lerobot 新旧版本导入
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    resolved = Path(lerobot_path).resolve()
    if not resolved.exists():
        print(f"[FAIL] 路径不存在: {resolved}")
        sys.exit(1)

    # --- 加载 ---
    print(f"[INFO] 加载数据集: {resolved}")
    dataset = LeRobotDataset(repo_id, root=resolved)

    n_frames = len(dataset)
    n_episodes = dataset.num_episodes
    n_tasks = len(dataset.meta.tasks)

    print(f"  repo_id:    {dataset.repo_id}")
    print(f"  fps:        {dataset.fps}")
    print(f"  robot_type: {dataset.meta.robot_type}")
    print(f"  frames:     {n_frames}")
    print(f"  episodes:   {n_episodes}")
    print(f"  tasks:      {n_tasks}")

    for idx, task in dataset.meta.tasks.items():
        print(f"    [{idx}] {task}")

    # --- 抽样第一帧 ---
    print(f"\n[INFO] 抽样 frame[0]:")
    sample = dataset[0]
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key:30s} shape={str(value.shape):15s} dtype={value.dtype}")
        elif hasattr(value, "size"):
            print(f"  {key:30s} size={value.size} mode={value.mode}")
        else:
            print(f"  {key:30s} = {value}")

    # --- 检查关键字段 ---
    ok = True
    for img_key in ["observation.image", "observation.wrist_image"]:
        if img_key not in sample:
            print(f"  [FAIL] 缺少 {img_key}")
            ok = False

    if "observation.state" in sample:
        dim = sample["observation.state"].shape[-1]
        print(f"\n  observation.state dim = {dim} (期望 7)")
        if dim != 7:
            print(f"  [WARN] 维度不符")

    if "action" in sample:
        dim = sample["action"].shape[-1]
        print(f"  action dim = {dim} (期望 7)")
        if dim != 7:
            print(f"  [WARN] 维度不符")

    print(f"\n[{'PASS' if ok else 'FAIL'}] 验证完成")


def main():
    parser = argparse.ArgumentParser(description="验证 LeRobot 数据集")
    parser.add_argument("--path", "-p", required=True, help="LeRobot 数据集路径")
    parser.add_argument("--repo-id", default=None, help="repo ID (默认使用目录名)")

    args = parser.parse_args()
    repo_id = args.repo_id or Path(args.path).name
    verify(args.path, repo_id)


if __name__ == "__main__":
    main()
