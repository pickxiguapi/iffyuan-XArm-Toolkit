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

    # --- 抽样第一帧，只打印我们关心的字段 ---
    print(f"\n[INFO] 抽样 frame[0]:")
    sample = dataset[0]

    # 我们转换写入的数据字段
    data_keys = [
        "observation.image",
        "observation.wrist_image",
        "observation.state",
        "action",
    ]
    # LeRobot 自动生成的元数据字段（标量 tensor，不需要关注）
    meta_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index", "task"}

    ok = True

    for key in data_keys:
        if key not in sample:
            print(f"  [FAIL] 缺少 {key}")
            ok = False
            continue
        value = sample[key]
        if hasattr(value, "shape"):
            shape_str = "x".join(str(d) for d in value.shape)
            print(f"  {key:30s} [{shape_str}] {value.dtype}")

    # state / action 维度检查
    if "observation.state" in sample:
        dim = sample["observation.state"].shape[-1]
        if dim != 7:
            print(f"  [WARN] observation.state dim={dim}, 期望 7")
            ok = False

    if "action" in sample:
        dim = sample["action"].shape[-1]
        if dim != 7:
            print(f"  [WARN] action dim={dim}, 期望 7")
            ok = False

    # task 描述
    if "task" in sample:
        print(f"  {'task':30s} = {sample['task']}")

    # 打印省略的元数据字段列表
    skipped = [k for k in sample if k in meta_keys and k != "task"]
    if skipped:
        print(f"  (元数据字段省略: {', '.join(skipped)})")

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
