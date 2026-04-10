#!/usr/bin/env python3
"""Zarr → LeRobot v2 数据集转换脚本.

将 xarm_toolkit Collector 采集的 Zarr 数据集转换为 LeRobot v2 格式。

用法::

    python scripts/convert_to_lerobot.py \
        --input datasets/demo.zarr \
        --output lerobot_datasets/demo \
        --repo-id demo --fps 15 \
        --task "pick up the red block"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _check_deps():
    missing = []
    for pkg in ("zarr", "lerobot", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append("Pillow" if pkg == "PIL" else pkg)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("  pip install -e '.[lerobot]'")
        sys.exit(1)


def _get_episode_ranges(meta_group) -> list[tuple[int, int]]:
    """从 meta/episode_ends 提取各 episode 的 [start, end) 范围."""
    episode_ends = meta_group["episode_ends"][:]
    ranges = []
    prev = 0
    for end in episode_ends:
        ranges.append((int(prev), int(end)))
        prev = end
    return ranges


def convert(
    zarr_path: str,
    output_dir: str,
    repo_id: str,
    fps: int = 15,
    robot_type: str = "xarm6",
    task_name: str | None = None,
):
    """执行单个 Zarr → LeRobot 转换."""
    import zarr
    from PIL import Image

    # 兼容 lerobot 新旧版本导入
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if task_name is None:
        task_name = Path(zarr_path).stem

    # --- 读取 Zarr ---
    store = zarr.open(str(zarr_path), "r")
    data = store["data"]
    meta = store["meta"]

    # 探测图像尺寸
    _, C, H, W = data["rgb_arm"].shape  # (N, 3, H, W)
    image_shape = (H, W, C)
    print(f"[INFO] 图像尺寸: {W}×{H}")

    ep_ranges = _get_episode_ranges(meta)
    n_eps = len(ep_ranges)
    n_frames = sum(end - start for start, end in ep_ranges)
    print(f"[INFO] Episodes: {n_eps}, 总帧数: {n_frames}")

    # --- 准备输出路径（已有则覆盖）---
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        shutil.rmtree(output_path)
    # 只创建父目录，最终目录由 LeRobot create() 内部创建
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 创建 LeRobot 数据集 ---
    # observation.state = pos(6) + gripper_state(1) = (7,)
    # action = action(6) + gripper_action(1) = (7,)
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }

    print(f"[INFO] 创建 LeRobot 数据集: repo_id={repo_id}, fps={fps}")
    print(f"[INFO] 输出路径: {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output_path,
        use_videos=False,
        image_writer_threads=4,
    )

    # --- 逐 episode 转换 ---
    t0 = time.time()

    for ep_idx, (start, end) in enumerate(ep_ranges):
        ep_len = end - start
        print(f"  Episode {ep_idx}: frames [{start}, {end}) = {ep_len} steps")

        # 批量读取（减少 zarr I/O）
        rgb_arm_batch = data["rgb_arm"][start:end]      # (L, 3, H, W)
        rgb_fix_batch = data["rgb_fix"][start:end]
        pos_batch = data["pos"][start:end]               # (L, 6)
        action_batch = data["action"][start:end]         # (L, 6)
        gs_batch = data["gripper_state"][start:end]      # (L, 1)
        ga_batch = data["gripper_action"][start:end]     # (L, 1)

        for i in range(ep_len):
            # 图像: (C, H, W) → (H, W, C) → PIL
            img_arm = Image.fromarray(rgb_arm_batch[i].transpose(1, 2, 0))
            img_fix = Image.fromarray(rgb_fix_batch[i].transpose(1, 2, 0))

            # state: pos(6) + gripper_state(1)
            state = np.concatenate([
                pos_batch[i].astype(np.float32),
                gs_batch[i].astype(np.float32),
            ])  # (7,)

            # action: action(6) + gripper_action(1)
            action_vec = np.concatenate([
                action_batch[i].astype(np.float32),
                ga_batch[i].astype(np.float32),
            ])  # (7,)

            dataset.add_frame({
                "observation.image": img_fix,
                "observation.wrist_image": img_arm,
                "observation.state": state,
                "action": action_vec,
                "task": task_name,
            })

        dataset.save_episode()

    # --- 完成 ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"[DONE] 转换完成!")
    print(f"  输出: {output_path}")
    print(f"  Episodes: {n_eps}, Frames: {n_frames}")
    print(f"  耗时: {elapsed:.1f}s ({n_frames / max(elapsed, 0.1):.0f} frames/s)")
    print(f"{'=' * 60}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Zarr → LeRobot v2 数据集转换")
    parser.add_argument("--input", "-i", required=True, help="输入 Zarr 数据集路径")
    parser.add_argument("--output", "-o", required=True, help="LeRobot 输出目录")
    parser.add_argument("--repo-id", required=True, help="数据集 repo ID")
    parser.add_argument("--fps", type=int, default=15, help="帧率 (默认 15)")
    parser.add_argument("--robot-type", default="xarm6", help="机器人类型 (默认 xarm6)")
    parser.add_argument("--task", default=None, help="任务描述 (默认使用 zarr 文件名)")

    args = parser.parse_args()
    _check_deps()

    if not Path(args.input).exists():
        print(f"[ERROR] 输入路径不存在: {args.input}")
        sys.exit(1)

    convert(
        zarr_path=args.input,
        output_dir=args.output,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task,
    )


if __name__ == "__main__":
    main()
