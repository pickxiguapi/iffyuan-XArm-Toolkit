#!/usr/bin/env python3
"""XArm6 ACT 训练入口.

使用 LeRobot 框架中的 ACT (Action Chunking Transformers) 策略，
在本地 LeRobot v3 数据集上训练 XArm6 操作任务。

用法::

    python scripts/train_act.py \
        --dataset lerobot_datasets/act_pick2 \
        --repo-id xarm6_act_pick2 \
        --output outputs/act_pick2 \
        --batch-size 64 \
        --steps 40000 \
        --chunk-size 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description="XArm6 ACT 训练")
    parser.add_argument("--dataset", "-d", required=True, help="本地 LeRobot 数据集路径")
    parser.add_argument("--repo-id", required=True, help="数据集 repo ID")
    parser.add_argument("--output", "-o", default="outputs/act_xarm6", help="输出目录 (默认 outputs/act_xarm6)")
    parser.add_argument("--device", default="cuda", help="设备: cuda / cpu / mps (默认 cuda)")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=64, help="批大小 (默认 64，显存不够降到 8/16)")
    parser.add_argument("--steps", type=int, default=40_000, help="总训练步数 (默认 40000)")
    parser.add_argument("--log-freq", type=int, default=100, help="日志打印间隔 (默认 100)")
    parser.add_argument("--save-freq", type=int, default=5_000, help="Checkpoint 保存间隔 (默认 5000)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 工作进程数 (默认 4)")

    # ACT 超参
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="动作预测长度 (默认 64，建议 30~100)")
    parser.add_argument("--n-action-steps", type=int, default=64,
                        help="每次策略调用执行的步数 (默认 64，≤ chunk_size)")

    args = parser.parse_args()

    dataset_path = args.dataset
    repo_id = args.repo_id
    output_dir = Path(args.output)
    device = torch.device(args.device)

    batch_size = args.batch_size
    training_steps = args.steps
    log_freq = args.log_freq
    save_freq = args.save_freq
    num_workers = args.num_workers

    chunk_size = args.chunk_size
    n_action_steps = args.n_action_steps

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 打印配置 ─────────────────────────────────────────
    print("=" * 60)
    print("  XArm6 ACT 训练")
    print("=" * 60)
    print(f"  数据集:       {dataset_path}")
    print(f"  repo_id:      {repo_id}")
    print(f"  输出目录:     {output_dir}")
    print(f"  设备:         {device}")
    print(f"  batch_size:   {batch_size}")
    print(f"  训练步数:     {training_steps}")
    print(f"  chunk_size:   {chunk_size}")
    print(f"  n_action_steps: {n_action_steps}")
    print("=" * 60)

    # 1. 加载数据集元数据，自动推断特征
    print("\n[INFO] 加载数据集元数据...")
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    print(f"  输入特征: {list(input_features.keys())}")
    print(f"  输出特征: {list(output_features.keys())}")

    # 2. 创建 ACT 策略
    print("[INFO] 创建 ACT 策略...")
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        # 以下为默认值，按需覆盖：
        # vision_backbone="resnet18",
        # dim_model=512,
        # n_heads=8,
        # use_vae=True,
        # kl_weight=10.0,
        # optimizer_lr=1e-5,
    )
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"  模型参数量: {n_params:.1f}M")

    # 3. 预处理器 / 后处理器（自动归一化）
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    # 4. 构建 delta_timestamps
    #    ACT 需要当前帧起未来 chunk_size 步的动作作为训练目标
    fps = dataset_metadata.fps
    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
    }
    # 图像特征只取当前帧
    delta_timestamps |= {k: [0] for k in cfg.image_features}

    # 5. 加载数据集
    print(f"[INFO] 加载数据集 (fps={fps})...")
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=device.type != "cpu", drop_last=True, num_workers=num_workers,
    )

    print(f"  数据集帧数: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  每 epoch 约 {len(dataset) // batch_size} 个 batch")

    # 6. 优化器（使用 ACT 内置预设：AdamW, lr=1e-5, wd=1e-4）
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # 7. 训练循环
    print(f"\n[INFO] 开始训练 ({training_steps} steps)...\n")
    step = 0
    done = False
    t0 = time.time()

    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                elapsed = time.time() - t0
                speed = step / max(elapsed, 1e-6)
                msg = (
                    f"step {step:>6d}/{training_steps} | "
                    f"loss: {loss.item():.4f}"
                )
                if "l1_loss" in loss_dict:
                    msg += f" | L1: {loss_dict['l1_loss']:.4f}"
                if "kld_loss" in loss_dict:
                    msg += f" | KLD: {loss_dict['kld_loss']:.4f}"
                msg += f" | {speed:.1f} steps/s"
                print(msg)

            if step > 0 and step % save_freq == 0:
                ckpt_dir = output_dir / f"checkpoint_{step}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                print(f"  → saved checkpoint: {ckpt_dir}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # 8. 保存最终模型
    elapsed = time.time() - t0
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    print(f"\n{'=' * 60}")
    print(f"  训练完成!")
    print(f"  总耗时: {elapsed / 60:.1f} min")
    print(f"  模型保存: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
