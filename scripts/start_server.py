#!/usr/bin/env python3
"""GPU-side VLA server startup script.

Loads a Pi0.5 model via openpi and starts the WebSocket inference server
so that robot-side clients can request action predictions.

Usage::

    python scripts/start_server.py \\
        --config pi05_droid \\
        --port 10093

    # 自定义 checkpoint（本地已下载）:
    python scripts/start_server.py \\
        --config pi05_droid \\
        --ckpt-dir /data/checkpoints/pi05_droid \\
        --port 10093
"""

from __future__ import annotations

import argparse

import numpy as np

from xarm_toolkit.deploy.server import VLAServer
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("start_server")


class Pi05Policy:
    """Pi0.5 VLA policy wrapper using openpi.

    Wraps ``openpi`` 的 trained policy，将 VLAClient 发来的
    ``images / instruction / state`` 转换为 openpi 期望的 example dict，
    调用 ``policy.infer()`` 返回 action chunk。

    Parameters
    ----------
    config_name : str
        openpi config 名称，如 ``"pi05_droid"``。
    ckpt_dir : str | None
        本地 checkpoint 目录。为 None 时自动从 GCS 下载。
    """

    def __init__(self, config_name: str = "pi05_droid", ckpt_dir: str | None = None):
        self.config_name = config_name
        self._load_model(ckpt_dir)

    def _load_model(self, ckpt_dir: str | None):
        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.shared import download

        logger.info("Loading openpi config: %s", self.config_name)
        config = _config.get_config(self.config_name)

        if ckpt_dir is None:
            # 自动从 GCS 下载默认 checkpoint
            default_gs = f"gs://openpi-assets/checkpoints/{self.config_name}"
            logger.info("Downloading checkpoint from %s ...", default_gs)
            ckpt_dir = download.maybe_download(default_gs)
        else:
            logger.info("Using local checkpoint: %s", ckpt_dir)

        logger.info("Creating trained policy ...")
        self.policy = policy_config.create_trained_policy(config, ckpt_dir)
        logger.info("Pi0.5 policy ready.")

    def predict_action(self, **kwargs) -> dict:
        """Run inference and return predicted actions.

        VLAClient 发送的 kwargs:
            images : list[np.ndarray]  — [arm_rgb, fix_rgb]，RGB uint8
            instruction : str          — 语言指令
            state : np.ndarray         — 机器人状态 (servo_angle 或 cart_pos)

        Returns
        -------
        dict
            ``{"actions": np.ndarray}`` — shape ``(horizon, action_dim)``
        """
        images = kwargs.get("images", [])
        instruction = kwargs.get("instruction", "")
        state = kwargs.get("state")

        logger.info(
            "Inference: %d images, instruction=%r, state_shape=%s",
            len(images),
            instruction[:60],
            state.shape if hasattr(state, "shape") else "N/A",
        )

        # -----------------------------------------------------------------
        # 组装 openpi example dict
        # openpi pi05_droid 期望的 key 格式:
        #   "observation/exterior_image_1_left" — 固定相机 RGB
        #   "observation/wrist_image_left"      — 臂上相机 RGB
        #   "observation/state"                 — 机器人状态
        #   "prompt"                            — 语言指令
        #
        # NOTE(iff): 如果你的 config 用了不同的 key 名称，在这里改
        # -----------------------------------------------------------------
        example = {"prompt": instruction}

        # 图像映射: images[0] = arm (wrist), images[1] = fix (exterior)
        if len(images) >= 1:
            example["observation/wrist_image_left"] = images[0]
        if len(images) >= 2:
            example["observation/exterior_image_1_left"] = images[1]

        # 机器人状态
        if state is not None:
            example["observation/state"] = np.asarray(state, dtype=np.float32)

        # -----------------------------------------------------------------
        # 推理
        # -----------------------------------------------------------------
        result = self.policy.infer(example)
        actions = result["actions"]  # np.ndarray, shape (horizon, action_dim)

        logger.info("Inference done, actions shape: %s", actions.shape)
        return {"actions": np.asarray(actions, dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser(description="Start Pi0.5 VLA inference server")
    parser.add_argument("--config", type=str, default="pi05_droid",
                        help="openpi config name (default: pi05_droid)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Local checkpoint directory (auto-download if omitted)")
    parser.add_argument("--port", type=int, default=10093,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server bind address")
    parser.add_argument("--idle-timeout", type=float, default=0,
                        help="Connection idle timeout in seconds (0 = disabled)")
    args = parser.parse_args()

    # Load model
    policy = Pi05Policy(
        config_name=args.config,
        ckpt_dir=args.ckpt_dir,
    )

    # Start server
    server = VLAServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "model": "pi0.5",
            "config": args.config,
            "action_format": "absolute_joint",
        },
    )

    logger.info("Starting VLA server on %s:%d", args.host, args.port)
    server.run()


if __name__ == "__main__":
    main()
