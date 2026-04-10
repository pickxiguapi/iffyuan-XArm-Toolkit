"""Data collector: env + cameras + teleop → Zarr dataset.

Manages the full collect loop:
  1. Init env / cameras / SpaceMouse
  2. Per-episode: reset → move to task start pos → wait for start signal
     → record steps → save buffer → repeat
  3. Compute episode_ends metadata

Reference: reference/collect.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import zarr

from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.collect")


# ---------------------------------------------------------------------------
# Keyboard listener (non-blocking, raw terminal)
# ---------------------------------------------------------------------------

class _KeyboardListener:
    """Non-blocking keyboard reader in raw terminal mode."""

    def __init__(self):
        import termios
        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)

    def start(self):
        import tty
        tty.setraw(sys.stdin.fileno())

    def stop(self):
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None


# ---------------------------------------------------------------------------
# Zarr dataset helper
# ---------------------------------------------------------------------------

def _open_or_create_zarr(
    path: str,
    image_shape: tuple[int, ...] = (3, 240, 320),
) -> tuple[zarr.Group, zarr.Group, int]:
    """Open existing or create new Zarr dataset.

    Returns (data_group, meta_group, start_episode).
    """
    dataset_path = str(path)
    exists = os.path.exists(dataset_path)

    if exists:
        logger.info("Opening existing dataset: %s", dataset_path)
        ds = zarr.open(dataset_path, "a")
        data = ds["data"]
        meta = ds["meta"]
        if "episode" in data:
            start_ep = len(np.unique(data["episode"][:]))
            logger.info("Found %d existing episodes", start_ep)
        else:
            start_ep = 0
        return data, meta, start_ep

    logger.info("Creating new dataset: %s", dataset_path)
    ds = zarr.open(dataset_path, "w")
    data = ds.create_group("data")
    meta = ds.create_group("meta")

    # Image compressor: Blosc zstd for RGB images
    try:
        from numcodecs import Blosc
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        compressor = None

    data.require_dataset(
        "rgb_arm", shape=(0, *image_shape), dtype=np.uint8,
        chunks=(1, *image_shape), compressor=compressor,
    )
    data.require_dataset(
        "rgb_fix", shape=(0, *image_shape), dtype=np.uint8,
        chunks=(1, *image_shape), compressor=compressor,
    )
    data.require_dataset("pos", shape=(0, 6), dtype=np.float32)
    data.require_dataset("force", shape=(0, 6), dtype=np.float32)
    data.require_dataset("action", shape=(0, 6), dtype=np.float32)
    data.require_dataset("gripper_state", shape=(0, 1), dtype=np.float32)
    data.require_dataset("gripper_action", shape=(0, 1), dtype=np.float32)
    data.require_dataset("episode", shape=(0,), dtype=np.uint16)

    return data, meta, 0


def _compute_episode_ends(data: zarr.Group, meta: zarr.Group):
    """Recompute ``meta/episode_ends`` from ``data/episode``."""
    all_ep = data["episode"][:]
    unique_eps = np.unique(all_ep)
    ends = []
    running = 0
    for ep in unique_eps:
        running += int(np.sum(all_ep == ep))
        ends.append(running)
    if "episode_ends" in meta:
        del meta["episode_ends"]
    meta.require_dataset("episode_ends", shape=(len(ends),), dtype=np.uint32)
    meta["episode_ends"][:] = np.array(ends, dtype=np.uint32)
    logger.info("episode_ends: %s (total %d episodes)", ends, len(ends))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Collector:
    """Orchestrates data collection: env + cameras + teleop → Zarr.

    Parameters
    ----------
    env : XArmEnv
        Robot environment (delta_eef mode, use_force=True recommended).
    cam_arm : RealsenseEnv
        Arm-mounted camera.
    cam_fix : RealsenseEnv
        Fixed camera.
    agent : SpacemouseAgent
        Teleoperation agent.
    dataset_path : str
        Path to Zarr dataset (created if not exists).
    task_config : dict | None
        Task-specific config with keys:
        - ``start_bias``: [x, y, z] offset from reset pose (default [0,0,0])
        - ``random_bias``: {x: [lo, hi], y: [lo, hi]} for randomised start (optional)
        - ``gripper_always_closed``: bool (for stamp-like tasks)
    num_episodes : int
        Number of episodes to collect in this session.
    image_size : tuple
        Target (W, H) for saved images (default 320×240).
    warmup_time : float
        Seconds to wait after init before first episode.
    """

    def __init__(
        self,
        env,
        cam_arm,
        cam_fix,
        agent,
        dataset_path: str = "datasets/demo.zarr",
        task_config: dict | None = None,
        num_episodes: int = 3,
        image_size: tuple[int, int] = (320, 240),
        warmup_time: float = 1.0,
    ):
        self.env = env
        self.cam_arm = cam_arm
        self.cam_fix = cam_fix
        self.agent = agent

        self.dataset_path = dataset_path
        self.task_cfg = task_config or {}
        self.num_episodes = num_episodes
        self.image_w, self.image_h = image_size
        self.warmup_time = warmup_time

        self._gripper_always_closed = self.task_cfg.get("gripper_always_closed", False)

    # ------------------------------------------------------------------
    # Task start position
    # ------------------------------------------------------------------

    def _get_start_bias(self) -> np.ndarray:
        """Compute [x, y, z, 0, 0, 0] bias for task start position."""
        base = np.array(self.task_cfg.get("start_bias", [0, 0, 0]), dtype=np.float64)

        # Optional random offset
        rand_cfg = self.task_cfg.get("random_bias", {})
        for i, axis in enumerate(["x", "y", "z"]):
            if axis in rand_cfg:
                lo, hi = rand_cfg[axis]
                base[i] += np.random.uniform(lo, hi)

        return np.array([base[0], base[1], base[2], 0, 0, 0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the collection session (blocking)."""
        # Open dataset
        image_shape = (3, self.image_h, self.image_w)
        data, meta, start_ep = _open_or_create_zarr(
            self.dataset_path, image_shape=image_shape,
        )

        # Initial reset
        obs = self.env.reset(close_gripper=True)
        self.cam_arm.step()
        self.cam_fix.step()
        time.sleep(0.5)

        # Move to task start position
        bias = self._get_start_bias()
        obs = self.env.step(bias, gripper_action=0, speed=100)
        self.cam_arm.step()
        self.cam_fix.step()
        logger.info("Initial position: %s", obs["goal_pos"][:3])

        logger.info("Warming up... (%.1fs)", self.warmup_time)
        time.sleep(self.warmup_time)

        kb = _KeyboardListener()

        try:
            for ep_idx in range(self.num_episodes):
                current_ep = start_ep + ep_idx
                self._run_episode(current_ep, ep_idx, data, kb)

                # Reset for next episode
                if ep_idx < self.num_episodes - 1:
                    logger.info("Resetting for next episode...")
                    obs = self.env.reset(close_gripper=True)
                    self.cam_arm.step()
                    self.cam_fix.step()
                    time.sleep(0.5)

                    bias = self._get_start_bias()
                    obs = self.env.step(bias, gripper_action=0, speed=100)
                    self.cam_arm.step()
                    self.cam_fix.step()
                    time.sleep(1)

        finally:
            kb.stop()

        # Compute episode_ends
        _compute_episode_ends(data, meta)
        logger.info(
            "Done! %d new episodes saved to %s",
            self.num_episodes, self.dataset_path,
        )

    # ------------------------------------------------------------------

    def _run_episode(
        self,
        current_ep: int,
        ep_idx: int,
        data: zarr.Group,
        kb: _KeyboardListener,
    ):
        """Record one episode."""
        print(
            f"\r\n=== Episode {current_ep} ({ep_idx + 1}/{self.num_episodes}) ===\r\n"
            f"  Space: start recording | Enter: end recording | Ctrl+C: abort\r\n"
        )

        if ep_idx == 0:
            kb.start()

        # --- Wait for start (space key) ---
        # During wait: buttons adjust gripper, but arm stays still
        goal_gripper = 0
        while True:
            _action, buttons = self.agent.act()
            if buttons[0]:
                goal_gripper = min(goal_gripper + 40, 840)
            elif buttons[1]:
                goal_gripper = max(goal_gripper - 40, 0)

            obs = self.env.step([0, 0, 0, 0, 0, 0], gripper_action=goal_gripper, speed=100)
            self.cam_arm.step()
            self.cam_fix.step()

            key = kb.get_key()
            if key == " ":
                break
            elif key == "\x03":
                raise KeyboardInterrupt
            time.sleep(0.1)

        # --- Zero force sensor ---
        if self.env.use_force:
            logger.info("Zeroing force sensor...")
            self.env.reset_force_sensor_zero()
            time.sleep(0.2)

        print(f"\r\n  Recording episode {current_ep}... (Enter to stop)\r\n")

        # --- Record ---
        buffer: dict[str, list] = {
            "rgb_arm": [], "rgb_fix": [],
            "pos": [], "force": [], "action": [],
            "gripper_state": [], "gripper_action": [],
            "episode": [],
        }

        goal_gripper = 0
        steps = 0
        t_start = time.time()

        while True:
            key = kb.get_key()
            if key == "\r":
                print(f"\r\n  Episode {current_ep} ended.\r\n")
                break
            elif key == "\x03":
                raise KeyboardInterrupt

            # SpaceMouse action
            action, buttons = self.agent.act()
            if buttons[0]:
                goal_gripper = 840
            elif buttons[1]:
                goal_gripper = 0

            # Camera observations (before step, matches reference)
            obs_arm = self.cam_arm.step()
            obs_fix = self.cam_fix.step()

            # Extract & resize images
            rgb_arm = np.asarray(obs_arm["rgbd"].color) if "rgbd" in obs_arm else np.asarray(obs_arm["rgb"])
            rgb_fix = np.asarray(obs_fix["rgbd"].color) if "rgbd" in obs_fix else np.asarray(obs_fix["rgb"])
            rgb_arm = cv2.resize(rgb_arm, (self.image_w, self.image_h))
            rgb_fix = cv2.resize(rgb_fix, (self.image_w, self.image_h))

            # Robot step
            obs = self.env.step(action, gripper_action=goal_gripper, speed=100)

            # Gripper state/action
            if self._gripper_always_closed:
                gs = 0.0
                ga = 0.0
            else:
                gs = 0.0 if obs["gripper_position"] <= 420 else 1.0
                ga = 1.0 if goal_gripper == 840 else 0.0

            # Force (fallback to zeros if no force sensor)
            force = obs["ext_force"] if obs["ext_force"] is not None else np.zeros(6)

            # Append to buffer
            buffer["rgb_arm"].append(rgb_arm.transpose(2, 0, 1)[None])  # (1, 3, H, W)
            buffer["rgb_fix"].append(rgb_fix.transpose(2, 0, 1)[None])
            buffer["pos"].append(obs["goal_pos"].astype(np.float32)[None])
            buffer["force"].append(np.asarray(force, dtype=np.float32)[None])
            buffer["action"].append(np.asarray(action, dtype=np.float32)[None])
            buffer["gripper_state"].append(np.array([[gs]], dtype=np.float32))
            buffer["gripper_action"].append(np.array([[ga]], dtype=np.float32))
            buffer["episode"].append(np.array([current_ep], dtype=np.uint16))

            steps += 1

            # Print stats every 50 steps
            if steps % 50 == 0:
                elapsed = time.time() - t_start
                fps = steps / elapsed if elapsed > 0 else 0
                fz = force[2] if obs["ext_force"] is not None else 0
                print(
                    f"\r\n  Step {steps:5d} | FPS {fps:.1f} | "
                    f"force_z {fz:.1f} | gripper {obs['gripper_position']:.0f}\r\n"
                )

        # --- Save buffer ---
        if steps == 0:
            logger.warning("Episode %d: 0 steps, skipping save.", current_ep)
            return

        logger.info("Episode %d: saving %d steps...", current_ep, steps)
        for key, val in buffer.items():
            data[key].append(np.concatenate(val, axis=0))

        logger.info("Episode %d saved.", current_ep)
