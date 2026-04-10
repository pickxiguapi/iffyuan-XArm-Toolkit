#!/usr/bin/env python3
"""验证双 RealSense 相机是否正常工作.

Usage:
    python scripts/test_cameras.py                # 双相机预览
    python scripts/test_cameras.py --arm-only      # 只测臂上相机
    python scripts/test_cameras.py --fix-only      # 只测固定相机
    python scripts/test_cameras.py --no-preview     # 不开窗口，只打印信息
"""

from __future__ import annotations

import argparse
import time
import traceback

import cv2
import numpy as np

from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("test_cameras")

CAM_ARM_SERIAL = "327122075644"  # D435i (arm-mounted)
CAM_FIX_SERIAL = "f1271506"      # L515  (fixed)


def main():
    parser = argparse.ArgumentParser(description="RealSense camera verification")
    parser.add_argument("--arm-only", action="store_true", help="Only test arm camera")
    parser.add_argument("--fix-only", action="store_true", help="Only test fixed camera")
    parser.add_argument("--no-preview", action="store_true", help="No GUI window, just print info")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to capture in no-preview mode")
    args = parser.parse_args()

    cameras: dict[str, RealsenseEnv] = {}

    try:
        # ---- Init cameras ----
        if not args.fix_only:
            logger.info("Opening arm camera (D435i) serial=%s ...", CAM_ARM_SERIAL)
            cameras["arm"] = RealsenseEnv(serial=CAM_ARM_SERIAL, mode="rgb")
            logger.info("✓ Arm camera opened successfully")

        if not args.arm_only:
            logger.info("Opening fixed camera (L515) serial=%s ...", CAM_FIX_SERIAL)
            cameras["fix"] = RealsenseEnv(serial=CAM_FIX_SERIAL, mode="rgb")
            logger.info("✓ Fixed camera opened successfully")

        if not cameras:
            logger.error("No cameras to test!")
            return

        # ---- Capture first frame & print info ----
        for name, cam in cameras.items():
            obs = cam.step()
            rgb = np.asarray(obs["rgb"])
            logger.info(
                "[%s] First frame: shape=%s dtype=%s min=%d max=%d  intrinsic=%s  depth_scale=%.4f",
                name, rgb.shape, rgb.dtype, rgb.min(), rgb.max(),
                cam.intrinsic_matrix.shape, cam.depth_scale,
            )

        # ---- Preview / FPS test ----
        if args.no_preview:
            logger.info("Running %d frames for FPS measurement ...", args.frames)
            t0 = time.time()
            for i in range(args.frames):
                for name, cam in cameras.items():
                    cam.step()
            elapsed = time.time() - t0
            fps = args.frames / elapsed
            logger.info("✓ Captured %d frames in %.2fs → %.1f FPS (per-camera)", args.frames, elapsed, fps)
        else:
            logger.info("Starting preview (press 'q' to quit) ...")
            frame_count = 0
            t0 = time.time()

            while True:
                frames = {}
                for name, cam in cameras.items():
                    obs = cam.step()
                    frames[name] = np.asarray(obs["rgb"])

                # Show each camera
                for name, rgb in frames.items():
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"camera_{name}", bgr)

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info("Frame %d, avg FPS: %.1f", frame_count, frame_count / elapsed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except Exception:
        logger.error("Camera test failed:\n%s", traceback.format_exc())
    finally:
        for name, cam in cameras.items():
            logger.info("Closing %s camera ...", name)
            cam.cleanup()
        cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    main()
