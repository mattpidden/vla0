#!/usr/bin/env python3
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

"""
Real robot client for VLA-0 inference on SO-100.

Usage:
    python rv_train/deploy/robot_client.py

Set TASK and SERVER_URL before running.
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import requests
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower.config_so101_follower import \
    SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# ── Config ───────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:10000"

# Task instruction sent to the model
TASK = "Reorient the block."

# Camera indices: right=10 (3p1), left=12 (3p2) — must match training order
RIGHT_CAMERA_IDX = 10
LEFT_CAMERA_IDX = 12

# Robot serial port
ROBOT_PORT = "/dev/ttyACM1"

# Control frequency for executing the returned action chunk.
# Must match the dataset FPS (30) so each action step covers the same
# time interval the model was trained on.
CONTROL_HZ = 30

# Number of action steps to execute before querying the model again.
# The model returns `horizon=8` steps; executing all of them before re-querying
# works well at the ~4 Hz inference speed documented in the VLA-0 paper.
STEPS_BEFORE_REQUERY = 8
# ─────────────────────────────────────────────────────────────────────────────

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _rgb_as_base64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.array(rgb, dtype=np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def predict(images: list, state: list, instruction: str) -> np.ndarray:
    """POST to /predict_base64, returns actions of shape (horizon, 6)."""
    payload = {
        "base64_rgb": [_rgb_as_base64(img) for img in images],
        "state": state,
        "instr": instruction,
    }
    resp = requests.post(f"{SERVER_URL}/predict_base64", json=payload, timeout=300)
    resp.raise_for_status()
    return np.array(resp.json())  # (horizon, 6)


def main():
    # Verify server is up before connecting hardware
    try:
        requests.get(f"{SERVER_URL}/health", timeout=5).raise_for_status()
        print(f"Server healthy at {SERVER_URL}")
    except Exception as e:
        print(f"Cannot reach server at {SERVER_URL}: {e}")
        return

    cameras = {
        # right camera (3p1) must be first to match training
        "right": OpenCVCameraConfig(
            index_or_path=RIGHT_CAMERA_IDX, width=640, height=480, fps=30
        ),
        "left": OpenCVCameraConfig(
            index_or_path=LEFT_CAMERA_IDX, width=640, height=480, fps=30
        ),
    }
    robot_cfg = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="blue_follower",
        cameras=cameras,
        use_degrees=True,
        calibration_dir=Path(
            "/vol/dissolve/matt/hf_cache/lerobot/calibration/robots/so_follower"
        ),
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print("Robot connected")

    dt = 1.0 / CONTROL_HZ

    try:
        step = 0
        actions = None

        while True:
            # Re-query the model every STEPS_BEFORE_REQUERY steps (or on first step)
            if step % STEPS_BEFORE_REQUERY == 0:
                obs = robot.get_observation()

                # Images: right first (3p1), then left (3p2)
                right_img = obs["right"]
                left_img = obs["left"]

                state = [obs[f"{m}.pos"] for m in MOTOR_NAMES]

                print(
                    f"Querying model (step {step}) | state={[f'{v:.1f}' for v in state]}"
                )

                # Debug: save camera frames so you can verify framing/content
                if step == 0:
                    import cv2

                    cv2.imwrite(
                        "/vol/dissolve/matt/models/vla0/tmp/vla0_right_cam.jpg",
                        cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        "/vol/dissolve/matt/models/vla0/tmp/vla0_left_cam.jpg",
                        cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR),
                    )
                    print(
                        "Saved camera frames to /tmp/vla0_right_cam.jpg and /tmp/vla0_left_cam.jpg"
                    )

                t_query = time.perf_counter()
                actions = predict([right_img, left_img], state, TASK)
                print(
                    f"Model responded in {time.perf_counter() - t_query:.2f}s | actions shape: {actions.shape}"
                )

                action_idx = 0

            # Execute one action step
            t0 = time.perf_counter()
            action = actions[action_idx]
            action_dict = {
                f"{m}.pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)
            }
            robot.send_action(action_dict)
            action_idx += 1

            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        robot.disconnect()
        print("Robot disconnected")


if __name__ == "__main__":
    main()
