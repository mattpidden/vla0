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
TASK = "Push the apple to the block."

# Camera indices — must match training order:
#   3p1 (first image) = wrist camera  (observation.images.wrist)
#   3p2 (second image) = middle camera (observation.images.middle)
WRIST_CAMERA_IDX = 3
MIDDLE_CAMERA_IDX = 8

# Robot serial port
ROBOT_PORT = "/dev/ttyACM0"

# Control frequency. Must match dataset FPS (10) so each action step covers
# the same time interval the model was trained on.
CONTROL_HZ = 10

# Action horizon returned by the model.
HORIZON = 8

# Set to False to execute a chunk of actions before re-querying (original behaviour).
# Set to True to query every step and blend overlapping predictions.
USE_ENSEMBLE = True

# How many overlapping predictions to blend (ensemble mode) or how many actions
# to execute per chunk before re-querying (non-ensemble mode).
# Lower = less averaging, more reactive. 4 is a good middle ground vs 8.
ENSEMBLE_DEPTH = 4

# Temporal ensembling decay. Lower = more weight on older predictions (smoother).
# Higher = newer predictions dominate. 0.1 is a good starting point.
ENSEMBLE_LAMBDA = 0.5
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


# Rolling buffer for temporal ensembling.
# stored_actions[0] = remaining predictions from 1 step ago (newest stored)
# stored_actions[k][0] = that buffer's prediction for the current step
_ensemble_buf: list[np.ndarray] = []


def ensemble_action(new_actions: np.ndarray) -> np.ndarray:
    """
    Blend new_actions[0] with buffered predictions of the current step.
    new_actions: (horizon, action_dim)
    Returns: (action_dim,) blended action.
    """
    # Collect candidates: newest first
    candidates = [new_actions[0]] + [buf[0] for buf in _ensemble_buf if len(buf) > 0]
    n = len(candidates)
    weights = np.array([np.exp(-ENSEMBLE_LAMBDA * k) for k in range(n)])
    weights /= weights.sum()
    blended = np.einsum("i,ij->j", weights, np.stack(candidates))

    # Advance buffer: shift each entry by one step (consume the just-executed step)
    advanced = [buf[1:] for buf in _ensemble_buf if len(buf) > 1]
    # Prepend remaining steps of this query as the newest stored entry
    if HORIZON > 1:
        advanced = [new_actions[1:]] + advanced
    _ensemble_buf[:] = advanced[: ENSEMBLE_DEPTH - 1]

    return blended


def main():
    # Verify server is up before connecting hardware
    try:
        requests.get(f"{SERVER_URL}/health", timeout=5).raise_for_status()
        print(f"Server healthy at {SERVER_URL}")
    except Exception as e:
        print(f"Cannot reach server at {SERVER_URL}: {e}")
        return

    cameras = {
        # wrist (3p1) must be first, middle (3p2) second — matches training order
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST_CAMERA_IDX, width=640, height=480, fps=30
        ),
        "middle": OpenCVCameraConfig(
            index_or_path=MIDDLE_CAMERA_IDX, width=640, height=480, fps=30
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
        action_idx = 0

        while True:
            t0 = time.perf_counter()

            if USE_ENSEMBLE:
                # Query every step and blend overlapping predictions
                obs = robot.get_observation()
                wrist_img = obs["wrist"]
                middle_img = obs["middle"]
                state = [obs[f"{m}.pos"] for m in MOTOR_NAMES]

                if step == 0:
                    import cv2

                    cv2.imwrite(
                        "/vol/dissolve/matt/models/vla0/tmp/vla0_wrist_cam.jpg",
                        cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        "/vol/dissolve/matt/models/vla0/tmp/vla0_middle_cam.jpg",
                        cv2.cvtColor(middle_img, cv2.COLOR_RGB2BGR),
                    )
                    print(
                        "Saved camera frames to tmp/vla0_wrist_cam.jpg and tmp/vla0_middle_cam.jpg"
                    )

                t_query = time.perf_counter()
                new_actions = predict([wrist_img, middle_img], state, TASK)
                print(
                    f"step {step} | inference {time.perf_counter() - t_query:.2f}s"
                    f" | ensemble depth {len(_ensemble_buf)+1}"
                    f" | state={[f'{v:.1f}' for v in state]}"
                )
                action = ensemble_action(new_actions)

            else:
                # Re-query every STEPS_BEFORE_REQUERY steps
                if step % ENSEMBLE_DEPTH == 0:
                    obs = robot.get_observation()
                    wrist_img = obs["wrist"]
                    middle_img = obs["middle"]
                    state = [obs[f"{m}.pos"] for m in MOTOR_NAMES]

                    if step == 0:
                        import cv2

                        cv2.imwrite(
                            "/vol/dissolve/matt/models/vla0/tmp/vla0_wrist_cam.jpg",
                            cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR),
                        )
                        cv2.imwrite(
                            "/vol/dissolve/matt/models/vla0/tmp/vla0_middle_cam.jpg",
                            cv2.cvtColor(middle_img, cv2.COLOR_RGB2BGR),
                        )
                        print(
                            "Saved camera frames to tmp/vla0_wrist_cam.jpg and tmp/vla0_middle_cam.jpg"
                        )

                    t_query = time.perf_counter()
                    actions = predict([wrist_img, middle_img], state, TASK)
                    print(
                        f"step {step} | inference {time.perf_counter() - t_query:.2f}s"
                        f" | state={[f'{v:.1f}' for v in state]}"
                    )
                    action_idx = 0

                action = actions[action_idx]
                action_idx += 1

            action_dict = {
                f"{m}.pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)
            }
            robot.send_action(action_dict)

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
