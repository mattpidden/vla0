#!/usr/bin/env python3
"""One-time conversion of physical-intelligence/libero from v2.0 to v3.0 format.

Run this once before training:
    cd /vol/dissolve/matt/models/vla0
    python convert_libero_v20_to_v30.py

What it does:
  1. Creates meta/tasks.parquet from meta/tasks.jsonl
  2. Creates meta/episodes/chunk-000/file-000.parquet from meta/episodes.jsonl,
     adding dataset_from_index / dataset_to_index columns (cumulative frame counts)
  3. Updates meta/info.json: codebase_version -> v3.0, adds required v3.0 fields
  4. Downloads data files from HuggingFace (main branch, v2.0 per-episode parquets)

After conversion the v3.0 LeRobotDataset loader works because:
  - Version check passes (info.json says v3.0)
  - load_tasks / load_episodes find the new parquet files
  - load_hf_dataset uses glob("*/*.parquet") which matches v2.0 per-episode files
  - No re-download happens (data already cached)
"""

import json
from pathlib import Path

import jsonlines
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "physical-intelligence/libero"
LIBERO_ROOT = Path("/vol/dissolve/matt/hf_cache/lerobot/physical-intelligence/libero")


def convert_tasks(root: Path) -> None:
    tasks_parquet = root / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        print("  meta/tasks.parquet already exists, skipping")
        return

    tasks_jsonl = root / "meta" / "tasks.jsonl"
    with jsonlines.open(tasks_jsonl) as reader:
        tasks = sorted(reader, key=lambda x: x["task_index"])

    # v3.0 format: task strings ARE the DataFrame index.
    # meta.tasks.iloc[i].name must return the task string, not a number.
    df = pd.DataFrame(
        {"task_index": [t["task_index"] for t in tasks]},
        index=[t["task"] for t in tasks],
    )
    df.to_parquet(tasks_parquet, index=True)
    print(f"  Created meta/tasks.parquet ({len(df)} tasks)")


def convert_episodes(root: Path) -> None:
    episodes_parquet = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet.exists():
        print("  meta/episodes parquet already exists, skipping")
        return

    episodes_jsonl = root / "meta" / "episodes.jsonl"
    with jsonlines.open(episodes_jsonl) as reader:
        episodes = sorted(reader, key=lambda x: x["episode_index"])

    cumsum = 0
    rows = []
    for ep in episodes:
        length = ep["length"]
        rows.append(
            {
                "episode_index": ep["episode_index"],
                "tasks": ep["tasks"],
                "length": length,
                "dataset_from_index": cumsum,
                "dataset_to_index": cumsum + length,
                # data/chunk_index and data/file_index are only used for selective
                # episode downloads (episodes != None). Full-dataset training never
                # calls get_data_file_path(), so 0,0 is safe here.
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        )
        cumsum += length

    df = pd.DataFrame(rows)
    episodes_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(episodes_parquet, index=False)
    print(
        f"  Created meta/episodes/chunk-000/file-000.parquet "
        f"({len(df)} episodes, {cumsum} total frames)"
    )


def update_info(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") == "v3.0":
        print("  meta/info.json already at v3.0, skipping")
        return

    info["codebase_version"] = "v3.0"
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = 100
    info["video_files_size_in_mb"] = 0
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["fps"] = int(info["fps"])

    for key, ft in info["features"].items():
        if ft["dtype"] not in ("video", "image"):
            ft["fps"] = info["fps"]

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("  Updated meta/info.json to v3.0")


def download_data(root: Path, repo_id: str) -> None:
    data_dir = root / "data"
    if data_dir.exists() and any(data_dir.rglob("*.parquet")):
        n = len(list(data_dir.rglob("*.parquet")))
        print(f"  Data already cached ({n} parquet files), skipping download")
        return

    print(f"  Downloading data from {repo_id} (main branch, v2.0 format)...")
    print("  This may take a while (~10-15 GB)...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=None,  # main branch = v2.0 format
        local_dir=root,
        allow_patterns=["data/**"],
        local_files_only=False,
    )
    n = len(list(data_dir.rglob("*.parquet")))
    print(f"  Downloaded {n} episode parquet files")


if __name__ == "__main__":
    print(f"Converting {REPO_ID} → v3.0 format in:\n  {LIBERO_ROOT}\n")

    print("Step 1: Converting tasks...")
    convert_tasks(LIBERO_ROOT)

    print("Step 2: Converting episodes metadata...")
    convert_episodes(LIBERO_ROOT)

    print("Step 3: Updating info.json...")
    update_info(LIBERO_ROOT)

    print("Step 4: Downloading data files...")
    download_data(LIBERO_ROOT, REPO_ID)

    print("\nDone! You can now run LIBERO training.")
