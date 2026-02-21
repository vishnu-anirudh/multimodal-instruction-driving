from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.style as mplstyle

if "seaborn-whitegrid" not in mplstyle.available and "seaborn-v0_8-whitegrid" in mplstyle.available:
    mplstyle.library["seaborn-whitegrid"] = mplstyle.library["seaborn-v0_8-whitegrid"]

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from doscenes.data.instructions import load_instruction_csvs


def pose_xy_yaw(nusc: NuScenes, sample_token: str) -> tuple[float, float, float]:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sample_data = nusc.get("sample_data", lidar_token)
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    x, y = pose["translation"][:2]
    yaw = Quaternion(pose["rotation"]).yaw_pitch_roll[0]
    return x, y, yaw


def to_ego_frame(points: list[tuple[float, float]], x0: float, y0: float, yaw0: float) -> np.ndarray:
    cos_yaw = np.cos(-yaw0)
    sin_yaw = np.sin(-yaw0)
    rel = []
    for x, y in points:
        dx, dy = x - x0, y - y0
        xr = dx * cos_yaw - dy * sin_yaw
        yr = dx * sin_yaw + dy * cos_yaw
        rel.append([xr, yr])
    return np.asarray(rel, dtype=np.float32)


def raster_map(
    nusc_map: NuScenesMap,
    x: float,
    y: float,
    yaw: float,
    patch_size: float,
    raster_size: int,
) -> np.ndarray:
    layers = ["lane", "road_segment", "drivable_area", "ped_crossing", "walkway"]
    available = [layer for layer in layers if layer in nusc_map.layer_names]
    patch_box = (x, y, patch_size, patch_size)
    patch_angle = np.degrees(yaw)
    try:
        mask = nusc_map.get_map_mask(patch_box, patch_angle, available, (raster_size, raster_size))
        return mask.astype(np.float32).reshape(-1)
    except TypeError:
        # Fallback for shapely MultiPolygon incompatibilities.
        channels = max(len(available), 1)
        return np.zeros((channels, raster_size, raster_size), dtype=np.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", type=str, required=True)
    parser.add_argument("--instructions_dir", type=str, required=True)
    parser.add_argument("--scene_count", type=int, default=20)
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--future_steps", type=int, default=12)
    parser.add_argument("--patch_size", type=float, default=50.0)
    parser.add_argument("--raster_size", type=int, default=64)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False)
    instructions = load_instruction_csvs(args.instructions_dir)
    instructions = instructions[instructions["scene_number"].notna()].copy()
    instructions["scene_number"] = instructions["scene_number"].astype(int)

    rows = []
    used = 0
    for scene in nusc.scene:
        scene_name = scene["name"]
        try:
            scene_number = int(scene_name.split("-")[-1])
        except Exception:
            continue
        inst_rows = instructions[instructions["scene_number"] == scene_number]
        inst_rows = inst_rows[inst_rows["instruction"].notna() & (inst_rows["instruction"].astype(str).str.len() > 0)]
        if inst_rows.empty:
            continue

        sample_tokens = []
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            token = nusc.get("sample", token)["next"]

        idx = args.history_steps
        if idx + args.future_steps >= len(sample_tokens):
            continue

        current = sample_tokens[idx]
        history_tokens = sample_tokens[idx - args.history_steps : idx]
        future_tokens = sample_tokens[idx + 1 : idx + 1 + args.future_steps]

        x0, y0, yaw0 = pose_xy_yaw(nusc, current)
        history_points = [pose_xy_yaw(nusc, t)[:2] for t in history_tokens]
        future_points = [pose_xy_yaw(nusc, t)[:2] for t in future_tokens]

        history = to_ego_frame(history_points, x0, y0, yaw0)
        future = to_ego_frame(future_points, x0, y0, yaw0)

        log = nusc.get("log", scene["log_token"])
        nusc_map = NuScenesMap(dataroot=args.nuscenes_root, map_name=log["location"])
        map_features = raster_map(nusc_map, x0, y0, yaw0, args.patch_size, args.raster_size)

        instruction = inst_rows.iloc[0]["instruction"]
        instruction_type = inst_rows.iloc[0].get("instruction_type", None)

        rows.append(
            {
                "scene_id": scene_name,
                "history": history.tolist(),
                "future": future.tolist(),
                "instruction": instruction,
                "instruction_type": instruction_type,
                "map_features": map_features.tolist(),
            }
        )
        used += 1
        if used >= args.scene_count:
            break

    if not rows:
        raise RuntimeError("No scenes matched instructions and history/future constraints.")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
