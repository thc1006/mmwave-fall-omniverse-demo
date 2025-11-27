"""Process FallAllD dataset and convert to training format.

FallAllD format:
- Filename: S{SubjectID}_D{Device}_A{ActivityID}_T{TrialNo}_{Type}.dat
- Type: A=Accelerometer, G=Gyroscope, M=Magnetometer, B=Barometer
- Device: 1=Neck, 2=Wrist, 3=Waist
- ActivityID: 101-135=Falls, 1-44=ADLs (normal activities)

Usage:
    python -m ml.process_fallalld --input-dir FallAllD/FallAllD --output-dir ml/data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Activity mapping
FALL_ACTIVITIES = set(range(101, 136))  # 101-135 are falls
NORMAL_ACTIVITIES = set(range(1, 45))   # 1-44 are ADLs


def parse_filename(filename: str) -> Dict[str, int | str] | None:
    """Parse FallAllD filename to extract metadata."""
    pattern = r"S(\d+)_D(\d)_A(\d+)_T(\d+)_([AGMB])\.dat"
    match = re.match(pattern, filename)
    if not match:
        return None

    subject_id, device_id, activity_id, trial_no, data_type = match.groups()
    device_map = {"1": "Neck", "2": "Wrist", "3": "Waist"}

    return {
        "subject_id": int(subject_id),
        "device": device_map.get(device_id, "Unknown"),
        "activity_id": int(activity_id),
        "trial_no": int(trial_no),
        "data_type": data_type,
    }


def load_dat_file(filepath: Path) -> np.ndarray:
    """Load a .dat file (CSV format with 3 columns: x, y, z)."""
    return np.genfromtxt(filepath, delimiter=",", dtype=np.float32)


def get_trial_files(input_dir: Path, subject: int, device: int, activity: int, trial: int) -> Dict[str, Path]:
    """Get all sensor files for a specific trial."""
    prefix = f"S{subject:02d}_D{device}_A{activity:03d}_T{trial:02d}"
    files = {}
    for suffix in ["A", "G", "M", "B"]:
        path = input_dir / f"{prefix}_{suffix}.dat"
        if path.exists():
            files[suffix] = path
    return files


def extract_features(acc: np.ndarray, gyr: np.ndarray, window_size: int = 128, stride: int = 64) -> np.ndarray:
    """Extract statistical features from accelerometer and gyroscope data."""
    # Combine acc and gyr: [N, 6]
    if acc.shape[0] != gyr.shape[0]:
        min_len = min(acc.shape[0], gyr.shape[0])
        acc = acc[:min_len]
        gyr = gyr[:min_len]

    combined = np.concatenate([acc, gyr], axis=1)  # [N, 6]

    features_list = []
    for start in range(0, len(combined) - window_size + 1, stride):
        window = combined[start:start + window_size]

        # Statistical features per channel
        feat = []
        for ch in range(6):
            ch_data = window[:, ch]
            feat.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data),
                np.min(ch_data),
                np.sqrt(np.mean(ch_data ** 2)),  # RMS
                np.sum(ch_data ** 2),  # Energy
            ])

        # Magnitude features
        acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
        gyr_mag = np.sqrt(np.sum(window[:, 3:] ** 2, axis=1))

        feat.extend([
            np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag),
            np.mean(gyr_mag), np.std(gyr_mag), np.max(gyr_mag),
        ])

        features_list.append(feat)

    if not features_list:
        # If sequence too short, pad
        window = combined
        if len(window) < window_size:
            pad = np.zeros((window_size - len(window), 6))
            window = np.vstack([window, pad])

        feat = []
        for ch in range(6):
            ch_data = window[:, ch]
            feat.extend([
                np.mean(ch_data), np.std(ch_data), np.max(ch_data),
                np.min(ch_data), np.sqrt(np.mean(ch_data ** 2)), np.sum(ch_data ** 2),
            ])
        acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
        gyr_mag = np.sqrt(np.sum(window[:, 3:] ** 2, axis=1))
        feat.extend([
            np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag),
            np.mean(gyr_mag), np.std(gyr_mag), np.max(gyr_mag),
        ])
        features_list.append(feat)

    return np.array(features_list, dtype=np.float32)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    device_filter: str = "Waist",
    max_samples_per_class: int = 500,
) -> Tuple[int, int]:
    """Process FallAllD dataset and save to npz files."""
    fall_dir = output_dir / "fall"
    normal_dir = output_dir / "normal"
    fall_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Find all accelerometer files (use as base)
    acc_files = list(input_dir.glob("*_A.dat"))
    print(f"Found {len(acc_files)} accelerometer files")

    fall_count = 0
    normal_count = 0

    device_map_inv = {"Neck": "1", "Wrist": "2", "Waist": "3"}
    target_device = device_map_inv.get(device_filter, "3")

    for acc_path in acc_files:
        meta = parse_filename(acc_path.name)
        if meta is None:
            continue

        # Filter by device
        device_id = acc_path.name.split("_")[1][1]
        if device_id != target_device:
            continue

        activity_id = meta["activity_id"]
        is_fall = activity_id in FALL_ACTIVITIES

        # Check limits
        if is_fall and fall_count >= max_samples_per_class:
            continue
        if not is_fall and normal_count >= max_samples_per_class:
            continue

        # Load gyroscope file
        gyr_path = acc_path.parent / acc_path.name.replace("_A.dat", "_G.dat")
        if not gyr_path.exists():
            continue

        try:
            acc_data = load_dat_file(acc_path)
            gyr_data = load_dat_file(gyr_path)

            if acc_data.ndim != 2 or gyr_data.ndim != 2:
                continue

            features = extract_features(acc_data, gyr_data)

            label = "fall" if is_fall else "normal"
            out_dir = fall_dir if is_fall else normal_dir
            count = fall_count if is_fall else normal_count

            out_path = out_dir / f"{label}_{count:04d}.npz"
            np.savez_compressed(out_path, data=features, label=label)

            if is_fall:
                fall_count += 1
            else:
                normal_count += 1

            if (fall_count + normal_count) % 100 == 0:
                print(f"Processed {fall_count} falls, {normal_count} normal samples...")

        except Exception as e:
            print(f"Error processing {acc_path}: {e}")
            continue

    return fall_count, normal_count


def main():
    parser = argparse.ArgumentParser(description="Process FallAllD dataset")
    parser.add_argument("--input-dir", type=str, default="FallAllD/FallAllD",
                        help="Directory containing FallAllD .dat files")
    parser.add_argument("--output-dir", type=str, default="ml/data",
                        help="Output directory for npz files")
    parser.add_argument("--device", type=str, default="Waist",
                        choices=["Neck", "Wrist", "Waist"],
                        help="Sensor location to use")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples per class")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found")
        return

    print(f"Processing FallAllD from {input_dir}")
    print(f"Using device: {args.device}")

    fall_count, normal_count = process_dataset(
        input_dir, output_dir,
        device_filter=args.device,
        max_samples_per_class=args.max_samples,
    )

    print(f"\nDone! Saved {fall_count} fall samples, {normal_count} normal samples")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
