"""Process SisFall dataset and convert to training format.

SisFall format:
- Filename: {Subject}_{Activity}_{Trial}.txt
- Subject: SA01-SA23 (young adults), SE01-SE15 (elderly)
- Activity: D01-D19 (ADL), F01-F15 (Falls)
- Data: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z (200Hz)

Usage:
    python -m ml.process_sisfall --input-dir SisFall --output-dir ml/data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np

# Activity categories
FALL_ACTIVITIES = {f"F{i:02d}" for i in range(1, 16)}  # F01-F15 are falls
ADL_ACTIVITIES = {f"D{i:02d}" for i in range(1, 20)}   # D01-D19 are ADLs


def parse_filename(filename: str) -> dict | None:
    """Parse SisFall filename to extract metadata.

    Actual format: {Activity}_{Subject}_R{Trial}.txt (e.g., D14_SE11_R03.txt)
    """
    pattern = r"(D\d+|F\d+)_(S[AE]\d+)_R(\d+)\.txt"
    match = re.match(pattern, filename)
    if not match:
        return None

    activity, subject, trial = match.groups()
    return {
        "subject": subject,
        "activity": activity,
        "trial": int(trial),
        "is_elderly": subject.startswith("SE"),
    }


def load_sisfall_file(filepath: Path) -> np.ndarray | None:
    """Load a SisFall .txt file.

    Format: Each line contains comma-separated values:
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z (some files have timestamps)
    """
    try:
        data = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                # Try to extract 6 sensor values (skip timestamp if present)
                values = []
                for p in parts:
                    try:
                        values.append(float(p.strip()))
                    except ValueError:
                        continue
                # Take last 6 values (acc + gyro)
                if len(values) >= 6:
                    data.append(values[-6:])
                elif len(values) >= 3:
                    # Only accelerometer
                    data.append(values[-3:] + [0, 0, 0])

        if not data:
            return None
        return np.array(data, dtype=np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_features(data: np.ndarray, window_size: int = 128, stride: int = 64) -> np.ndarray:
    """Extract statistical features from IMU data."""
    features_list = []

    for start in range(0, max(1, len(data) - window_size + 1), stride):
        window = data[start:start + window_size]
        if len(window) < window_size:
            pad = np.zeros((window_size - len(window), data.shape[1]))
            window = np.vstack([window, pad])

        feat = []
        for ch in range(min(6, window.shape[1])):
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
        if window.shape[1] >= 3:
            acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
            feat.extend([np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag)])
        if window.shape[1] >= 6:
            gyr_mag = np.sqrt(np.sum(window[:, 3:6] ** 2, axis=1))
            feat.extend([np.mean(gyr_mag), np.std(gyr_mag), np.max(gyr_mag)])

        features_list.append(feat)

    if not features_list:
        return np.zeros((1, 42), dtype=np.float32)

    return np.array(features_list, dtype=np.float32)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    max_samples_per_class: int = 500,
) -> Tuple[int, int]:
    """Process SisFall dataset and save to npz files."""
    fall_dir = output_dir / "fall"
    normal_dir = output_dir / "normal"
    fall_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Find all .txt files recursively
    txt_files = list(input_dir.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files")

    fall_count = 0
    normal_count = 0

    for txt_path in txt_files:
        meta = parse_filename(txt_path.name)
        if meta is None:
            continue

        activity = meta["activity"]
        is_fall = activity in FALL_ACTIVITIES

        # Check limits
        if is_fall and fall_count >= max_samples_per_class:
            continue
        if not is_fall and normal_count >= max_samples_per_class:
            continue

        data = load_sisfall_file(txt_path)
        if data is None or len(data) < 50:
            continue

        features = extract_features(data)

        label = "fall" if is_fall else "normal"
        out_dir = fall_dir if is_fall else normal_dir
        count = fall_count if is_fall else normal_count

        out_path = out_dir / f"sisfall_{label}_{count:04d}.npz"
        np.savez_compressed(out_path, data=features, label=label)

        if is_fall:
            fall_count += 1
        else:
            normal_count += 1

        if (fall_count + normal_count) % 100 == 0:
            print(f"Processed {fall_count} falls, {normal_count} normal samples...")

    return fall_count, normal_count


def main():
    parser = argparse.ArgumentParser(description="Process SisFall dataset")
    parser.add_argument("--input-dir", type=str, default="SisFall",
                        help="Directory containing SisFall data")
    parser.add_argument("--output-dir", type=str, default="ml/data",
                        help="Output directory for npz files")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples per class")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found")
        return

    print(f"Processing SisFall from {input_dir}")

    fall_count, normal_count = process_dataset(
        input_dir, output_dir,
        max_samples_per_class=args.max_samples,
    )

    print(f"\nDone! Saved {fall_count} fall samples, {normal_count} normal samples")


if __name__ == "__main__":
    main()
