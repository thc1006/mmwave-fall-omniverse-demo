"""Generate Synthetic IMU Data for Fall Detection.

This script generates realistic synthetic IMU data (accelerometer + gyroscope)
for initial model testing before real data is available.

Patterns simulated:
- Normal activities: walking, standing, sitting, lying
- Fall events: forward fall, backward fall, side fall

The generated data mimics characteristics of real IMU sensors:
- Gravity component in accelerometer
- Movement-induced accelerations
- Angular velocities during motion
- Realistic noise levels

Output: .npz files in ml/data/{normal,fall}/ directories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    # Output settings
    output_dir: str = "ml/data"
    num_normal_samples: int = 100
    num_fall_samples: int = 100

    # Sensor settings
    sampling_rate: float = 100.0  # Hz
    duration: float = 5.0  # seconds per sample

    # IMU characteristics
    gravity: float = 9.81  # m/s^2
    accel_noise_std: float = 0.1  # m/s^2
    gyro_noise_std: float = 0.02  # rad/s

    # Feature dimension (for compatibility with FallNet)
    # If provided, will extract features; otherwise save raw windows
    feature_dim: Optional[int] = 256

    # Random seed
    seed: int = 42


class SyntheticIMUGenerator:
    """Generate synthetic IMU data for fall detection."""

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def _generate_base_signal(self, duration: float) -> Tuple[np.ndarray, int]:
        """Generate time array and base noise.

        Returns:
            Tuple of (time_array, num_samples).
        """
        num_samples = int(duration * self.config.sampling_rate)
        t = np.linspace(0, duration, num_samples)
        return t, num_samples

    def _add_sensor_noise(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add realistic sensor noise to IMU data."""
        accel += self._rng.normal(0, self.config.accel_noise_std, accel.shape)
        gyro += self._rng.normal(0, self.config.gyro_noise_std, gyro.shape)
        return accel, gyro

    def generate_standing(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate standing/stationary data.

        Characteristics:
        - Gravity dominates accelerometer Z-axis
        - Minimal accelerations in X, Y
        - Very small gyroscope readings (slight body sway)
        """
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        # Accelerometer: mainly gravity with small variations
        accel = np.zeros((n, 3), dtype=np.float32)
        accel[:, 2] = self.config.gravity  # Z-axis (vertical)

        # Small body sway
        sway_freq = self._rng.uniform(0.2, 0.5)
        sway_amp = self._rng.uniform(0.05, 0.15)
        accel[:, 0] += sway_amp * np.sin(2 * np.pi * sway_freq * t)
        accel[:, 1] += sway_amp * np.cos(2 * np.pi * sway_freq * t + self._rng.uniform(0, np.pi))

        # Gyroscope: minimal rotation
        gyro = np.zeros((n, 3), dtype=np.float32)
        gyro += self._rng.normal(0, 0.01, (n, 3))

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_walking(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate walking data.

        Characteristics:
        - Periodic acceleration patterns (step frequency ~1-2 Hz)
        - Vertical bouncing motion
        - Forward/backward acceleration bursts
        - Periodic leg swing in gyroscope
        """
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        # Walking parameters
        step_freq = self._rng.uniform(1.5, 2.2)  # Steps per second
        step_amp = self._rng.uniform(1.5, 3.0)  # m/s^2

        accel = np.zeros((n, 3), dtype=np.float32)

        # X-axis: forward/backward (heel strike and push-off)
        accel[:, 0] = step_amp * 0.5 * np.sin(2 * np.pi * step_freq * t)

        # Y-axis: side-to-side sway
        accel[:, 1] = step_amp * 0.3 * np.sin(np.pi * step_freq * t)

        # Z-axis: gravity + vertical bounce (double frequency for both feet)
        accel[:, 2] = self.config.gravity + step_amp * np.abs(np.sin(2 * np.pi * step_freq * t))

        # Gyroscope: leg swing and body rotation
        gyro = np.zeros((n, 3), dtype=np.float32)
        gyro_amp = self._rng.uniform(0.3, 0.6)

        gyro[:, 0] = gyro_amp * np.sin(2 * np.pi * step_freq * t)  # Roll
        gyro[:, 1] = gyro_amp * 0.5 * np.sin(np.pi * step_freq * t)  # Pitch
        gyro[:, 2] = gyro_amp * 0.3 * np.sin(2 * np.pi * step_freq * t + np.pi/4)  # Yaw

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_sitting_down(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate sitting down motion.

        Characteristics:
        - Gradual downward acceleration
        - Hip flexion (pitch rotation)
        - Controlled deceleration at the end
        """
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        # Timing: sitting happens in the middle portion
        sit_start = n // 4
        sit_end = 3 * n // 4

        accel = np.zeros((n, 3), dtype=np.float32)
        accel[:, 2] = self.config.gravity

        gyro = np.zeros((n, 3), dtype=np.float32)

        # Sitting motion envelope
        for i in range(sit_start, sit_end):
            progress = (i - sit_start) / (sit_end - sit_start)

            # Downward acceleration followed by deceleration
            if progress < 0.5:
                # Accelerating down
                accel[i, 2] += -2.0 * np.sin(np.pi * progress * 2)
            else:
                # Decelerating (braking)
                accel[i, 2] += 1.5 * np.sin(np.pi * (progress - 0.5) * 2)

            # Forward lean during sitting
            accel[i, 0] += 0.5 * np.sin(np.pi * progress)

            # Hip flexion (pitch)
            gyro[i, 1] = 0.5 * np.sin(np.pi * progress)

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_lying_down(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate lying down/resting data.

        Characteristics:
        - Gravity distributed across axes (depending on orientation)
        - Very minimal movement
        - Occasional small shifts
        """
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        # Random lying orientation (on back, side, etc.)
        orientation = self._rng.choice(["back", "left_side", "right_side"])

        accel = np.zeros((n, 3), dtype=np.float32)

        if orientation == "back":
            accel[:, 2] = self.config.gravity
        elif orientation == "left_side":
            accel[:, 1] = self.config.gravity
        else:  # right_side
            accel[:, 1] = -self.config.gravity

        # Occasional small movements (breathing, adjusting position)
        breath_freq = self._rng.uniform(0.2, 0.35)
        accel[:, 2] += 0.1 * np.sin(2 * np.pi * breath_freq * t)

        # Very small gyro readings
        gyro = np.zeros((n, 3), dtype=np.float32)
        gyro += self._rng.normal(0, 0.005, (n, 3))

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_forward_fall(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate forward fall event.

        Characteristics:
        - Initial stumble/trip
        - Free-fall phase (reduced gravity sensation)
        - High-impact ground contact
        - Post-fall stillness
        """
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        # Fall timing
        fall_start = n // 3
        fall_duration = n // 4
        impact_point = fall_start + fall_duration
        recovery_end = min(impact_point + n // 8, n)

        accel = np.zeros((n, 3), dtype=np.float32)
        accel[:, 2] = self.config.gravity  # Default standing

        gyro = np.zeros((n, 3), dtype=np.float32)

        # Pre-fall: normal standing with slight disturbance
        for i in range(fall_start):
            accel[i, 0] += 0.2 * np.sin(2 * np.pi * 0.3 * t[i])

        # Fall phase
        fall_progress = 0
        for i in range(fall_start, impact_point):
            fall_progress = (i - fall_start) / fall_duration

            # Reduced gravity sensation (free fall)
            accel[i, 2] = self.config.gravity * (1 - 0.7 * fall_progress)

            # Forward acceleration (falling forward)
            accel[i, 0] = 3.0 * fall_progress

            # High angular velocity (body rotating forward)
            gyro[i, 1] = 2.0 * np.sin(np.pi * fall_progress)  # Pitch (forward rotation)
            gyro[i, 0] = 0.5 * fall_progress  # Some roll

        # Impact phase - high acceleration spike
        if impact_point < n:
            impact_duration = min(10, n - impact_point)
            for j in range(impact_duration):
                idx = impact_point + j
                if idx < n:
                    # Sharp deceleration spike
                    spike = np.exp(-j * 0.5) * 4.0 * self.config.gravity
                    accel[idx, 0] = -spike * 0.3  # Forward impact
                    accel[idx, 2] = self.config.gravity + spike

                    # Impact vibration in gyro
                    gyro[idx, :] = self._rng.uniform(-2, 2, 3) * np.exp(-j * 0.3)

        # Post-fall: lying still (gravity now on different axis)
        for i in range(recovery_end, n):
            accel[i, 0] = self.config.gravity * 0.7  # Lying face down
            accel[i, 2] = self.config.gravity * 0.3
            gyro[i, :] = 0

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_backward_fall(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate backward fall event."""
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        fall_start = n // 3
        fall_duration = n // 4
        impact_point = fall_start + fall_duration

        accel = np.zeros((n, 3), dtype=np.float32)
        accel[:, 2] = self.config.gravity

        gyro = np.zeros((n, 3), dtype=np.float32)

        # Fall phase - backward
        for i in range(fall_start, impact_point):
            fall_progress = (i - fall_start) / fall_duration

            accel[i, 2] = self.config.gravity * (1 - 0.7 * fall_progress)
            accel[i, 0] = -2.5 * fall_progress  # Backward

            # Backward pitch
            gyro[i, 1] = -1.8 * np.sin(np.pi * fall_progress)

        # Impact
        if impact_point < n:
            impact_duration = min(12, n - impact_point)
            for j in range(impact_duration):
                idx = impact_point + j
                if idx < n:
                    spike = np.exp(-j * 0.4) * 3.5 * self.config.gravity
                    accel[idx, 0] = spike * 0.2
                    accel[idx, 2] = self.config.gravity + spike
                    gyro[idx, :] = self._rng.uniform(-1.5, 1.5, 3) * np.exp(-j * 0.3)

        # Post-fall: lying on back
        post_impact = impact_point + 15
        for i in range(post_impact, n):
            if i < n:
                accel[i, 0] = -self.config.gravity * 0.1
                accel[i, 2] = self.config.gravity
                gyro[i, :] = 0

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_side_fall(self, duration: Optional[float] = None) -> np.ndarray:
        """Generate sideways fall event."""
        duration = duration or self.config.duration
        t, n = self._generate_base_signal(duration)

        fall_start = n // 3
        fall_duration = n // 4
        impact_point = fall_start + fall_duration

        # Randomly choose left or right
        direction = self._rng.choice([-1, 1])

        accel = np.zeros((n, 3), dtype=np.float32)
        accel[:, 2] = self.config.gravity

        gyro = np.zeros((n, 3), dtype=np.float32)

        # Fall phase - sideways
        for i in range(fall_start, impact_point):
            fall_progress = (i - fall_start) / fall_duration

            accel[i, 2] = self.config.gravity * (1 - 0.6 * fall_progress)
            accel[i, 1] = direction * 2.8 * fall_progress

            # Roll rotation (falling sideways)
            gyro[i, 0] = direction * 2.2 * np.sin(np.pi * fall_progress)

        # Impact
        if impact_point < n:
            impact_duration = min(10, n - impact_point)
            for j in range(impact_duration):
                idx = impact_point + j
                if idx < n:
                    spike = np.exp(-j * 0.45) * 3.8 * self.config.gravity
                    accel[idx, 1] = direction * spike * 0.4
                    accel[idx, 2] = self.config.gravity + spike
                    gyro[idx, :] = self._rng.uniform(-2, 2, 3) * np.exp(-j * 0.35)

        # Post-fall: lying on side
        post_impact = impact_point + 12
        for i in range(post_impact, n):
            if i < n:
                accel[i, 1] = direction * self.config.gravity
                accel[i, 2] = 0.2 * self.config.gravity
                gyro[i, :] = 0

        accel, gyro = self._add_sensor_noise(accel, gyro)
        return np.hstack([accel, gyro])

    def generate_normal_activity(self) -> np.ndarray:
        """Generate a random normal (non-fall) activity."""
        activities = [
            self.generate_standing,
            self.generate_walking,
            self.generate_sitting_down,
            self.generate_lying_down,
        ]
        activity = self._rng.choice(activities)
        return activity()

    def generate_fall_activity(self) -> np.ndarray:
        """Generate a random fall event."""
        falls = [
            self.generate_forward_fall,
            self.generate_backward_fall,
            self.generate_side_fall,
        ]
        fall = self._rng.choice(falls)
        return fall()


def extract_simple_features(data: np.ndarray, target_dim: int = 256) -> np.ndarray:
    """Extract simple statistical features to match FallNet input dimension.

    Args:
        data: Raw IMU data of shape [num_samples, 6].
        target_dim: Target feature dimension.

    Returns:
        Feature vector of shape [target_dim].
    """
    features = []

    # Per-channel statistics
    for ch in range(data.shape[1]):
        channel = data[:, ch]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.max(channel),
            np.min(channel),
            np.median(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75),
            np.sum(channel ** 2) / len(channel),  # Energy
            np.sqrt(np.sum(channel ** 2) / len(channel)),  # RMS
            np.sum(np.abs(np.diff(np.sign(channel)))) / (2 * len(channel)),  # Zero-crossing rate
        ])

    # Magnitude features
    acc_mag = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(data[:, 3:6] ** 2, axis=1))

    for mag in [acc_mag, gyro_mag]:
        features.extend([
            np.mean(mag),
            np.std(mag),
            np.max(mag),
            np.min(mag),
            np.percentile(mag, 10),
            np.percentile(mag, 90),
        ])

    # Correlation between axes
    for i in range(3):
        for j in range(i + 1, 3):
            if np.std(data[:, i]) > 0 and np.std(data[:, j]) > 0:
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
            else:
                features.append(0)

    # Pad or truncate to target dimension
    features = np.array(features, dtype=np.float32)

    if len(features) < target_dim:
        # Pad with derived features (higher-order statistics, windowed features)
        num_windows = 4
        window_size = data.shape[0] // num_windows

        for w in range(num_windows):
            start = w * window_size
            end = start + window_size
            window = data[start:end]

            for ch in range(min(data.shape[1], 3)):  # Just accelerometer for padding
                features = np.append(features, [
                    np.mean(window[:, ch]),
                    np.std(window[:, ch]),
                    np.max(window[:, ch]) - np.min(window[:, ch]),
                ])

        # Final padding with zeros if still short
        if len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)), mode="constant")

    return features[:target_dim]


def generate_dataset(config: SyntheticDataConfig) -> None:
    """Generate synthetic dataset and save to disk.

    Args:
        config: Configuration for data generation.
    """
    generator = SyntheticIMUGenerator(config)

    output_dir = Path(config.output_dir)
    normal_dir = output_dir / "normal"
    fall_dir = output_dir / "fall"

    normal_dir.mkdir(parents=True, exist_ok=True)
    fall_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {config.num_normal_samples} normal activity samples...")
    for i in range(config.num_normal_samples):
        data = generator.generate_normal_activity()

        if config.feature_dim:
            features = extract_simple_features(data, config.feature_dim)
            # Save as [1, feature_dim] to match expected format
            save_data = features.reshape(1, -1)
        else:
            save_data = data

        output_path = normal_dir / f"normal_{i:04d}.npz"
        np.savez(output_path, data=save_data, label="normal")

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{config.num_normal_samples} normal samples")

    print(f"Generating {config.num_fall_samples} fall event samples...")
    for i in range(config.num_fall_samples):
        data = generator.generate_fall_activity()

        if config.feature_dim:
            features = extract_simple_features(data, config.feature_dim)
            save_data = features.reshape(1, -1)
        else:
            save_data = data

        output_path = fall_dir / f"fall_{i:04d}.npz"
        np.savez(output_path, data=save_data, label="fall")

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{config.num_fall_samples} fall samples")

    print(f"\nDataset generated successfully!")
    print(f"  Normal samples: {normal_dir} ({config.num_normal_samples} files)")
    print(f"  Fall samples: {fall_dir} ({config.num_fall_samples} files)")

    if config.feature_dim:
        print(f"  Feature dimension: {config.feature_dim}")
    else:
        print(f"  Raw data shape: ({int(config.duration * config.sampling_rate)}, 6)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic IMU data for fall detection testing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/data",
        help="Output directory for generated data.",
    )
    parser.add_argument(
        "--num-normal",
        type=int,
        default=100,
        help="Number of normal activity samples to generate.",
    )
    parser.add_argument(
        "--num-fall",
        type=int,
        default=100,
        help="Number of fall event samples to generate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of each sample in seconds.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=100.0,
        help="Sampling rate in Hz.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=256,
        help="Feature dimension for FallNet. Set to 0 to save raw data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = SyntheticDataConfig(
        output_dir=args.output_dir,
        num_normal_samples=args.num_normal,
        num_fall_samples=args.num_fall,
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        feature_dim=args.feature_dim if args.feature_dim > 0 else None,
        seed=args.seed,
    )

    generate_dataset(config)
