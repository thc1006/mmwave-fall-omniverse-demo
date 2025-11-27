"""IMU Data Pipeline for Fall Detection.

This module provides utilities for loading, preprocessing, and augmenting
IMU sensor data (accelerometer + gyroscope) from datasets like FallAllD.

Features:
- CSV data loading for IMU sensors (3-axis accel + 3-axis gyro)
- Statistical and spectral feature extraction
- Sliding window segmentation with configurable overlap
- Data augmentation (noise injection, time stretching)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Spectral features will be unavailable.")


@dataclass
class IMUConfig:
    """Configuration for IMU data pipeline."""

    # Sensor configuration
    sampling_rate: float = 100.0  # Hz (FallAllD uses 238Hz, adjust as needed)
    accel_columns: List[str] = field(default_factory=lambda: ["Acc_X", "Acc_Y", "Acc_Z"])
    gyro_columns: List[str] = field(default_factory=lambda: ["Gyr_X", "Gyr_Y", "Gyr_Z"])

    # Windowing parameters
    window_size: int = 128  # Number of frames per window
    stride: int = 64  # Stride for sliding window (overlap = window_size - stride)

    # Feature extraction
    extract_spectral: bool = True  # Whether to extract frequency-domain features

    # Augmentation parameters
    noise_std: float = 0.05  # Standard deviation for Gaussian noise
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)  # Range for time stretching


class IMUDataLoader:
    """Loader for IMU data from CSV files."""

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config = config or IMUConfig()

    def load_csv(
        self,
        file_path: Union[str, Path],
        accel_cols: Optional[List[str]] = None,
        gyro_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Load IMU data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            accel_cols: Column names for accelerometer data (X, Y, Z).
            gyro_cols: Column names for gyroscope data (X, Y, Z).

        Returns:
            Array of shape [N, 6] where columns are [Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z].
        """
        import pandas as pd

        accel_cols = accel_cols or self.config.accel_columns
        gyro_cols = gyro_cols or self.config.gyro_columns

        df = pd.read_csv(file_path)

        # Check if columns exist
        all_cols = accel_cols + gyro_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

        data = df[all_cols].values.astype(np.float32)
        return data

    def load_fallalld_file(
        self,
        file_path: Union[str, Path],
        sensor_location: str = "Waist",
    ) -> np.ndarray:
        """Load a single FallAllD dataset file.

        FallAllD format: Each file contains data from 3 sensor locations
        (Neck, Wrist, Waist) with columns like Acc_X_Neck, Gyr_Z_Waist, etc.

        Args:
            file_path: Path to the FallAllD CSV file.
            sensor_location: One of "Neck", "Wrist", "Waist".

        Returns:
            Array of shape [N, 6] for the specified sensor location.
        """
        import pandas as pd

        df = pd.read_csv(file_path)

        # FallAllD column naming convention
        accel_cols = [f"Acc_X_{sensor_location}", f"Acc_Y_{sensor_location}", f"Acc_Z_{sensor_location}"]
        gyro_cols = [f"Gyr_X_{sensor_location}", f"Gyr_Y_{sensor_location}", f"Gyr_Z_{sensor_location}"]

        all_cols = accel_cols + gyro_cols

        # Check for columns (FallAllD may have different naming)
        if not all(c in df.columns for c in all_cols):
            # Try alternative naming: AccX_Waist, etc.
            accel_cols = [f"AccX_{sensor_location}", f"AccY_{sensor_location}", f"AccZ_{sensor_location}"]
            gyro_cols = [f"GyrX_{sensor_location}", f"GyrY_{sensor_location}", f"GyrZ_{sensor_location}"]
            all_cols = accel_cols + gyro_cols

        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Cannot find IMU columns for {sensor_location}. Available: {list(df.columns)}")

        data = df[all_cols].values.astype(np.float32)
        return data


class FeatureExtractor:
    """Extract statistical and spectral features from IMU windows."""

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config = config or IMUConfig()

    def extract_statistical_features(self, window: np.ndarray) -> np.ndarray:
        """Extract time-domain statistical features from a window.

        Args:
            window: Array of shape [window_size, channels] (e.g., [128, 6]).

        Returns:
            Feature vector containing per-channel statistics.
        """
        features = []

        for ch in range(window.shape[1]):
            channel_data = window[:, ch]

            # Basic statistics
            features.append(np.mean(channel_data))  # Mean
            features.append(np.std(channel_data))   # Standard deviation
            features.append(np.max(channel_data))   # Maximum
            features.append(np.min(channel_data))   # Minimum
            features.append(np.max(channel_data) - np.min(channel_data))  # Range

            # Higher-order statistics
            features.append(np.median(channel_data))  # Median
            features.append(np.percentile(channel_data, 25))  # Q1
            features.append(np.percentile(channel_data, 75))  # Q3

            # Signal energy
            energy = np.sum(channel_data ** 2) / len(channel_data)
            features.append(energy)

            # Root mean square
            rms = np.sqrt(energy)
            features.append(rms)

            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(channel_data)))) / (2 * len(channel_data))
            features.append(zero_crossings)

            # Signal magnitude area (for motion intensity)
            sma = np.sum(np.abs(channel_data)) / len(channel_data)
            features.append(sma)

        # Cross-channel features (acceleration magnitude, gyro magnitude)
        if window.shape[1] >= 3:
            # Acceleration magnitude statistics
            acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
            features.append(np.mean(acc_mag))
            features.append(np.std(acc_mag))
            features.append(np.max(acc_mag))
            features.append(np.min(acc_mag))

        if window.shape[1] >= 6:
            # Gyroscope magnitude statistics
            gyro_mag = np.sqrt(np.sum(window[:, 3:6] ** 2, axis=1))
            features.append(np.mean(gyro_mag))
            features.append(np.std(gyro_mag))
            features.append(np.max(gyro_mag))
            features.append(np.min(gyro_mag))

        return np.array(features, dtype=np.float32)

    def extract_spectral_features(self, window: np.ndarray) -> np.ndarray:
        """Extract frequency-domain features from a window.

        Args:
            window: Array of shape [window_size, channels].

        Returns:
            Spectral feature vector.
        """
        if not HAS_SCIPY:
            return np.array([], dtype=np.float32)

        features = []
        fs = self.config.sampling_rate
        n = window.shape[0]
        freqs = fftfreq(n, 1/fs)[:n//2]

        for ch in range(window.shape[1]):
            channel_data = window[:, ch]

            # FFT
            fft_vals = np.abs(fft(channel_data))[:n//2]

            # Spectral energy
            spectral_energy = np.sum(fft_vals ** 2)
            features.append(spectral_energy)

            # Dominant frequency
            if np.sum(fft_vals) > 0:
                dominant_freq_idx = np.argmax(fft_vals)
                dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0
            else:
                dominant_freq = 0
            features.append(dominant_freq)

            # Spectral centroid
            if np.sum(fft_vals) > 0:
                spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            else:
                spectral_centroid = 0
            features.append(spectral_centroid)

            # Spectral entropy
            fft_normalized = fft_vals / (np.sum(fft_vals) + 1e-10)
            spectral_entropy = -np.sum(fft_normalized * np.log2(fft_normalized + 1e-10))
            features.append(spectral_entropy)

            # Band powers (low: 0-5Hz, mid: 5-15Hz, high: 15+ Hz)
            low_band = np.sum(fft_vals[(freqs >= 0) & (freqs < 5)] ** 2)
            mid_band = np.sum(fft_vals[(freqs >= 5) & (freqs < 15)] ** 2)
            high_band = np.sum(fft_vals[freqs >= 15] ** 2)
            features.extend([low_band, mid_band, high_band])

        return np.array(features, dtype=np.float32)

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract all features from a window.

        Args:
            window: Array of shape [window_size, channels].

        Returns:
            Combined feature vector.
        """
        stat_features = self.extract_statistical_features(window)

        if self.config.extract_spectral and HAS_SCIPY:
            spec_features = self.extract_spectral_features(window)
            return np.concatenate([stat_features, spec_features])

        return stat_features


class SlidingWindowSegmenter:
    """Segment time series data into overlapping windows."""

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config = config or IMUConfig()

    def segment(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """Segment data into overlapping windows.

        Args:
            data: Array of shape [N, channels].
            window_size: Number of frames per window.
            stride: Step size between windows.

        Returns:
            Array of shape [num_windows, window_size, channels].
        """
        window_size = window_size or self.config.window_size
        stride = stride or self.config.stride

        n_samples = data.shape[0]

        if n_samples < window_size:
            # Pad if data is shorter than window
            pad_width = window_size - n_samples
            data = np.pad(data, ((0, pad_width), (0, 0)), mode="edge")
            n_samples = window_size

        n_windows = (n_samples - window_size) // stride + 1

        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            windows.append(data[start:end])

        return np.array(windows, dtype=np.float32)


class DataAugmenter:
    """Apply data augmentation to IMU windows."""

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config = config or IMUConfig()
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def add_noise(
        self,
        window: np.ndarray,
        noise_std: Optional[float] = None,
    ) -> np.ndarray:
        """Add Gaussian noise to the window.

        Args:
            window: Array of shape [window_size, channels] or [num_windows, window_size, channels].
            noise_std: Standard deviation of the noise.

        Returns:
            Noisy window with the same shape.
        """
        noise_std = noise_std or self.config.noise_std
        noise = self._rng.normal(0, noise_std, size=window.shape).astype(np.float32)
        return window + noise

    def time_stretch(
        self,
        window: np.ndarray,
        stretch_factor: Optional[float] = None,
    ) -> np.ndarray:
        """Apply time stretching to the window.

        Args:
            window: Array of shape [window_size, channels].
            stretch_factor: Factor to stretch (>1 = slower, <1 = faster).
                           If None, randomly sample from config range.

        Returns:
            Time-stretched window (interpolated back to original length).
        """
        if stretch_factor is None:
            low, high = self.config.time_stretch_range
            stretch_factor = self._rng.uniform(low, high)

        original_length = window.shape[0]
        stretched_length = int(original_length * stretch_factor)

        if stretched_length < 2:
            return window

        # Interpolate each channel
        x_original = np.linspace(0, 1, original_length)
        x_stretched = np.linspace(0, 1, stretched_length)

        stretched = np.zeros((stretched_length, window.shape[1]), dtype=np.float32)
        for ch in range(window.shape[1]):
            stretched[:, ch] = np.interp(x_stretched, x_original, window[:, ch])

        # Interpolate back to original length
        x_final = np.linspace(0, 1, original_length)
        x_stretched_norm = np.linspace(0, 1, stretched_length)

        result = np.zeros_like(window)
        for ch in range(window.shape[1]):
            result[:, ch] = np.interp(x_final, x_stretched_norm, stretched[:, ch])

        return result

    def scale_amplitude(
        self,
        window: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> np.ndarray:
        """Scale the amplitude of the window.

        Args:
            window: Array of shape [window_size, channels].
            scale_range: Range for random scaling factor.

        Returns:
            Scaled window.
        """
        scale = self._rng.uniform(scale_range[0], scale_range[1])
        return window * scale

    def random_rotation(
        self,
        window: np.ndarray,
        max_angle: float = 15.0,
    ) -> np.ndarray:
        """Apply small random rotation to 3D sensor axes.

        Simulates slight sensor misalignment or orientation changes.

        Args:
            window: Array of shape [window_size, channels] where channels >= 3.
            max_angle: Maximum rotation angle in degrees.

        Returns:
            Rotated window.
        """
        if window.shape[1] < 3:
            return window

        result = window.copy()

        # Random rotation angles for each axis
        angles = self._rng.uniform(-max_angle, max_angle, size=3)
        angles = np.deg2rad(angles)

        # Rotation matrices
        def rot_x(a):
            return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

        def rot_y(a):
            return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

        def rot_z(a):
            return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        R = rot_x(angles[0]) @ rot_y(angles[1]) @ rot_z(angles[2])

        # Apply rotation to accelerometer (first 3 channels)
        result[:, :3] = (R @ result[:, :3].T).T

        # Apply rotation to gyroscope if present (channels 3-5)
        if window.shape[1] >= 6:
            result[:, 3:6] = (R @ result[:, 3:6].T).T

        return result.astype(np.float32)

    def augment(
        self,
        window: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Apply multiple augmentations to a window.

        Args:
            window: Array of shape [window_size, channels].
            augmentations: List of augmentation names to apply.
                          Options: "noise", "time_stretch", "scale", "rotation".
                          If None, applies all.

        Returns:
            Augmented window.
        """
        if augmentations is None:
            augmentations = ["noise", "time_stretch", "scale", "rotation"]

        result = window.copy()

        for aug in augmentations:
            if aug == "noise":
                result = self.add_noise(result)
            elif aug == "time_stretch":
                result = self.time_stretch(result)
            elif aug == "scale":
                result = self.scale_amplitude(result)
            elif aug == "rotation":
                result = self.random_rotation(result)

        return result


class IMUDataPipeline:
    """Complete pipeline for IMU data processing.

    Combines loading, segmentation, feature extraction, and augmentation.
    """

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config = config or IMUConfig()
        self.loader = IMUDataLoader(config)
        self.segmenter = SlidingWindowSegmenter(config)
        self.feature_extractor = FeatureExtractor(config)
        self.augmenter = DataAugmenter(config)

    def process_file(
        self,
        file_path: Union[str, Path],
        extract_features: bool = True,
        augment: bool = False,
        augment_factor: int = 1,
    ) -> Tuple[np.ndarray, int]:
        """Process a single IMU file through the pipeline.

        Args:
            file_path: Path to CSV file.
            extract_features: Whether to extract features or return raw windows.
            augment: Whether to apply data augmentation.
            augment_factor: Number of augmented copies per original window.

        Returns:
            Tuple of (processed_data, num_windows).
            If extract_features=True: data shape is [num_windows, num_features].
            If extract_features=False: data shape is [num_windows, window_size, channels].
        """
        # Load data
        data = self.loader.load_csv(file_path)

        # Segment into windows
        windows = self.segmenter.segment(data)

        # Augmentation
        if augment and augment_factor > 0:
            augmented = [windows]
            for _ in range(augment_factor):
                aug_windows = np.array([
                    self.augmenter.augment(w) for w in windows
                ])
                augmented.append(aug_windows)
            windows = np.concatenate(augmented, axis=0)

        # Feature extraction
        if extract_features:
            features = np.array([
                self.feature_extractor.extract_features(w) for w in windows
            ])
            return features, len(features)

        return windows, len(windows)

    def process_directory(
        self,
        data_dir: Union[str, Path],
        label: str,
        extract_features: bool = True,
        augment: bool = False,
        augment_factor: int = 1,
        file_pattern: str = "*.csv",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process all files in a directory.

        Args:
            data_dir: Directory containing CSV files.
            label: Label for all files in this directory.
            extract_features: Whether to extract features.
            augment: Whether to augment data.
            augment_factor: Number of augmented copies.
            file_pattern: Glob pattern for files.

        Returns:
            Tuple of (data_array, labels_array).
        """
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob(file_pattern))

        if not files:
            raise ValueError(f"No files matching '{file_pattern}' found in {data_dir}")

        all_data = []
        for f in files:
            try:
                data, _ = self.process_file(f, extract_features, augment, augment_factor)
                all_data.append(data)
            except Exception as e:
                warnings.warn(f"Error processing {f}: {e}")

        if not all_data:
            raise RuntimeError(f"Failed to process any files in {data_dir}")

        combined = np.concatenate(all_data, axis=0)
        labels = np.full(len(combined), label, dtype=object)

        return combined, labels

    def get_feature_dim(self) -> int:
        """Get the expected feature dimension based on config.

        Returns:
            Number of features per window.
        """
        # Create a dummy window to calculate feature dimension
        dummy_window = np.zeros((self.config.window_size, 6), dtype=np.float32)
        features = self.feature_extractor.extract_features(dummy_window)
        return len(features)


def save_to_npz(
    data: np.ndarray,
    label: str,
    output_path: Union[str, Path],
) -> None:
    """Save processed data in the format expected by train_fallnet.py.

    Args:
        data: Array of shape [num_windows, features] or [num_windows, window_size, channels].
        label: Label string (e.g., "fall", "normal").
        output_path: Output .npz file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, data=data, label=label)


def load_from_npz(npz_path: Union[str, Path]) -> Tuple[np.ndarray, str]:
    """Load data from .npz file.

    Args:
        npz_path: Path to .npz file.

    Returns:
        Tuple of (data, label).
    """
    arr = np.load(npz_path, allow_pickle=True)
    data = arr["data"]
    label = str(arr.get("label", "unknown"))
    return data, label
