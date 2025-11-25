import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Configuration dictionary to centralize parameters
CONFIG = {
    'timesteps': 70,
    'window_size': 10,  # Aligned with main.py
    'stride': 5,  # Aligned with main.py
    'nominal_class': 0,
}


def load_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load flight data and labels from a .npz file.

    Args:
        file_path (str): Path to the .npz file containing 'data' and 'label' arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data array (n_flights, n_timesteps, n_vars) and labels array (n_flights,).

    Raises:
        ValueError: If data or labels have invalid shapes.
        Exception: If file loading fails.
    """
    try:
        full_data = np.load(file_path, mmap_mode='r')
        data = full_data['data']
        labels = full_data['label']
        if len(data.shape) != 3 or labels.shape[0] != data.shape[0]:
            raise ValueError(f"Invalid shape: data {data.shape}, labels {labels.shape}")
        logger.info(f"Loaded data shape: {data.shape}, labels shape: {labels.shape}")
        return data, labels
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def trim_timesteps(data: np.ndarray, timesteps: int = CONFIG['timesteps']) -> np.ndarray:
    """
    Trim time series to the last specified number of timesteps (default: 70 seconds prior to 1,000 ft AGL).

    Args:
        data (np.ndarray): Input data of shape (n_flights, n_timesteps, n_vars).
        timesteps (int): Number of timesteps to retain. Default: 70.

    Returns:
        np.ndarray: Trimmed data of shape (n_flights, timesteps, n_vars).

    Raises:
        ValueError: If data is not 3D or has insufficient timesteps.
        Exception: If trimming fails.
    """
    try:
        if len(data.shape) != 3:
            raise ValueError(f"Data must be 3D, got {data.shape}")
        current_timesteps = data.shape[1]
        if current_timesteps < timesteps:
            raise ValueError(f"Data has {current_timesteps} timesteps, need {timesteps}")
        start_idx = current_timesteps - timesteps
        trimmed_data = data[:, start_idx:start_idx + timesteps, :]
        logger.info(f"Trimmed from {current_timesteps} to {timesteps}, shape: {trimmed_data.shape}")
        return trimmed_data
    except Exception as e:
        logger.error(f"Error trimming timesteps: {str(e)}")
        raise


def apply_time_windows(
        data: np.ndarray,
        window_size: int = CONFIG['window_size'],
        stride: int = CONFIG['stride']
) -> np.ndarray:
    """
    Apply sliding windows to capture temporal changes, as an extension to Garcia et al. (2024).

    Args:
        data (np.ndarray): Input data of shape (n_flights, n_timesteps, n_vars).
        window_size (int): Size of each window. Default: 10.
        stride (int): Stride between windows. Default: 5.

    Returns:
        np.ndarray: Windowed data of shape (n_flights, n_windows, window_size, n_vars).

    Raises:
        ValueError: If data is not 3D or no valid windows can be created.
    """
    try:
        if len(data.shape) != 3:
            raise ValueError(f"Data must be 3D, got {data.shape}")
        n_flights, timesteps, n_vars = data.shape

        windows = []
        start = 0
        while start + window_size <= timesteps:
            windows.append((start, start + window_size))
            start += stride
        n_windows = len(windows)
        if n_windows == 0:
            raise ValueError(f"No windows with window_size={window_size}, stride={stride}, timesteps={timesteps}")
        logger.info(f"Applying {n_windows} windows of size {window_size}: {windows}")

        windowed_data = np.zeros((n_flights, n_windows, window_size, n_vars), dtype=np.float32)
        for win_idx, (start, end) in enumerate(windows):
            windowed_data[:, win_idx, :, :] = data[:, start:end, :]
            if (end - start) != window_size:
                raise ValueError(f"Window {win_idx} size mismatch: expected {window_size}, got {end - start}")

        logger.info(f"Windowed data shape: {windowed_data.shape}")
        return windowed_data
    except Exception as e:
        logger.error(f"Error applying time windows: {str(e)}")
        raise



