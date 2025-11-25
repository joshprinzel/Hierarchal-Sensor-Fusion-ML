import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import gc
import psutil
import os
import matplotlib.pyplot as plt
from time import time

logger = logging.getLogger(__name__)

# Precompute pattern-to-symbol mapping as per Garcia et al. (2024), Table 5
PATTERN_TO_SYMBOL = {
    (1, 2, 3, 4): 1, (1, 2, 4, 3): 2, (1, 3, 2, 4): 3, (1, 3, 4, 2): 4,
    (1, 4, 2, 3): 5, (1, 4, 3, 2): 6, (2, 1, 3, 4): 7, (2, 1, 4, 3): 8,
    (2, 3, 1, 4): 9, (2, 3, 4, 1): 10, (2, 4, 1, 3): 11, (2, 4, 3, 1): 12,
    (3, 1, 2, 4): 13, (3, 1, 4, 2): 14, (3, 2, 1, 4): 15, (3, 2, 4, 1): 16,
    (3, 4, 1, 2): 17, (3, 4, 2, 1): 18, (4, 1, 2, 3): 19, (4, 1, 3, 2): 20,
    (4, 2, 1, 3): 21, (4, 2, 3, 1): 22, (4, 3, 1, 2): 23, (4, 3, 2, 1): 24
}

# Initialize lookup array with sentinel value for invalid permutations
PATTERN_LOOKUP = np.full((5, 5, 5, 5), -1, dtype=np.int8)
for pattern, symbol in PATTERN_TO_SYMBOL.items():
    PATTERN_LOOKUP[pattern[0], pattern[1], pattern[2], pattern[3]] = symbol

# Default class frequencies, aligned with main.py and sample_flights_by_class.py
DEFAULT_CLASS_FREQUENCIES = {0: 0.8997, 1: 0.0706, 2: 0.0223, 3: 0.0097}

def pattern_to_symbol(pattern: np.ndarray) -> int:
    """
    Map a permutation to a symbol using PATTERN_LOOKUP, following Garcia et al. (2024), Table 5.

    Args:
        pattern (np.ndarray): Array of shape (4,) representing a permutation of [1, 2, 3, 4].

    Returns:
        int: Symbol (1-24) corresponding to the permutation.

    Raises:
        ValueError: If the permutation is invalid.
    """
    try:
        if len(pattern) != 4 or not np.all((pattern >= 1) & (pattern <= 4)):
            raise ValueError(f"Invalid permutation {pattern}: must be a permutation of [1, 2, 3, 4]")
        symbol = PATTERN_LOOKUP[pattern[0], pattern[1], pattern[2], pattern[3]]
        if symbol == -1:
            raise ValueError(f"Permutation {pattern} not found in PATTERN_TO_SYMBOL")
        return symbol
    except Exception as e:
        logger.error(f"Error in pattern_to_symbol: {str(e)}")
        raise

def log_memory_usage(step: str = "") -> None:
    """Log current memory usage."""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2
        logger.info(f"Memory usage at {step}: {mem:.2f} MB")
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")

def estimate_batch_size(
    n_samples: int,
    n_timesteps: int,
    n_features: int,
    target_memory_mb: float = 500.0
) -> int:
    """
    Estimate batch size based on memory constraints, following Garcia et al. (2024) for large datasets.

    Args:
        n_samples (int): Number of samples (flights).
        n_timesteps (int): Number of timesteps per sample.
        n_features (int): Number of features per timestep (4).
        target_memory_mb (float): Target memory usage per batch in MB. Default: 500.0.

    Returns:
        int: Estimated batch size.
    """
    try:
        bytes_per_sample_timestep = (n_features * 4) + (1 * n_timesteps)
        bytes_per_sample = bytes_per_sample_timestep * n_timesteps
        target_bytes = target_memory_mb * 1024 * 1024
        batch_size = max(1, int(target_bytes // bytes_per_sample))
        batch_size = min(batch_size, n_samples, 100)
        logger.info(f"Estimated batch size: {batch_size} for {n_samples} samples, {n_timesteps} timesteps")
        return batch_size
    except Exception as e:
        logger.error(f"Error in estimate_batch_size: {str(e)}")
        raise

def plot_pattern_evolution(
    patterns: np.ndarray,
    variables: np.ndarray,
    combo: Tuple[int],
    window_size: int,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ordinal pattern and variable evolution per window, inspired by Garcia et al. (2024), Fig. 8.

    Args:
        patterns (np.ndarray): Array of shape (n_timesteps,) with symbols (1-24).
        variables (np.ndarray): Array of shape (n_timesteps, 4) with normalized variables.
        combo (Tuple[int]): 4-variable combination indices.
        window_size (int): Size of each window for temporal segmentation.
        save_path (Optional[str]): Path to save the plot. If None, display only.
    """
    try:
        n_timesteps = patterns.shape[0]
        n_windows = n_timesteps // window_size
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        timesteps = np.arange(n_timesteps)

        # Plot patterns with window boundaries
        ax1.plot(timesteps, patterns, marker='o', label='Ordinal Patterns')
        for w in range(1, n_windows):
            ax1.axvline(w * window_size, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f"Ordinal Pattern Evolution (Combo {combo})")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Pattern Symbol (1-24)")
        ax1.grid(True)
        ax1.legend()

        # Plot variables
        for i in range(variables.shape[1]):
            ax2.plot(timesteps, variables[:, i], label=f"Variable {combo[i]}")
        for w in range(1, n_windows):
            ax2.axvline(w * window_size, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title("Normalized Variable Evolution")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Normalized Value")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error in plot_pattern_evolution: {str(e)}")
        raise

def extract_ordinal_patterns_for_combination(
    reduced_data: np.ndarray,
    valid_combinations: List[Tuple[int]],
    feature_mappings: List[Dict[int, int]],
    comb_idx: int,
    labels: Optional[np.ndarray] = None,
    timestep: Optional[int] = None,
    dtype: type = np.int8,
    noise_scale: float = 5e-3,
    low_variance_threshold: float = 5e-2,
    min_noise_scale_factor: float = 0.3,
    class_frequencies: Dict[int, float] = None,
    balance_classes: bool = False,
    is_test: bool = False,
    window_size: int = 10,
    visualize: bool = False,
    plot_save_path: Optional[str] = None
) -> np.ndarray:
    """
    Extract ordinal patterns for a single combination, following Garcia et al. (2024), Section II.B.2.
    Supports windowed data for temporal anomaly detection, class-aware noise scaling for low-variability combinations,
    and test data processing.

    Args:
        reduced_data (np.ndarray): Array of shape (n_samples, n_windows, window_size, 4) or (n_samples, n_timesteps, 4).
        valid_combinations (List[Tuple[int]]): List of 4-variable tuples (up to 20, Garcia et al., Section II.B.1).
        feature_mappings (List[Dict[int, int]]): List of dicts mapping global to local indices.
        comb_idx (int): Index of the current combination.
        labels (Optional[np.ndarray]): Array of shape (n_samples,) with class labels (0-3). Default: None.
        timestep (Optional[int]): Timestep to extract (None for all). Default: None.
        dtype (type): Data type for patterns. Default: np.int8.
        noise_scale (float): Base scale for noise addition to low-variance timesteps. Default: 5e-2.
        low_variance_threshold (float): Threshold for low variance detection. Default: 5e-3.
        min_noise_scale_factor (float): Minimum noise scale factor for class-aware scaling. Default: 0.3.
        class_frequencies (Dict[int, float]): Class frequencies for noise scaling. Default: None (uses DEFAULT_CLASS_FREQUENCIES).
        balance_classes (bool): If True, assume balanced classes (no class-aware noise scaling). Default: False.
        is_test (bool): If True, process as test data (no noise scaling). Default: False.
        window_size (int): Size of each window for temporal segmentation. Default: 10.
        visualize (bool): Whether to plot pattern evolution per window. Default: False.
        plot_save_path (Optional[str]): Path to save plots. Default: None.

    Returns:
        np.ndarray: Array of shape (n_samples, n_timesteps) if timestep=None, else (n_samples, 1).

    Raises:
        ValueError: If input shapes, permutations, or combinations are invalid.
    """
    try:
        start_time = time()
        if comb_idx >= len(valid_combinations):
            raise ValueError(f"Invalid comb_idx {comb_idx}, only {len(valid_combinations)} combinations available")
        current_combination = valid_combinations[comb_idx]
        logger.info(f"Processing combination {comb_idx}/{len(valid_combinations)}: {current_combination}")

        # Handle windowed data
        if reduced_data.ndim == 3:
            n_samples, n_timesteps, n_features = reduced_data.shape
        else:
            raise ValueError(f"Expected 3D or 4D data, got {reduced_data.shape}")

        if n_features != 4:
            raise ValueError(f"Expected 4 features, got {n_features}")
        if labels is not None and len(labels) != n_samples:
            raise ValueError(f"Labels length ({len(labels)}) must match n_samples ({n_samples})")

        # Validate data range and clip
        data_min, data_max = np.min(reduced_data), np.max(reduced_data)
        if data_min < 0 or data_max > 1:
            logger.warning(f"Data outside [0, 1]: min={data_min:.3f}, max={data_max:.3f}. Clipping to [0, 1].")
            reduced_data = np.clip(reduced_data, 0, 1)
        logger.info(f"Data range: [{data_min:.3f}, {data_max:.3f}]")

        # Skip noise scaling for test data
        if not is_test:
            # Compute feature variances and noise scales
            feature_vars = np.var(reduced_data, axis=(0, 1))  # (4,)
            feature_vars = np.maximum(feature_vars, 1e-6)
            noise_scales = noise_scale * np.sqrt(feature_vars)

            # Detect low variability per timestep
            variances = np.var(reduced_data, axis=0)  # (n_timesteps, n_features)
            low_variance = variances < low_variance_threshold
            low_variance = np.tile(low_variance[np.newaxis, :, :], (n_samples, 1, 1))
            if np.any(low_variance):
                logger.info(f"Low variability in {np.sum(low_variance)} sample-timestep-feature instances")
                low_var_features = np.any(low_variance, axis=(0, 1))
                logger.info(f"Features with low variance: {np.where(low_var_features)[0]}")
                noise = np.zeros_like(reduced_data)
                for f in range(n_features):
                    noise[:, :, f] = np.random.normal(0, noise_scales[f], size=(n_samples, reduced_data.shape[1]))
                if labels is not None and not balance_classes:
                    class_frequencies = class_frequencies or DEFAULT_CLASS_FREQUENCIES
                    for cls in np.unique(labels):
                        class_mask = labels == cls
                        class_mask_expanded = np.tile(class_mask[:, np.newaxis, np.newaxis], (1, reduced_data.shape[1], n_features))
                        freq = class_frequencies.get(cls, 1.0)
                        noise_scale_factor = max(np.sqrt(freq), min_noise_scale_factor)
                        noise[class_mask_expanded] *= noise_scale_factor
                reduced_data = reduced_data + noise * low_variance
                logger.info(f"After noise: [{np.min(reduced_data):.3f}, {np.max(reduced_data):.3f}]")

        reduced_data = reduced_data.astype(np.float32)
        n_timesteps = reduced_data.shape[1]

        if timestep is not None:
            if timestep >= n_timesteps or timestep < 0:
                raise ValueError(f"Invalid timestep: {timestep}, n_timesteps={n_timesteps}")
            data_slice = reduced_data[:, timestep, :]
            ordinal_patterns = np.argsort(data_slice, axis=1) + 1
            if not np.all(np.sort(ordinal_patterns, axis=1) == np.array([1, 2, 3, 4])):
                raise ValueError(f"Invalid patterns at timestep {timestep}")
            symbols = PATTERN_LOOKUP[ordinal_patterns[:, 0], ordinal_patterns[:, 1], ordinal_patterns[:, 2], ordinal_patterns[:, 3]]
            if np.any(symbols == -1):
                raise ValueError(f"Invalid permutations at timestep {timestep}")
            symbols = symbols.astype(dtype)
            logger.info(f"Combo {comb_idx}, timestep {timestep}: symbols shape: {symbols.shape}")
            unique, counts = np.unique(symbols, return_counts=True)
            logger.info(f"Symbol distribution (timestep {timestep}, {'test' if is_test else 'train'}): {dict(zip(unique, counts))}")
            if visualize and not is_test:
                plot_pattern_evolution(symbols, data_slice, current_combination, window_size, plot_save_path)
            return symbols.reshape(-1, 1)

        # Try vectorized processing first
        try:
            ordinal_patterns = np.argsort(reduced_data, axis=2) + 1
            if not np.all(np.sort(ordinal_patterns, axis=2) == np.array([1, 2, 3, 4])):
                raise ValueError("Invalid patterns in vectorized processing")
            symbols = PATTERN_LOOKUP[
                ordinal_patterns[:, :, 0],
                ordinal_patterns[:, :, 1],
                ordinal_patterns[:, :, 2],
                ordinal_patterns[:, :, 3]
            ]
            if np.any(symbols == -1):
                raise ValueError("Invalid permutations in vectorized processing")
            symbols = symbols.astype(dtype)
        except MemoryError:
            logger.warning("Vectorized processing failed due to memory constraints. Falling back to batch processing.")
            batch_size = estimate_batch_size(n_samples, n_timesteps, n_features)
            symbols = np.zeros((n_samples, n_timesteps), dtype=dtype)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_data = reduced_data[start_idx:end_idx]
                ordinal_patterns = np.argsort(batch_data, axis=2) + 1
                if not np.all(np.sort(ordinal_patterns, axis=2) == np.array([1, 2, 3, 4])):
                    raise ValueError(f"Invalid patterns in batch {start_idx}:{end_idx}")
                batch_symbols = PATTERN_LOOKUP[
                    ordinal_patterns[:, :, 0],
                    ordinal_patterns[:, :, 1],
                    ordinal_patterns[:, :, 2],
                    ordinal_patterns[:, :, 3]
                ]
                if np.any(batch_symbols == -1):
                    raise ValueError(f"Invalid permutations in batch {start_idx}:{end_idx}")
                symbols[start_idx:end_idx] = batch_symbols
                del batch_data, ordinal_patterns, batch_symbols
                gc.collect()
                log_memory_usage(f"after batch {start_idx}:{end_idx} in combo {comb_idx}")

        # Log pattern diversity and temporal variation
        unique, counts = np.unique(symbols, return_counts=True)
        logger.info(f"Combo {comb_idx} symbols shape: {symbols.shape}")
        logger.info(f"Overall symbol distribution ({'test' if is_test else 'train'}): {dict(zip(unique, counts))}")
        if labels is not None:
            for cls in np.unique(labels):
                class_symbols = symbols[labels == cls]
                unique_cls, counts_cls = np.unique(class_symbols, return_counts=True)
                # Compute temporal variation per window
                n_windows = n_timesteps // window_size
                temporal_vars = []
                for w in range(n_windows):
                    window_symbols = class_symbols[:, w * window_size:(w + 1) * window_size]
                    temporal_vars.append(np.std(window_symbols, axis=1).mean())
                mean_temporal_var = np.mean(temporal_vars)
                logger.info(f"Class {cls} symbol distribution: {dict(zip(unique_cls, counts_cls))}")
                logger.info(f"Class {cls} temporal variation (std, mean across windows): {mean_temporal_var:.4f}")
                logger.info(f"Class {cls} per-window temporal variations: {[f'{v:.4f}' for v in temporal_vars]}")
                if len(unique_cls) < 5:
                    logger.warning(f"Class {cls} has skewed distribution: {len(unique_cls)} unique symbols")
                if mean_temporal_var < 1.0:
                    logger.warning(f"Class {cls} has low temporal variation: {mean_temporal_var:.4f}")
        if len(unique) < 5:
            logger.warning(f"Highly skewed distribution: {len(unique)} unique symbols out of 24")

        # Visualize if requested
        if visualize and not is_test:
            plot_pattern_evolution(symbols[0], reduced_data[0], current_combination, window_size, plot_save_path)

        logger.info(f"Combination {comb_idx} processed in {time() - start_time:.2f} seconds")
        return symbols

    except Exception as e:
        logger.error(f"Error in extract_ordinal_patterns_for_combination (comb_idx={comb_idx}): {str(e)}")
        raise

def validate_ordinal_pattern_extraction(
    reduced_datasets: List[np.ndarray],
    valid_combinations: List[Tuple[int]],
    feature_mappings: List[Dict[int, int]],
    labels: Optional[np.ndarray] = None,
    class_frequencies: Dict[int, float] = None,
    balance_classes: bool = False,
    is_test: bool = False,
    window_size: int = 10
) -> None:
    """
    Validate ordinal pattern extraction for a list of reduced datasets, following Garcia et al. (2024).

    Args:
        reduced_datasets (List[np.ndarray]): List of arrays with shape (n_samples, n_windows, window_size, 4).
        valid_combinations (List[Tuple[int]]): List of 4-variable tuples (up to 20).
        feature_mappings (List[Dict[int, int]]): List of dicts mapping global to local indices.
        labels (Optional[np.ndarray]): Array of shape (n_samples,) with class labels (0-3). Default: None.
        class_frequencies (Dict[int, float]): Class frequencies for noise scaling. Default: None.
        balance_classes (bool): If True, assume balanced classes. Default: False.
        is_test (bool): If True, process as test data. Default: False.
        window_size (int): Size of each window. Default: 10.
    """
    try:
        logger.info("Ordinal Pattern Extraction Validation:")
        logger.info(f"Number of reduced datasets: {len(reduced_datasets)}")
        logger.info(f"Number of valid combinations: {len(valid_combinations)}")
        if len(valid_combinations) > 20:
            logger.warning(f"Number of combinations ({len(valid_combinations)}) exceeds paper's 20 (Garcia et al., 2024)")

        for comb_idx, (dataset, combo, mapping) in enumerate(zip(reduced_datasets, valid_combinations, feature_mappings)):
            logger.info(f"\nCombination {comb_idx}:")
            logger.info(f"  Dataset shape: {dataset.shape if dataset is not None else 'None'}")
            logger.info(f"  Feature combination: {combo}")
            logger.info(f"  Feature mappings: {mapping}")

            if dataset is None or combo is None:
                raise ValueError(f"Dataset or combination is None for comb_idx={comb_idx}")

            patterns = extract_ordinal_patterns_for_combination(
                dataset, valid_combinations, feature_mappings, comb_idx,
                labels=labels, class_frequencies=class_frequencies,
                balance_classes=balance_classes, is_test=is_test,
                window_size=window_size
            )
            logger.info("  Extraction successful:")
            logger.info(f"    Number of flights: {patterns.shape[0]}")
            logger.info(f"    Timesteps per flight: {patterns.shape[1]}")
            unique_patterns = set(patterns.flatten())
            logger.info(f"    Unique pattern symbols: {sorted(unique_patterns)} (count: {len(unique_patterns)})")

    except Exception as e:
        logger.error(f"Error in validate_ordinal_pattern_extraction: {str(e)}")
        raise