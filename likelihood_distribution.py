import time

import numpy as np
import logging

from matplotlib import pyplot as plt
from scipy.stats import entropy
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default class frequencies, aligned with ordinal_patterns.py and sample_flights_by_class.py
DEFAULT_CLASS_FREQUENCIES = {0: 0.8997, 1: 0.0706, 2: 0.0223, 3: 0.0097}


def create_likelihood_distribution(
        patterns: np.ndarray,
        labels: np.ndarray,
        hierarchy_level: int,
        num_patterns: int = 24,
        window_size: int = 10,
        stride: int = 5,
        epsilon: float = 1e-6,
        class_frequencies: Optional[Dict[int, float]] = None,
        balance_classes: bool = False,
        is_test: bool = False,
        alpha: float = 1.0,  # Smoothing parameter
) -> Dict[int, np.ndarray]:
    try:
        start_time = time.time()
        if patterns.shape[0] != len(labels):
            raise ValueError(f"Patterns ({patterns.shape[0]}) and labels ({len(labels)}) mismatch")
        if hierarchy_level not in [0, 1]:
            raise ValueError(f"Invalid hierarchy_level: {hierarchy_level}")
        if not np.all(np.isin(labels, [0, 1, 2, 3])):
            raise ValueError(f"Invalid labels: {np.unique(labels)}")
        if not np.all((patterns >= 1) & (patterns <= num_patterns)):
            raise ValueError(f"Patterns must be in [1, {num_patterns}]")
        n_timesteps = patterns.shape[1]
        if n_timesteps != window_size:
            raise ValueError(f"Expected n_timesteps={window_size}, got {n_timesteps}")

        classes = [0, 1] if hierarchy_level == 0 else [1, 2, 3]
        logger.info(f"Hierarchy level {hierarchy_level}: Classes {classes}")

        if hierarchy_level == 0:
            adjusted_labels = np.where(labels == 0, 0, 1)
            adjusted_patterns = patterns
        else:
            mask = np.isin(labels, [1, 2, 3])
            if not np.any(mask):
                raise ValueError("No flights with labels [1, 2, 3]")
            adjusted_labels = labels[mask]
            adjusted_patterns = patterns[mask]

        n_flights = adjusted_patterns.shape[0]
        logger.info(f"Processing {n_flights} flights, {n_timesteps} timesteps")


        likelihood_distributions = {}
        for cls in classes:
            cls_mask = (adjusted_labels == cls)
            cls_counts = np.count_nonzero(cls_mask)
            if cls_counts == 0:
                logger.warning(f"No samples for class {cls}")
                likelihood_distributions[cls] = np.ones((window_size, num_patterns), dtype=np.float32) / num_patterns
                continue

            cls_patterns = adjusted_patterns[cls_mask]
            cls_distribution = np.zeros((window_size, num_patterns), dtype=np.float32)

            for t in range(window_size):
                pattern_indices = cls_patterns[:, t].astype(np.int32) - 1
                pattern_counts = np.bincount(pattern_indices, minlength=num_patterns)
                # Apply smoothing and prior adjustment
                smoothed_counts = pattern_counts + alpha
                cls_distribution[t] = smoothed_counts / (smoothed_counts.sum() + epsilon)

            likelihood_distributions[cls] = cls_distribution

            unique_patterns = len(np.unique(cls_patterns))
            logger.info(f"Class {cls} unique patterns: {unique_patterns}")
            temporal_var = np.std(cls_patterns, axis=1).mean()
            logger.info(f"Class {cls} temporal variation: {temporal_var:.4f}")

        # Normalize distributions
        for cls in classes:
            likelihood_distributions[cls] /= likelihood_distributions[cls].sum(axis=1, keepdims=True) + 1e-10

        for cls1 in range(len(classes)):
            for cls2 in range(cls1 + 1, len(classes)):
                c1, c2 = classes[cls1], classes[cls2]
                kl_values = np.array([
                    entropy(likelihood_distributions[c1][t], likelihood_distributions[c2][t] + 1e-10, base=2)
                    for t in range(window_size)
                ])
                avg_kl = np.mean(kl_values)
                logger.info(f"Class {c1} vs {c2} KL-divergence: {[f'{v:.4f}' for v in kl_values]}, mean: {avg_kl:.4f}")

        return likelihood_distributions

    except Exception as e:
        logger.error(f"Error in create_likelihood_distribution: {str(e)}")
        raise


def plot_likelihood_comparison(
        dist_nominal: np.ndarray,
        dist_anomaly: np.ndarray,
        timestep: int,
        save_path: str = None
):
    """
    Plots a side-by-side bar chart of nominal and anomaly likelihood distributions
    to visually diagnose pattern ambiguity.
    """
    num_patterns = dist_nominal.shape[0]
    pattern_labels = [str(i) for i in range(1, num_patterns + 1)]

    x = np.arange(num_patterns)
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width / 2, dist_nominal, width, label='P(Pattern | Nominal)')
    rects2 = ax.bar(x + width / 2, dist_anomaly, width, label='P(Pattern | Anomaly)')

    ax.set_ylabel('Probability')
    ax.set_title(f'Comparison of Likelihood Distributions at Timestep {timestep}')
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_labels, rotation=90)
    ax.legend()

    # Calculate and display KL Divergence to quantify the difference
    kl_div = entropy(dist_nominal, dist_anomaly, base=2)
    ax.text(0.05, 0.95, f'KL Divergence (Nominal || Anomaly) = {kl_div:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved likelihood comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


def validate_likelihood_distribution(
        likelihood_distributions: Dict[int, np.ndarray],
        classes: list,
        window_size: int,
        num_patterns: int,
        combination_idx: int,
        window_idx: int,
        hierarchy_level: int
) -> None:
    """
    Validate likelihood distributions for a single window.

    Args:
        likelihood_distributions: Class to distributions (window_size, num_patterns).
        classes: List of class labels.
        window_size: Timesteps per window.
        num_patterns: Number of patterns.
        combination_idx: Combination index.
        window_idx: Window index.
        hierarchy_level: 0 or 1.

    Raises:
        ValueError: If distributions are invalid.
    """
    try:
        logger.info(
            f"Validating likelihood distributions for comb {combination_idx}, window {window_idx}, hierarchy {hierarchy_level}")
        missing_classes = set(classes) - set(likelihood_distributions.keys())
        if missing_classes:
            raise ValueError(f"Comb {combination_idx}, Win {window_idx}: Missing classes: {missing_classes}")

        for cls in classes:
            ld = likelihood_distributions[cls]
            expected_shape = (window_size, num_patterns)
            if ld.shape != expected_shape:
                raise ValueError(
                    f"Comb {combination_idx}, Win {window_idx}, Class {cls}: Invalid shape {ld.shape}, expected {expected_shape}")
            if np.any(np.isnan(ld)) or np.any(ld < 0):
                raise ValueError(f"Comb {combination_idx}, Win {window_idx}, Class {cls}: NaNs or negative values")
            sums = np.sum(ld, axis=1)
            if not np.allclose(sums, 1.0, atol=1e-6):
                raise ValueError(
                    f"Comb {combination_idx}, Win {window_idx}, Class {cls}: Distributions do not sum to 1")

        logger.info(f"Validation successful for comb {combination_idx}, window {window_idx}")

    except Exception as e:
        logger.error(f"Error in validate_likelihood_distribution: {str(e)}")
        raise