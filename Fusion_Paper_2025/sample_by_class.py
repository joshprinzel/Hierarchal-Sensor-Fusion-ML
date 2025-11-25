
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def sample_flights_by_class(
        X: np.ndarray,
        y: np.ndarray,
        target_total: int,
        fold_idx: int = 0,
        n_classes: int = 4,
        nominal_proportion: float = 0.8997,
        balance_classes: bool = False,
        replace: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample flights to create training or test sets, optionally balancing classes or preserving the original imbalanced
    distribution (89.97% nominal), following Garcia et al. (2024) with extensions for balanced sampling.

    Args:
        X (np.ndarray): Input data of shape (n_flights, ...).
        y (np.ndarray): Labels of shape (n_flights,).
        target_total (int): Total number of flights to sample.
        fold_idx (int): Fold index for logging. Default: 0.
        n_classes (int): Number of classes (4: Nominal, High Speed, High Path, Late Flaps). Default: 4.
        nominal_proportion (float): Proportion of nominal flights in the original dataset. Default: 0.8997.
        balance_classes (bool): If True, sample equal numbers of flights per class. If False, preserve original distribution.
                               Default: False.
        replace (bool): If True, allow sampling with replacement for minority classes. Default: False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Sampled data, sampled labels, and indices of sampled flights.

    Raises:
        ValueError: If input data is empty, target_total is not provided, or no anomaly flights exist.
    """
    try:
        logger.info(
            f"Fold {fold_idx}: Sampling flights with target_total={target_total}, balance_classes={balance_classes}")

        # Validate inputs
        if target_total is None:
            raise ValueError("Must provide target_total")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        # Count flights per class
        class_counts = np.bincount(y, minlength=n_classes)
        logger.info(f"Fold {fold_idx}: Original class counts: {class_counts}")

        # Indices for each class
        indices_per_class = [np.where(y == i)[0] for i in range(n_classes)]
        total_flights = np.sum(class_counts)
        if total_flights == 0:
            raise ValueError(f"Fold {fold_idx}: No flights to sample")

        # Calculate target counts
        target_counts = []
        if balance_classes:
            # Equal counts per class (e.g., 760 per class for test_total=3040)
            flights_per_class = target_total // n_classes
            target_counts = [flights_per_class] * n_classes
            # Adjust to match target_total
            remainder = target_total - (flights_per_class * n_classes)
            for i in range(remainder):
                target_counts[i] += 1
        else:
            # Preserve original distribution
            target_nominal = int(target_total * nominal_proportion)
            target_anomalies = target_total - target_nominal
            total_anomalies = np.sum(class_counts[1:])
            if total_anomalies == 0:
                raise ValueError(f"Fold {fold_idx}: No anomaly flights to sample")
            anomaly_proportions = class_counts[1:] / total_anomalies
            target_counts = [target_nominal]
            for cls in range(1, n_classes):
                target_counts.append(int(target_anomalies * anomaly_proportions[cls - 1]))
            # Adjust nominal count to match target_total
            current_total = sum(target_counts)
            if current_total != target_total:
                target_counts[0] += target_total - current_total
            # Log adjusted proportion
            adjusted_proportion = target_counts[0] / target_total
            if abs(adjusted_proportion - nominal_proportion) > 0.01:
                logger.warning(
                    f"Fold {fold_idx}: Adjusted nominal proportion {adjusted_proportion:.4f} deviates from {nominal_proportion:.4f}")

        logger.info(f"Fold {fold_idx}: Target counts per class: {target_counts}")

        # Sample flights for each class
        sampled_indices = []
        for cls in range(n_classes):
            cls_indices = indices_per_class[cls]
            n_samples = min(len(cls_indices), target_counts[cls]) if not replace else target_counts[cls]
            if n_samples < target_counts[cls]:
                logger.warning(
                    f"Fold {fold_idx}: Class {cls} has only {len(cls_indices)} flights, requested {target_counts[cls]}")
            if n_samples > 0:
                sampled_cls_indices = np.random.choice(cls_indices, size=n_samples, replace=replace)
                sampled_indices.extend(sampled_cls_indices)

        if not sampled_indices:
            raise ValueError(f"Fold {fold_idx}: No flights sampled")

        # Combine and shuffle indices
        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        # Return sampled data
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        sampled_counts = np.bincount(y_sampled, minlength=n_classes)
        logger.info(f"Fold {fold_idx}: Sampled class counts: {sampled_counts}")

        # Validate sampled counts
        for cls in range(n_classes):
            if sampled_counts[cls] < target_counts[cls] and not replace:
                logger.info(f"Fold {fold_idx}: Class {cls} undersampled: {sampled_counts[cls]} vs {target_counts[cls]}")

        return X_sampled, y_sampled, sampled_indices

    except Exception as e:
        logger.error(f"Fold {fold_idx}: Error in sampling: {str(e)}")
        raise


