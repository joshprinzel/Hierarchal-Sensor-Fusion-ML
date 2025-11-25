
import numpy as np
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import logging
from joblib import Parallel, delayed
import gc
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def create_valid_combinations(
        x_train: np.ndarray,
        num_comb: int = 5,
        max_corr: float = 0.3,
        fold_idx: int = 0
) -> List[Tuple[int, ...]]:
    """
    Generate random 4-variable combinations with maximum correlation < 0.3, following Garcia et al. (2024).

    Args:
        x_train (np.ndarray): Training data of shape (n_flights, n_windows, window_size, n_vars).
        num_comb (int): Number of combinations to generate. Default: 5.
        max_corr (float): Maximum allowed correlation between variables. Default: 0.3.
        fold_idx (int): Fold index for logging. Default: 0.

    Returns:
        List[Tuple[int, ...]]: List of valid 4-variable combinations.

    Raises:
        ValueError: If input data is not 4D.
    """
    try:
        if x_train.ndim != 4:
            raise ValueError(f"Expected 4D data, got {x_train.ndim}D")
        num_features = x_train.shape[3]
        corr_matrix = np.zeros((num_features, num_features), dtype=np.float32)
        for win in range(x_train.shape[1]):
            for t in range(x_train.shape[2]):
                corr_t = np.corrcoef(x_train[:, win, t, :].T)
                corr_matrix += corr_t / (x_train.shape[1] * x_train.shape[2])
        corr_matrix[np.isnan(corr_matrix)] = 0
        logger.info(f"Fold {fold_idx + 1}: Corr range: [{corr_matrix.min():.3f}, {corr_matrix.max():.3f}]")

        all_combinations = list(combinations(range(num_features), 4))
        np.random.seed(42 + fold_idx)
        np.random.shuffle(all_combinations)

        def is_valid_combination(subset: Tuple[int, ...]) -> bool:
            subset_corr = np.abs(corr_matrix[np.ix_(subset, subset)])
            np.fill_diagonal(subset_corr, 0)
            return np.all(subset_corr < max_corr)

        valid_combinations = [subset for subset in all_combinations if is_valid_combination(subset)][:num_comb]
        if len(valid_combinations) < num_comb:
            logger.warning(f"Fold {fold_idx + 1}: Requested {num_comb} combinations, found {len(valid_combinations)}")
        logger.info(f"Fold {fold_idx + 1}: Found {len(valid_combinations)} valid combinations")
        return valid_combinations
    except Exception as e:
        logger.error(f"Fold {fold_idx + 1}: Error in create_valid_combinations: {str(e)}")
        raise



