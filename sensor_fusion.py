import logging
import numpy as np
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from itertools import chain
from scipy.stats import entropy
from sklearn.utils import resample
from typing import Dict, List, Optional, Tuple, Union
import time

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
logger.debug("Logging initialized with DEBUG level")

# Pattern mappings, aligned with ordinal_patterns.py
PATTERN_TO_SYMBOL = {
    (1, 2, 3, 4): 1, (1, 2, 4, 3): 2, (1, 3, 2, 4): 3, (1, 3, 4, 2): 4,
    (1, 4, 2, 3): 5, (1, 4, 3, 2): 6, (2, 1, 3, 4): 7, (2, 1, 4, 3): 8,
    (2, 3, 1, 4): 9, (2, 3, 4, 1): 10, (2, 4, 1, 3): 11, (2, 4, 3, 1): 12,
    (3, 1, 2, 4): 13, (3, 1, 4, 2): 14, (3, 2, 1, 4): 15, (3, 2, 4, 1): 16,
    (3, 4, 1, 2): 17, (3, 4, 2, 1): 18, (4, 1, 2, 3): 19, (4, 1, 3, 2): 20,
    (4, 2, 1, 3): 21, (4, 2, 3, 1): 22, (4, 3, 1, 2): 23, (4, 3, 2, 1): 24
}
SYMBOL_TO_PATTERN = {idx: np.array(pattern, dtype=np.int8) for pattern, idx in PATTERN_TO_SYMBOL.items()}
# Precompute distances for all 24 patterns using bubble sort
PRECOMPUTED_DISTANCES = np.zeros((24, 24), dtype=np.int32)
for i in range(24):
    for j in range(24):
        pi_i = SYMBOL_TO_PATTERN[i + 1]  # 1-based to 0-based index
        pi_j = SYMBOL_TO_PATTERN[j + 1]
        # Temporary bubble sort implementation for precomputation
        if not np.all(np.sort(pi_i) == np.sort(pi_j)):
            PRECOMPUTED_DISTANCES[i, j] = 6  # Maximum distance for m=4
        else:
            m = len(pi_i)
            order = np.zeros(m + 1, dtype=np.int32)
            for idx, val in enumerate(pi_i):
                order[val] = idx
            pi_j_copy = pi_j.copy()
            swaps = 0
            for a in range(m):
                for b in range(m - a - 1):
                    if order[pi_j_copy[b]] > order[pi_j_copy[b + 1]]:
                        pi_j_copy[b], pi_j_copy[b + 1] = pi_j_copy[b + 1], pi_j_copy[b]
                        swaps += 1
            PRECOMPUTED_DISTANCES[i, j] = swaps
logger.info("Precomputed distances for all pattern pairs")

CLASS_NAMES = {0: 'Nominal', 1: 'High Speed', 2: 'High Path', 3: 'Late Flaps'}
DEFAULT_CLASS_FREQUENCIES = {0: 0.8997, 1: 0.0706, 2: 0.0223, 3: 0.0097}

def estimate_batch_size(n_samples: int, n_timesteps: int, n_features: int, target_memory_mb: float = 500.0) -> int:
    """Estimate batch size based on memory constraints."""
    bytes_per_sample = (n_features * 4 + n_timesteps) * n_timesteps
    target_bytes = target_memory_mb * 1024 * 1024
    batch_size = max(1, min(int(target_bytes // bytes_per_sample), n_samples, 100))
    logger.info(f"Estimated batch size: {batch_size} for {n_samples} samples, {n_timesteps} timesteps")
    return batch_size

def create_boe_batch(
        patterns: np.ndarray,
        likelihood_distributions: List[List[Dict[int, np.ndarray]]],
        class_labels: List[int],
        hierarchy_level: int,
        fixed_priors: List[float],
        pattern_weights: np.ndarray,
        window_size: int = 10,
        stride: int = 5,
        total_timesteps: int = 70,
        batch_size: int = 10,
        all_patterns: Optional[np.ndarray] = None,
        all_labels: Optional[np.ndarray] = None,
        is_test: bool = False,
        balance_classes: bool = False,
        class_frequencies: Optional[Dict[int, float]] = None
) -> List[List[np.ndarray]]:
    try:
        start_time = time.time()
        n_combinations, n_flights, actual_timesteps = patterns.shape
        if actual_timesteps != total_timesteps:
            raise ValueError(f"Expected {total_timesteps} timesteps, got {actual_timesteps}")
        n_windows = (total_timesteps - window_size) // stride + 1
        logger.info(f"n_combinations: {n_combinations}, n_windows: {n_windows}")
        if len(likelihood_distributions) != n_combinations or any(
                len(ld) != n_windows for ld in likelihood_distributions):
            raise ValueError(f"Expected {n_combinations} x {n_windows} likelihood distributions")
        if len(fixed_priors) != len(class_labels):
            raise ValueError(f"Priors length ({len(fixed_priors)}) mismatches class_labels ({len(class_labels)})")
        # NOTE: all_patterns and all_labels checks can be simplified since the reference set
        # for old uncertainty is removed, but we keep the checks for robust input validation.
        if all_patterns is None or all_labels is None:
            # Retained for input validation, though not used for uncertainty
            logger.warning("all_patterns and all_labels are not used for uncertainty but were required for validation.")

        # --- NO REFERENCE SET PRE-COMPUTATION HERE (DELETED) ---

        all_combination_boes = [[[] for _ in range(n_flights)] for _ in range(n_combinations)]
        # Assuming estimate_batch_size is defined elsewhere
        batch_size = estimate_batch_size(n_flights, total_timesteps, 4)

        for batch_start in range(0, n_flights, batch_size):
            batch_end = min(batch_start + batch_size, n_flights)
            num_in_batch = batch_end - batch_start

            for comb_idx in range(n_combinations):
                for win_idx in range(n_windows):
                    start = win_idx * stride
                    end = start + window_size
                    window_patterns_batch = patterns[comb_idx, batch_start:batch_end, start:end]

                    for i in range(num_in_batch):
                        flight_idx = batch_start + i
                        pattern_indices_flight = window_patterns_batch[i, :] - 1

                        # --- POSTERIOR CALCULATION ---
                        current_likelihoods = likelihood_distributions[comb_idx][win_idx]
                        window_likelihoods = np.zeros(len(class_labels))
                        for cls_idx, cls in enumerate(class_labels):
                            weights = pattern_weights[pattern_indices_flight]
                            likelihoods = current_likelihoods[cls][np.arange(window_size), pattern_indices_flight]
                            window_likelihoods[cls_idx] = np.mean(likelihoods * weights)

                        window_posteriors = window_likelihoods * np.array(fixed_priors)
                        post_sum = window_posteriors.sum()
                        flight_posterior = window_posteriors / post_sum if post_sum > 1e-10 else np.full(
                            len(class_labels), 1 / len(class_labels))

                        # --- STABLE ENTROPY-BASED UNCERTAINTY CALCULATION (Replacement) ---
                        # 1. Calculate Shannon Entropy H(X)
                        safe_posterior = np.maximum(flight_posterior, 1e-10)
                        entropy_val = -np.sum(safe_posterior * np.log(safe_posterior))

                        # 2. Normalize Entropy to get the uncertainty mass m('Ω')
                        num_classes = len(class_labels)
                        max_entropy = np.log(num_classes) if num_classes > 1 else 1.0

                        # Uncertainty m('Ω') is high when posterior is confused (high entropy)
                        uncertainty = np.clip(entropy_val / max_entropy, 0.0, 1.0)

                        # 3. Assign the Discounted Mass
                        boe = {str(cls): flight_posterior[idx] * (1.0 - uncertainty) for idx, cls in
                               enumerate(class_labels)}
                        boe['Ω'] = uncertainty

                        # --- CRITICAL STEP: APPEND THE BOE ---
                        all_combination_boes[comb_idx][flight_idx].append(boe)

        logger.info(f"BOEs created in {time.time() - start_time:.2f}s")
        return all_combination_boes

    except Exception as e:
        logger.error(f"Error in create_boe_batch: {e}", exc_info=True)
        raise

def dempster_combination_rule(boe1: Dict[str, float], boe2: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Apply Dempster's Combination Rule to fuse two BOEs, with corrected conflict calculation.
    """
    # Identify all class labels, excluding the uncertainty placeholder 'Ω'
    class_labels = sorted([k for k in set(boe1.keys()) | set(boe2.keys()) if k != 'Ω'])

    # --- CORRECT CONFLICT CALCULATION ---
    # Conflict is the sum of products of belief masses for all DISJOINT sets.
    # The only disjoint sets here are pairs of different classes (e.g., '0' and '1').
    K = 0.0
    for i in range(len(class_labels)):
        for j in range(i + 1, len(class_labels)):
            cls1 = class_labels[i]
            cls2 = class_labels[j]
            K += boe1.get(cls1, 0.0) * boe2.get(cls2, 0.0)
            K += boe1.get(cls2, 0.0) * boe2.get(cls1, 0.0)

    # The normalization factor is (1 - K)
    norm_factor = 1.0 - K

    # Handle high conflict or numerical instability
    if norm_factor < 1e-6:
        # If conflict is total, averaging is a common fallback
        combined_masses = {cls: (boe1.get(cls, 0) + boe2.get(cls, 0)) / 2 for cls in class_labels}
        combined_masses['Ω'] = (boe1.get('Ω', 0) + boe2.get('Ω', 0)) / 2
        total = sum(combined_masses.values())
        return {k: v / total for k, v in combined_masses.items()}, K

    # --- CORRECT FUSION FORMULA ---
    combined_masses = {}
    for cls in class_labels:
        # Fuse belief for each specific class
        m1 = boe1.get(cls, 0.0)
        m2 = boe2.get(cls, 0.0)
        m1_omega = boe1.get('Ω', 0.0)
        m2_omega = boe2.get('Ω', 0.0)
        combined_masses[cls] = (m1 * m2 + m1 * m2_omega + m2 * m1_omega) / norm_factor

    # Fuse the uncertainty mass
    combined_masses['Ω'] = (boe1.get('Ω', 0.0) * boe2.get('Ω', 0.0)) / norm_factor
    if K > 0.3:
        logger.debug(f"Fusion Step: K={K:.4f}, Norm Factor 1/(1-K)={1.0/norm_factor:.4f}, Fused BOE: {combined_masses}")


    # Final normalization for stability
    total_mass = sum(combined_masses.values()) + 1e-10
    return {k: v / total_mass for k, v in combined_masses.items()}, K


# In sensor_fusion.py

def convert_boe_to_pignistic(boe: Dict[str, float], class_labels: list) -> Dict[str, float]:
    """
    Correctly converts a Body of Evidence (BOE) to a pignistic probability distribution.
    This function handles singleton, composite, and uncertainty focal elements.
    """
    if not boe:
        # Return a uniform distribution if the BOE is empty
        return {str(cls): 1.0 / len(class_labels) for cls in class_labels}

    pignistic_probs = {str(cls): 0.0 for cls in class_labels}
    num_classes = len(class_labels)

    for focal_element, mass in boe.items():
        if mass > 0:
            # Handle the uncertainty mass 'Ω'
            if focal_element == 'Ω':
                # Distribute the uncertainty mass equally among all singleton classes
                for cls in class_labels:
                    pignistic_probs[str(cls)] += mass / num_classes
            else:
                # Handle singleton or composite focal elements (e.g., '0' or '0,1')
                elements = [int(e) for e in focal_element.split(',')]
                num_elements_in_set = len(elements)
                if num_elements_in_set > 0:
                    # Distribute the mass of the set equally among its members
                    for el in elements:
                        if str(el) in pignistic_probs:
                            pignistic_probs[str(el)] += mass / num_elements_in_set

    # Normalize the final probabilities to ensure they sum to 1
    total_prob = sum(pignistic_probs.values())
    if total_prob > 1e-9:
        pignistic_probs = {k: v / total_prob for k, v in pignistic_probs.items()}
    else:
        # Fallback to uniform if total probability is near zero
        return {str(cls): 1.0 / num_classes for cls in class_labels}

    return pignistic_probs


def apply_temporal_discounting(boes: List[Dict[str, float]],
                               overlap_ratio: float = 0.5) -> List[Dict[str, float]]:
    """
    Apply uniform reliability discounting to account for temporal correlation
    between overlapping sliding windows.

    Mathematical basis: α = 1/(1 + ρ) where ρ is the overlap ratio.
    For stride=5, window_size=10: ρ = (10-5)/10 = 0.5
    This means α = 1/1.5 ≈ 0.667

    Each window contributes ~67% independent information, with ~33%
    transferred to uncertainty due to redundancy from temporal overlap.

    Args:
        boes: List of BOEs from overlapping temporal windows
        overlap_ratio: Fraction of overlap between consecutive windows

    Returns:
        List of discounted BOEs

    References:
        Shafer, G. (1976). A Mathematical Theory of Evidence.
        Princeton University Press. (Section on reliability discounting)
    """
    if len(boes) <= 1:
        return boes

    # Uniform discount factor for all windows
    alpha = 1.0 / (1.0 + overlap_ratio)
    logger.debug(f"Applying temporal discounting with α={alpha:.3f} (overlap_ratio={overlap_ratio})")

    discounted_boes = []
    for idx, boe in enumerate(boes):
        discounted = {}
        total_belief = sum(v for k, v in boe.items() if k != 'Ω')

        for k, v in boe.items():
            if k == 'Ω':
                # Uncertainty increases by absorbing discounted belief mass
                discounted['Ω'] = v + (1 - alpha) * total_belief
            else:
                # Scale down specific belief masses
                discounted[k] = alpha * v

        # Normalize to ensure valid mass function (sum = 1)
        total = sum(discounted.values())
        if total > 1e-10:
            discounted = {k: v / total for k, v in discounted.items()}
        else:
            logger.warning(f"Discounting resulted in zero total mass for BOE {idx}, using original")
            discounted = boe

        discounted_boes.append(discounted)

    logger.debug(f"Discounted {len(boes)} BOEs: avg Ω before={np.mean([b.get('Ω', 0) for b in boes]):.3f}, "
                 f"after={np.mean([b.get('Ω', 0) for b in discounted_boes]):.3f}")

    return discounted_boes


def fuse_all_boes(boes: List[Dict[str, float]],
                  class_labels: List[int],
                  apply_temporal_discount: bool = False,
                  overlap_ratio: float = 0.5) -> Tuple[Dict[str, float], float]:
    """
    Fuse multiple BOEs using Dempster's rule with conflict-based weighting.

    Args:
        boes: List of BOEs to fuse
        class_labels: List of class labels
        apply_temporal_discount: If True, apply temporal discounting before fusion
        overlap_ratio: Overlap ratio for temporal discounting (default: 0.5)

    Returns:
        Tuple of (fused BOE, average conflict)
    """
    if not boes:
        return {str(cls): 1.0 / len(class_labels) for cls in class_labels}, 0
    if len(boes) == 1:
        return boes[0], 0



    # === APPLY TEMPORAL DISCOUNTING IF REQUESTED ===
    if apply_temporal_discount:
        boes_to_fuse = apply_temporal_discounting(boes, overlap_ratio)
    else:
        boes_to_fuse = boes
    # === END TEMPORAL DISCOUNTING ===

    fused_boe = boes_to_fuse[0]
    conflicts = []
    for pattern_boe in boes_to_fuse[1:]:
        fused_boe, conflict = dempster_combination_rule(fused_boe, pattern_boe)
        conflicts.append(conflict)

    avg_conflict = np.mean(conflicts) if conflicts else 0.0
    if len(class_labels) > 1:
        total = sum(v for k, v in fused_boe.items() if k != 'Ω') + 1e-10
        uncertainty = fused_boe.get('Ω', 0.0)
        fused_boe = {k: v / total * (1 - uncertainty) if k != 'Ω' else v for k, v in fused_boe.items()}
    return fused_boe, avg_conflict
def extract_decision_from_boe(boe: Dict[str, float], class_labels: List[int]) -> Tuple[int, float, Dict[str, float]]:
    """Extract final decision from a BOE using pignistic transformation."""
    pignistic = convert_boe_to_pignistic(boe, class_labels)
    pred = int(max(pignistic, key=pignistic.get))
    confidence = pignistic[str(pred)]
    return pred, confidence, pignistic

def hierarchical_sensor_fusion(
    patterns: np.ndarray,
    likelihood_distributions: List[List[Dict[int, np.ndarray]]],
    class_labels_map: Dict[int, str],
    pattern_weights: np.ndarray,
    priors: List[float],
    hierarchy_level: int,
    rf_probs: np.ndarray,
    fusion_weights: Union[Dict[str, float], List[Dict[str, float]]],
    window_size: int = 10,
    stride: int = 5,
    total_timesteps: int = 70,
    batch_size: int = 10,
    fold_idx: int = 0,
    flight_indices: Optional[np.ndarray] = None,
    binary_threshold: float = 0.15,  # Kept as parameter but not used
    all_patterns: Optional[np.ndarray] = None,
    all_labels: Optional[np.ndarray] = None,
    is_test: bool = False,
    balance_classes: bool = False,
    class_frequencies: Optional[Dict[int, float]] = None
) -> Dict:
    stage_name = "Binary" if hierarchy_level == 0 else "Minority"
    logger.info(f"--- Starting Fusion: {stage_name} Stage (Fold {fold_idx}) ---")

    n_combinations, n_flights, actual_timesteps = patterns.shape
    if actual_timesteps != total_timesteps:
        raise ValueError(f"{stage_name}: Expected {total_timesteps} timesteps, got {actual_timesteps}")

    n_windows = (total_timesteps - window_size) // stride + 1
    logger.info(f"n_combinations: {n_combinations}, n_windows: {n_windows}")
    logger.info(f"Likelihood distributions: {len(likelihood_distributions)} combinations, {[len(ld) for ld in likelihood_distributions]} windows")

    if len(likelihood_distributions) != n_combinations or any(len(ld) != n_windows for ld in likelihood_distributions):
        logger.error(f"Expected {n_combinations} x {n_windows} likelihood distributions, got {len(likelihood_distributions)} x {[len(ld) for ld in likelihood_distributions]}")
        raise ValueError(f"Expected {n_combinations} x {n_windows} likelihood distributions")

    class_labels = list(class_labels_map.keys())
    if rf_probs.shape != (n_flights, len(class_labels)):
        raise ValueError(f"{stage_name}: RF probs shape mismatch, expected ({n_flights}, {len(class_labels)}), got {rf_probs.shape}")

    rf_probs_adj = np.maximum(rf_probs, 1e-3)
    rf_probs_adj /= rf_probs_adj.sum(axis=1, keepdims=True)

    if isinstance(fusion_weights, list):
        if not fusion_weights:
            raise ValueError(f"{stage_name}: Empty fusion_weights list")
        fusion_weights = fusion_weights[0]
        logger.debug(f"{stage_name}: Using fusion_weights[0]: {fusion_weights}")
    else:
        logger.debug(f"{stage_name}: Using fusion_weights: {fusion_weights}")

    pattern_boes_all_combs = create_boe_batch(
        patterns, likelihood_distributions, class_labels, hierarchy_level,
        fixed_priors=priors, pattern_weights=pattern_weights,
        window_size=window_size, stride=stride, total_timesteps=total_timesteps, batch_size=batch_size,
        all_patterns=all_patterns, all_labels=all_labels, is_test=is_test,
        balance_classes=balance_classes, class_frequencies=class_frequencies
    )
    logger.info(f"{stage_name}: Generated pattern-based BOEs.")

    rf_boes = []
    min_uncertainty = 0.235 if hierarchy_level == 0 else 0.25
    for i in range(n_flights):
        # Define base BOE using local uncertainty (max uncertainty is min_uncertainty)
        rf_confidence = rf_probs_adj[i].max()
        local_uncertainty = max(min_uncertainty, 1.0 - rf_confidence ** 0.85)

        # Base BOE creation (m_original sums to 1.0)
        base_boe = {str(cls): rf_probs_adj[i, idx] * (1.0 - local_uncertainty) for idx, cls in enumerate(class_labels)}
        base_boe['Ω'] = local_uncertainty

        # Apply fusion weight w_rf as a reliability factor (discounting)
        w_rf = fusion_weights['rf']
        boe = {k: v * w_rf for k, v in base_boe.items()}
        boe['Ω'] = base_boe['Ω'] * w_rf + (1.0 - w_rf)  # Transfer (1 - w_rf) mass to uncertainty

        # Note: No need for re-normalization, this BOE already sums to 1.0
        rf_boes.append(boe)

    final_preds = np.zeros(n_flights, dtype=np.int8)
    final_confidences = np.zeros(n_flights, dtype=np.float32)
    final_pignistics = [{} for _ in range(n_flights)]
    all_conflicts = np.zeros(n_flights, dtype=np.float32)

    for i in range(n_flights):
        boes_to_fuse_flight = []
        if fusion_weights['rf'] > 1e-6:
            boes_to_fuse_flight.append(rf_boes[i])
        if fusion_weights['dst'] > 1e-6:
            for comb_idx in range(n_combinations):
                window_boes_flight_comb = pattern_boes_all_combs[comb_idx][i]
                if len(window_boes_flight_comb) > 0:
                    # === APPLY TEMPORAL DISCOUNTING FOR OVERLAPPING WINDOWS ===
                    fused_comb_boe, avg_win_conflict = fuse_all_boes(
                        window_boes_flight_comb,
                        class_labels,
                        apply_temporal_discount=False,  # Enable for temporal windows
                        overlap_ratio=0.5  # (window_size - stride) / window_size
                    )
                    # === END CHANGE ===
                    # Use original fusion weights directly without adjustment
                    w_dst = fusion_weights['dst']
                    weighted_boe = {k: v * w_dst for k, v in fused_comb_boe.items()}
                    weighted_boe['Ω'] = fused_comb_boe.get('Ω', 0.0) * w_dst + (1.0 - w_dst)

                    # weighted_boe now sums to 1.0
                    boes_to_fuse_flight.append(weighted_boe)

        if not boes_to_fuse_flight:
            logger.error(f"Flight {i}: No valid BOEs, assigning default")
            final_preds[i] = class_labels[0]
            final_confidences[i] = 0.0
            final_pignistics[i] = {str(cls): 1.0 / len(class_labels) for cls in class_labels}
            all_conflicts[i] = 1.0
            continue

        # --- The NEW, CORRECTED block ---
        fused_final_boe, final_conflict = fuse_all_boes(boes_to_fuse_flight, class_labels)
        all_conflicts[i] = final_conflict

        # Step 1: Correctly convert the BOE to a pignistic probability distribution
        pignistic = convert_boe_to_pignistic(fused_final_boe, class_labels)

        # Step 2: Extract the decision and confidence from the corrected pignistic distribution
        if not pignistic:  # Handle empty pignistic dict
            pred = class_labels[0]
            confidence = 0.0
        else:
            # The prediction is the class with the highest pignistic probability
            pred_str = max(pignistic, key=pignistic.get)
            pred = int(pred_str)
            confidence = pignistic[pred_str]

        if pred not in class_labels:
            logger.warning(f"Flight {i}: Invalid prediction {pred}, setting to {class_labels[0]}")
            pred = class_labels[0]

        final_preds[i] = pred
        final_confidences[i] = confidence
        # Also store the uncertainty from the original BOE for our metrics
        pignistic['Ω'] = fused_final_boe.get('Ω', 0.0)
        final_pignistics[i] = pignistic

        log_idx = flight_indices[i] if flight_indices is not None and i < len(flight_indices) else i
        if i < 5:
            logger.info(f"Fold {fold_idx} - Flight {log_idx}: {stage_name} Pred={pred}, Conf={final_confidences[i]:.4f}, Conflict={final_conflict:.4f}")

    logger.info(f"{stage_name}: Avg conflict: {all_conflicts.mean():.4f}, Pred counts: {np.bincount(final_preds, minlength=max(class_labels) + 1)}")
    return {
        'predictions': final_preds,
        'confidences': final_confidences,
        'pignistics': final_pignistics,
        'avg_conflict': all_conflicts.mean()
    }

def extract_features(
    data: np.ndarray,
    valid_combinations: List,
    feature_mappings: Dict,
    labels: Optional[np.ndarray] = None,
    window_size: int = 10,
    stride: int = 5,
    total_timesteps: int = 70
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and patterns from 5D data."""
    logger.info(f"Extracting features: data shape {data.shape}, n_combinations {len(valid_combinations)}")
    n_combinations, n_flights, n_windows, window_size_data, n_vars = data.shape
    expected_n_windows = (total_timesteps - window_size) // stride + 1  # 13
    if n_windows != expected_n_windows:
        raise ValueError(f"Expected n_windows={expected_n_windows}, got {n_windows}")
    if window_size_data != window_size:
        raise ValueError(f"Expected window_size={window_size}, got {window_size_data}")

    data = data.astype(np.float32)
    # Initialize flight_data and count for averaging overlaps
    flight_data = np.zeros((n_flights, total_timesteps, n_vars), dtype=np.float32)
    count = np.zeros((n_flights, total_timesteps, n_vars), dtype=np.float32)
    patterns = np.zeros((n_combinations, n_flights, total_timesteps), dtype=np.int8)

    for comb_idx in range(n_combinations):
        for win_idx in range(n_windows):
            start = win_idx * stride
            end = min(start + window_size, total_timesteps)  # Ensure not to exceed total_timesteps
            window_data = data[comb_idx, :, win_idx, :end - start, :]  # Trim if needed
            flight_data[:, start:end, :] += window_data
            count[:, start:end, :] += 1

        # Average overlapping regions
        count[count == 0] = 1  # Avoid division by zero
        flight_data_avg = flight_data / count

        # Compute patterns
        pattern = np.zeros((n_flights, total_timesteps), dtype=np.int8)
        for flight_idx in range(n_flights):
            for t in range(total_timesteps):
                values = flight_data_avg[flight_idx, t, :]
                ranks = np.argsort(np.argsort(values)) + 1
                pattern[flight_idx, t] = PATTERN_TO_SYMBOL.get(tuple(ranks), 1)
        patterns[comb_idx] = pattern

    # Compute raw features
    raw = data.transpose(1, 0, 2, 3, 4).reshape(n_flights, -1)
    features = raw
    if np.isnan(features).sum() > 0:
        logger.info(f"NaNs detected: {np.isnan(features).sum()}")
        features = SimpleImputer(strategy='median').fit_transform(features)

    if labels is not None:
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_patterns = patterns[:, cls_mask, :].ravel()
            unique_patterns = len(np.unique(cls_patterns))
            logger.info(f"Class {cls}: Unique patterns={unique_patterns}")

    return features, patterns

def compute_per_class_metrics(confusion_matrix: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class precision, recall, and F1 scores."""
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for cls in range(num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
    return precision, recall, f1

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], fold_idx: int, title: str, save_path: str = "confusion_matrix_fold_{}_{}.png"):
    """Plot and save confusion matrix."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{title} - Fold {fold_idx}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_filename = save_path.format(fold_idx, title.lower().replace(" ", "_"))
        plt.savefig(save_filename)
        logger.info(f"Saved confusion matrix to {save_filename}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")

def plot_per_class_f1(f1_fusion: np.ndarray, f1_rf: np.ndarray, class_names: List[str], fold_idx: int, save_path: str = "per_class_f1_fold_{}.png"):
    """Plot and save per-class F1 scores."""
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_names))
        plt.bar(x - 0.2, f1_fusion, 0.4, label='Hierarchical Fusion', color='skyblue')
        plt.bar(x + 0.2, f1_rf, 0.4, label='XGBoost-Only', color='lightcoral')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title(f'Per-Class F1 Scores - Fold {fold_idx}')
        plt.xticks(x, class_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path.format(fold_idx))
        logger.info(f"Saved per-class F1 plot to {save_path.format(fold_idx)}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting per-class F1: {e}")

def plot_roc_curve(binary_true: np.ndarray, binary_probs: np.ndarray, fold_idx: int, save_path: str = "roc_curve_fold_{}.png"):
    """Plot and save ROC curve for binary stage."""
    try:
        fpr, tpr, _ = roc_curve(binary_true, binary_probs)
        auc_score = roc_auc_score(binary_true, binary_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Binary Stage - Fold {fold_idx}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(save_path.format(fold_idx))
        logger.info(f"Saved ROC curve to {save_path.format(fold_idx)}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")

def plot_uncertainty_trend(uncertainties: List[np.ndarray], n_combinations: int, n_flights: int, fold_idx: int, save_path: str = "uncertainty_trend_fold_{}.png"):
    """Plot uncertainty trends across windows and combinations."""
    try:
        plt.figure(figsize=(12, 6))
        for comb_idx in range(n_combinations):
            mean_uncertainties = np.mean(uncertainties[comb_idx], axis=1)
            plt.plot(mean_uncertainties, label=f'Combination {comb_idx}')
        plt.xlabel('Window Index')
        plt.ylabel('Mean Uncertainty')
        plt.title(f'Uncertainty Trends Across Windows - Fold {fold_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path.format(fold_idx))
        logger.info(f"Saved uncertainty trend plot to {save_path.format(fold_idx)}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting uncertainty trend: {e}")
# In sensor_fusion.py

def dst_confidence_metrics(pignistic: dict):
    """Return (max_singleton_prob, omega). Works with empty or partial dicts."""
    if not pignistic:
        return 0.0, 1.0 # If no pignistic, treat as max uncertainty
    omega = float(pignistic.get('Ω', 0.0))
    singletons = [float(v) for k, v in pignistic.items() if str(k).isdigit()]
    max_singleton = max(singletons) if singletons else 0.0
    return max_singleton, omega

def perform_sensor_fusion(
    training_data_windowed: np.ndarray,
    test_data_windowed: np.ndarray,
    train_patterns: np.ndarray,
    test_patterns: np.ndarray,
    training_labels: np.ndarray,
    test_labels: np.ndarray,
    likelihood_distributions_all: List[List[Dict[int, np.ndarray]]],
    valid_combinations: List,
    feature_mappings: Dict,
    fold_idx: int = 0,
    fixed_fusion_weights: Optional[Dict[str, float]] = None,
    class_weights_rf: Optional[Dict[int, float]] = None,
    dynamic_weights: bool = False,
    all_patterns: Optional[np.ndarray] = None,  # This will now be train_patterns
    all_labels: Optional[np.ndarray] = None,  # This will now be training_labels
    window_size: int = 10,
    stride: int = 5,
    total_timesteps: int = 70,
    batch_size: int = 50,
    balance_classes: bool = False,
    class_frequencies: Optional[Dict[int, float]] = None,
    is_test: bool = False
) -> Dict:
    """
    Perform hierarchical sensor fusion with XGBoost and DST, following Garcia et al. (2024), Section II.C.
    Extends with class-aware weighting for imbalanced data (89.97% nominal).
    Returns:
        Dict: Predictions and performance metrics.
    """
    train_start_time = time.time()
    binary_labels_map = {0: 'Nominal', 1: 'Anomaly'}
    minority_labels_map = {1: 'High Speed', 2: 'High Path', 3: 'Late Flaps'}
    all_class_labels = list(CLASS_NAMES.keys())
    n_windows = (total_timesteps - window_size) // stride + 1


    n_train_flights = training_data_windowed.shape[0]
    n_flights_test = test_data_windowed.shape[0]
    train_features = training_data_windowed.reshape(n_train_flights, -1)
    test_features = test_data_windowed.reshape(n_flights_test, -1)

    n_combinations = test_patterns.shape[0]

    class_frequencies = class_frequencies or DEFAULT_CLASS_FREQUENCIES
    binary_priors = [class_frequencies[0], sum(class_frequencies[i] for i in [1, 2, 3])]
    binary_priors = [p / sum(binary_priors) for p in binary_priors]
    minority_priors = [class_frequencies[i] for i in [1, 2, 3]]
    minority_priors = [p / sum(minority_priors) for p in minority_priors]
    logger.info(f"Fold {fold_idx} - Priors - Binary: {binary_priors}, Minority: {minority_priors}")
    class_frequencies = class_frequencies or DEFAULT_CLASS_FREQUENCIES

    # Use balanced priors for DST to prevent prior-dominated predictions
    binary_priors = [0.51, 0.49]  # Balanced: nominal vs anomaly
    minority_priors = [1 / 3, 1 / 3, 1 / 3]  # Balanced: high_speed vs high_path vs late_flaps

    logger.info(f"Fold {fold_idx} - Priors - Binary: {binary_priors}, Minority: {minority_priors}")
    logger.info(
        f"Fold {fold_idx} - (Dataset frequencies: Binary=[{class_frequencies[0]:.4f}, {sum(class_frequencies[i] for i in [1, 2, 3]):.4f}])")

    expected_len_per_stage = n_combinations * n_windows
    if len(likelihood_distributions_all) != 2 or any(len(ld) != expected_len_per_stage for ld in likelihood_distributions_all):
        logger.info(f"{len(likelihood_distributions_all)} != {expected_len_per_stage}")
        raise ValueError(f"Expected 2 x {expected_len_per_stage} likelihood distributions")
    binary_ld = [
        likelihood_distributions_all[0][i * n_windows:(i + 1) * n_windows]
        for i in range(n_combinations)
    ]
    minority_ld = [
        likelihood_distributions_all[1][i * n_windows:(i + 1) * n_windows]
        for i in range(n_combinations)
    ]
    logger.info(
        f"Fold {fold_idx} - Reshaped binary_ld: {len(binary_ld)} x {len(binary_ld[0])}, minority_ld: {len(minority_ld)} x {len(minority_ld[0])}")


    # Modified pattern weights to boost minority class influence
    pattern_counts_per_class = np.zeros((len(all_class_labels), 24), dtype=np.float64)
    class_weights = {0: 1.0, 1: 5.0, 2: 8.0, 3: 10.0}  # Increased weights for minority classes
    class_weights_rf = class_weights_rf or class_weights
    sample_weights = np.array([class_weights_rf[cls] for cls in training_labels])
    sample_weights /= sample_weights.sum() / len(sample_weights)
    class_proportions = np.bincount(training_labels, minlength=4) / len(training_labels)
    class_balancing_weights = [class_weights[cls] / (class_proportions[cls] + 1e-10) for cls in range(4)]
    for cls in range(4):
        cls_mask = training_labels == cls
        if cls_mask.sum() > 0:
            cls_patterns = train_patterns[:, cls_mask, :].ravel()
            cls_counts = np.bincount(cls_patterns, minlength=25)[1:]
            pattern_counts_per_class[cls] = cls_counts   # Boost minority pattern influence
    pseudocount = 1e-6
    pattern_counts_per_class = (pattern_counts_per_class.T / (pattern_counts_per_class.sum(axis=1) + pseudocount)).T
    pattern_counts = pattern_counts_per_class.mean(axis=0)
    pattern_rarity = 1.0 / (pattern_counts + pseudocount)
    pattern_rarity_weights = pattern_rarity / pattern_rarity.sum()
    logger.info(f"Pattern rarity weights: {pattern_rarity_weights[:5]}")

    rf_model = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.02, objective='multi:softprob',
        num_class=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42
    )

    class_fusion_weights = [fixed_fusion_weights or {'rf':0.8, 'dst':0.2} for _ in range(4)]
    rf_model.fit(train_features, training_labels, sample_weight=sample_weights)
    #--TRAINING TIME END --
    train_end_time = time.time()
    training_time = train_end_time - train_start_time

    #--- INFERENCE TIME START ---
    inference_start_time = time.time()
    test_rf_probs = rf_model.predict_proba(test_features)
    test_rf_preds = np.argmax(test_rf_probs, axis=1)

    # Compute DST-only predictions with fixed threshold
    binary_rf_probs = np.zeros((n_flights_test, 2), dtype=np.float32)
    binary_rf_probs[:, 0] = test_rf_probs[:, 0]
    binary_rf_probs[:, 1] = test_rf_probs[:, 1:].sum(axis=1)
    binary_true = (test_labels > 0).astype(np.int8)  # Define binary_true here

    binary_rf_probs_adj = np.maximum(binary_rf_probs, 1e-3)
    binary_rf_probs_adj /= binary_rf_probs_adj.sum(axis=1, keepdims=True)
    binary_results_dst_only = hierarchical_sensor_fusion(
        test_patterns, binary_ld, binary_labels_map, pattern_rarity_weights, binary_priors, 0,
        binary_rf_probs_adj, {'rf': 0.0, 'dst': 1.0}, window_size, stride, total_timesteps, batch_size,
        fold_idx, np.arange(n_flights_test), 0.5, all_patterns, all_labels, is_test, balance_classes, class_frequencies
    )
    binary_pred_fused_dst_only = binary_results_dst_only['predictions']
    minority_flights_dst_only = np.where(binary_pred_fused_dst_only == 1)[0]
    if len(minority_flights_dst_only) == 0:
        logger.warning("No anomalies detected in DST-only, skipping minority stage")
        final_pred_fused_dst_only = np.zeros(n_flights_test, dtype=np.int32)
    else:
        minority_patterns_dst_only = test_patterns[:, minority_flights_dst_only, :]
        minority_rf_probs_dst_only = test_rf_probs[minority_flights_dst_only, 1:]
        minority_rf_probs_dst_only /= minority_rf_probs_dst_only.sum(axis=1, keepdims=True) + 1e-10
        minority_results_dst_only = hierarchical_sensor_fusion(
            minority_patterns_dst_only, minority_ld, minority_labels_map, pattern_rarity_weights, minority_priors, 1,
            minority_rf_probs_dst_only, [{'rf': 0.0, 'dst': 1.0} for _ in range(3)], window_size, stride, total_timesteps, batch_size,
            fold_idx, minority_flights_dst_only, 0.5, all_patterns, all_labels, is_test, balance_classes, class_frequencies
        )
        final_pred_fused_dst_only = np.zeros(n_flights_test, dtype=np.int32)
        final_pred_fused_dst_only[minority_flights_dst_only] = minority_results_dst_only['predictions']
        final_pred_fused_dst_only[binary_pred_fused_dst_only == 0] = 0

    dst_only_cm = confusion_matrix(test_labels, final_pred_fused_dst_only, labels=[0, 1, 2, 3])
    class_names = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
    plot_confusion_matrix(dst_only_cm, class_names, fold_idx, "DST Only")

    # Hierarchical fusion with fixed threshold
    binary_rf_probs_adj = np.maximum(binary_rf_probs, 1e-3)
    binary_rf_probs_adj /= binary_rf_probs_adj.sum(axis=1, keepdims=True)
    binary_results = hierarchical_sensor_fusion(
        test_patterns, binary_ld, binary_labels_map, pattern_rarity_weights, binary_priors, 0,
        binary_rf_probs_adj, class_fusion_weights[0], window_size, stride, total_timesteps, batch_size,
        fold_idx, np.arange(n_flights_test), 0.5, all_patterns, all_labels, is_test, balance_classes, class_frequencies
    )
    binary_pred_fused = binary_results['predictions']
    binary_confidences = binary_results['confidences']
    binary_pignistics = binary_results['pignistics']

    minority_flights = np.where(binary_pred_fused == 1)[0]
    if len(minority_flights) == 0:
        logger.warning("No anomalies detected, skipping minority stage")
        final_pred_fused = np.zeros(n_flights_test, dtype=np.int32)
        final_confidences = binary_confidences
        final_pignistics = binary_pignistics
    else:
        minority_patterns = test_patterns[:, minority_flights, :]
        minority_rf_probs = test_rf_probs[minority_flights, 1:]
        minority_rf_probs /= minority_rf_probs.sum(axis=1, keepdims=True) + 1e-10
        minority_results = hierarchical_sensor_fusion(
            minority_patterns, minority_ld, minority_labels_map, pattern_rarity_weights, minority_priors, 1,
            minority_rf_probs, class_fusion_weights[1], window_size, stride, total_timesteps, batch_size,
            fold_idx, minority_flights, 0.5, all_patterns, all_labels, is_test, balance_classes, class_frequencies
        )
        final_pred_fused = np.zeros(n_flights_test, dtype=np.int32)
        final_confidences = np.zeros(n_flights_test, dtype=np.float32)
        final_pignistics = [{} for _ in range(n_flights_test)]
        nominal_indices = np.where(binary_pred_fused == 0)[0]
        final_pred_fused[nominal_indices] = 0
        final_confidences[nominal_indices] = np.array([binary_confidences[i] for i in nominal_indices])
        for i, idx in enumerate(nominal_indices):
            final_pignistics[idx] = binary_pignistics[idx]
        minority_indices = np.where(binary_pred_fused == 1)[0]
        final_pred_fused[minority_indices] = minority_results['predictions']
        final_confidences[minority_indices] = minority_results['confidences']
        for i, idx in enumerate(minority_indices):
            final_pignistics[idx] = minority_results['pignistics'][i]

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    final_cm = confusion_matrix(test_labels, final_pred_fused, labels=[0, 1, 2, 3])
    rf_cm = confusion_matrix(test_labels, test_rf_preds, labels=[0, 1, 2, 3])
    final_precision, final_recall, final_f1 = compute_per_class_metrics(final_cm, 4)
    rf_precision, rf_recall, rf_f1 = compute_per_class_metrics(rf_cm, 4)

    """
    use_rf_classes = [cls for cls in range(4) if rf_f1[cls] - final_f1[cls] > 0.1]
    if use_rf_classes:
        rf_confidences = test_rf_probs.max(axis=1)
        rf_pignistics = [{str(cls): prob for cls, prob in enumerate(probs)} for probs in test_rf_probs]
        for cls in use_rf_classes:
            combined_mask = (final_pred_fused == cls) | (test_rf_preds == cls)
            final_pred_fused[combined_mask] = test_rf_preds[combined_mask]
            final_confidences[combined_mask] = rf_confidences[combined_mask]
            for idx in np.where(combined_mask)[0]:
                final_pignistics[idx] = rf_pignistics[idx]
    """

    final_cm = confusion_matrix(test_labels, final_pred_fused, labels=[0, 1, 2, 3])
    binary_probs_anomaly = np.array([1.0 - prob[0] if pred == 0 else prob[1:].sum() for pred, prob in zip(final_pred_fused, test_rf_probs)])
    binary_auc = roc_auc_score(binary_true, binary_probs_anomaly)

    anomaly_indices = np.where(test_labels > 0)[0]
    anomaly_auc = None
    if len(anomaly_indices) > 1 and len(np.unique(test_labels[anomaly_indices])) > 1:
        anomaly_true = test_labels[anomaly_indices] - 1
        anomaly_pred_probs = test_rf_probs[anomaly_indices, 1:]
        anomaly_pred_probs /= anomaly_pred_probs.sum(axis=1, keepdims=True) + 1e-10
        anomaly_auc = roc_auc_score(anomaly_true, anomaly_pred_probs, multi_class='ovr', average='macro', labels=[0, 1, 2])

    full_auc = roc_auc_score(test_labels, test_rf_probs, multi_class='ovr', average='macro', labels=[0, 1, 2, 3]) if len(np.unique(test_labels)) > 1 else None

    fused_f1 = f1_score(test_labels, final_pred_fused, average='micro', labels=[0, 1, 2, 3], zero_division=0)
    fused_mcc = matthews_corrcoef(test_labels, final_pred_fused)
    rf_f1 = f1_score(test_labels, test_rf_preds, average='micro', labels=[0, 1, 2, 3], zero_division=0)
    rf_mcc = matthews_corrcoef(test_rf_preds, test_labels)
    balanced_accuracy = final_recall.mean()
    logger.info(f"Fold {fold_idx} - Balanced Accuracy: {balanced_accuracy:.4f}")
    majority_pred = np.zeros_like(test_labels)
    majority_f1 = f1_score(test_labels, majority_pred, average='micro', zero_division=0)
    majority_mcc = matthews_corrcoef(test_labels, majority_pred)
    logger.info(f"Fold {fold_idx} - Majority Vote - F1: {majority_f1:.4f}, MCC: {majority_mcc:.4f}")

    class_names = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
    plot_confusion_matrix(rf_cm, class_names, fold_idx, "XGBoost Only")
    plot_confusion_matrix(final_cm, class_names, fold_idx, "Hierarchical Fusion")
    plot_confusion_matrix(dst_only_cm, class_names, fold_idx, "DST Only")
    plot_per_class_f1(final_f1, rf_f1, class_names, fold_idx)
    plot_roc_curve(binary_true, binary_probs_anomaly, fold_idx)
    return {
        'predictions': {
            'final': final_pred_fused,
            'binary': binary_pred_fused,
            'rf_only': test_rf_preds,
            'dst_only': final_pred_fused_dst_only,
            'binary_probs': binary_pignistics,
            'final_pignistics': final_pignistics,
            'xgb_probs': test_rf_probs
        },
        'metrics': {
            'f1_micro': fused_f1,
            'mcc': fused_mcc,
            'balanced_accuracy': balanced_accuracy,
            'majority_f1': majority_f1,
            'majority_mcc': majority_mcc,
            'confusion_matrix': final_cm,
            'rf_confusion_matrix': rf_cm,
            'dst_confusion_matrix': dst_only_cm,
            'binary_auc': binary_auc,
            'anomaly_auc': anomaly_auc,
            'full_auc': full_auc,
            'rf_f1_micro': rf_f1,
            'rf_mcc': rf_mcc,
            'hierarchical_f1': fused_f1,
            'hierarchical_mcc': fused_mcc,
            'per_class_precision': final_precision,
            'per_class_recall': final_recall,
            'per_class_f1': final_f1
        },
        'timing':{
            'training_time': training_time,
            'inference_time': inference_time
        }
    }




#Seperate idea:
"""
            # ==============================================================================
            # ### --- INTELLIGENT FUSION: Confidence-Based Override ("Safety Net") --- ###
            # ==============================================================================
            logger.info("Applying dynamic, confidence-based override ('Safety Net')...")
            override_mask = np.zeros(n_flights_test, dtype=bool)
            rf_confidences = test_rf_probs.max(axis=1)

            for i in range(n_flights_test):
                max_s, omega = dst_confidence_metrics(final_pignistics[i])
                if max_s < 0.35 or omega > 0.45:
                    override_mask[i] = True

            override_count = np.sum(override_mask)
            logger.info(
                f"Overriding {override_count}/{n_flights_test} flights ({override_count / n_flights_test:.2%}) due to low DST confidence.")

            # Apply the override for flights that met the condition
            final_pred_fused[override_mask] = test_rf_preds[override_mask]
            final_confidences[override_mask] = rf_confidences[override_mask]

            # Preserve explainability: store the fallback without erasing the original
            for idx in np.where(override_mask)[0]:
                rf_fallback = {str(cls): float(p) for cls, p in enumerate(test_rf_probs[idx])}
                if final_pignistics[idx]:
                    final_pignistics[idx]['rf_fallback'] = rf_fallback
                else:
                    final_pignistics[idx] = {'rf_fallback': rf_fallback, 'Ω': 1.0}
            # ==============================================================================


           """
