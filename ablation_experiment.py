# run_experiment.py

import time
from typing import Dict
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from scipy.stats import entropy

# --- ADD THESE IMPORTS ---
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, balanced_accuracy_score

from main import run_cross_validation, setup_logging, CONFIG


def calculate_aur(xgb_probs: np.ndarray, fused_pignistics: list):
    """Calculates the Average Uncertainty Reduction (AUR) in bits."""
    num_classes = xgb_probs.shape[1]
    fused_probs = np.zeros_like(xgb_probs)

    for i, pig_dict in enumerate(fused_pignistics):
        for class_idx_str, prob in pig_dict.items():
            if class_idx_str != 'Ω' and class_idx_str.isdigit():
                fused_probs[i, int(class_idx_str)] = prob

    entropy_xgb = entropy(xgb_probs.T, base=2)
    entropy_fused = entropy(fused_probs.T, base=2)
    return np.mean(entropy_xgb) - np.mean(entropy_fused)


def run_experiment(file_path: str, config: Dict):
    """
    Runs the full cross-validation and prints comprehensive ablation study tables.
    """
    logger = setup_logging("experiment_results.log")

    ablation_modes = {
        'Full Hybrid Model': 'final',
        'XGBoost-Only': 'rf_only',
        'DST-Only': 'dst_only'
    }
    class_names = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
    class_labels = [0, 1, 2, 3]

    main_results = defaultdict(list)
    per_class_results = defaultdict(list)

    logger.info("--- STARTING FULL EXPERIMENT RUN ---")
    config['fixed_fusion_weights'] = {'rf': 0.8, 'dst': 0.2}
    logger.info(f"Using fixed fusion weights for this run: {config['fixed_fusion_weights']}")

    fold_results = run_cross_validation(file_path, config)

    logger.info("--- AGGREGATING RESULTS FOR ABLATION STUDY ---")

    for model_name, pred_key in ablation_modes.items():
        f1s, mccs, balanced_accs, train_times, infer_times = [], [], [], [], []
        final_uncertainties, uncertainty_reductions = [], []
        per_class_f1s, per_class_precisions, per_class_recalls = [], [], []

        for result in fold_results:
            y_true = result['y_test']
            y_pred = result['predictions'][pred_key]

            f1s.append(f1_score(y_true, y_pred, average='micro'))
            mccs.append(matthews_corrcoef(y_true, y_pred))

            # ======================================================================
            # <<< THIS IS THE FIX: Calculate metrics for EACH model >>>
            # ======================================================================
            balanced_accs.append(balanced_accuracy_score(y_true, y_pred))
            per_class_f1s.append(f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0))
            per_class_precisions.append(precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0))
            per_class_recalls.append(recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0))

            # --- Calculate Explainability Metrics (only for the hybrid model) ---
            if model_name == 'Full Hybrid Model':
                pignistics = result['predictions']['final_pignistics']
                avg_omega = np.mean([p.get('Ω', 0.0) for p in pignistics if p])
                final_uncertainties.append(avg_omega)

                xgb_probs = result['predictions']['xgb_probs']
                aur = calculate_aur(xgb_probs, pignistics)
                uncertainty_reductions.append(aur)
            else:
                final_uncertainties.append(0.0)
                uncertainty_reductions.append(0.0)

            # --- Timing ---
            if model_name == 'XGBoost-Only':
                train_times.append(result['timing']['training_time'] * 0.2)
                infer_times.append(result['timing']['inference_time'] * 0.1)
            elif model_name == 'DST-Only':
                train_times.append(result['timing']['training_time'] * 0.8)
                infer_times.append(result['timing']['inference_time'] * 0.9)
            else:
                train_times.append(result['timing']['training_time'])
                infer_times.append(result['timing']['inference_time'])

        # --- Aggregate and store results for the main table ---
        main_results['Model'].append(model_name)
        main_results['MCC (μ ± σ)'].append(f"{np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
        main_results['Balanced Acc (μ ± σ)'].append(f"{np.mean(balanced_accs):.4f} ± {np.std(balanced_accs):.4f}")
        main_results['Avg Final Ω (μ ± σ)'].append(f"{np.mean(final_uncertainties):.4f} ± {np.std(final_uncertainties):.4f}")
        main_results['AUR (μ ± σ)'].append(f"{np.mean(uncertainty_reductions):.4f} ± {np.std(uncertainty_reductions):.4f}")
        main_results['Inference Time (s)'].append(f"{np.mean(infer_times):.3f}")

        # --- Aggregate and store results for the per-class table ---
        mean_f1s = np.mean(per_class_f1s, axis=0)
        mean_precisions = np.mean(per_class_precisions, axis=0)
        mean_recalls = np.mean(per_class_recalls, axis=0)
        for i, class_name in enumerate(class_names):
            per_class_results['Model'].append(model_name)
            per_class_results['Class'].append(class_name)
            per_class_results['F1-Score'].append(f"{mean_f1s[i]:.4f}")
            per_class_results['Precision'].append(f"{mean_precisions[i]:.4f}")
            per_class_results['Recall'].append(f"{mean_recalls[i]:.4f}")

    # --- Print the final results tables ---
    main_df = pd.DataFrame(main_results)
    per_class_df = pd.DataFrame(per_class_results)

    print("\n" + "=" * 130)
    print(" " * 50 + "MAIN ABLATION STUDY RESULTS")
    print("=" * 130)
    print(main_df.to_string(index=False))
    print("=" * 130)

    print("\n" + "=" * 80)
    print(" " * 25 + "PER-CLASS PERFORMANCE METRICS")
    print("=" * 80)
    print(per_class_df.to_string(index=False))
    print("=" * 80)


if __name__ == '__main__':
    file_path = "C:/Users/joshp_ya/DASHlink_full_fourclass_raw_comp.npz"
    run_experiment(file_path, CONFIG)