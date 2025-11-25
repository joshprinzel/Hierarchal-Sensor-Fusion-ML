
import os

import pandas as pd
import psutil
import traceback
import gc
import logging
import numpy as np
from typing import List, Dict, Tuple, Any


from dataloader import load_file, trim_timesteps, apply_time_windows
from five_fold_cv import create_valid_combinations
from likelihood_distribution import create_likelihood_distribution, validate_likelihood_distribution
from sensor_fusion import perform_sensor_fusion, plot_confusion_matrix, plot_roc_curve
from sample_by_class import sample_flights_by_class
from ordinal_pattern import extract_ordinal_patterns_for_combination

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import entropy, chi2_contingency
import scipy.stats as stats

from SWC.SWC_aviation import run_swc_validation


# Configuration dictionary to centralize parameters
CONFIG = {
    'kfold': 5,
    'max_combinations': 5,  #can adjust to 20 as per Garcia et al.
    'timesteps': 70,
    'window_size': 10,
    'stride': 5,
    'n_classes': 4,
    'nominal_proportion': 0.8997,  # %89.97 nominal for imbalanced
    'train_anomaly_total': 760,  # 760 per anomaly class for balanced training
    'test_total': 2000,  # 760 per class for balanced testing
    'num_patterns': 24,
    'epsilon': 0.005,
}


def setup_logging(log_file: str = "demo_processing.log") -> logging.Logger:
    """
    Configure logging with console and file handlers using UTF-8 encoding.

    Args:
        log_file (str): Path to the log file. Default: "demo_processing.log".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('FlightAnomaly')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_memory_usage(step: str = "") -> None:
    """
    Log current memory usage in MB.

    Args:
        step (str): Description of the current processing step.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 ** 2
    logger.info(f"Memory usage at {step}: {mem:.2f} MB")


def run_cross_validation(file_path: str, config: Dict) -> List[Dict]:
    """
    Main driver for running the k-fold cross-validation with a leak-free workflow.
    """
    try:
        # --- STEP 1: Load RAW data (NO preprocessing yet) ---
        raw_data, labels = load_file(file_path)
        labels = labels.astype(np.int8)
        logger.info(f"Loaded raw data shape: {raw_data.shape}, labels: {labels.shape}")

        # --- STEP 2: Set up K-Fold indices BEFORE any processing ---
        skf = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=42)
        fold_indices = list(skf.split(np.zeros(len(labels)), labels))

        all_fold_results = []

        # --- STEP 3: Main Cross-Validation Loop ---
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            log_memory_usage(f"start of fold {fold_idx}")

            # --- 3a. Split RAW data for this fold ---
            X_train_raw, X_test_raw = raw_data[train_idx], raw_data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # --- 3b. Fit Normalizer ONLY on training data and transform both sets ---
            mean_train, std_train = np.mean(X_train_raw, axis=(0, 1)), np.std(X_train_raw, axis=(0, 1))

            def transform_data(data, mean, std):
                min_bound = mean - 2 * std
                max_bound = mean + 2 * std
                normalized = (data - min_bound) / (max_bound - min_bound + 1e-8)
                return np.clip(normalized, 0, 1)

            X_train_norm = transform_data(X_train_raw, mean_train, std_train)
            X_test_norm = transform_data(X_test_raw, mean_train, std_train)

            # --- 3c. Apply Trimming and Windowing to each set ---
            X_train_trimmed = trim_timesteps(X_train_norm, config['timesteps'])
            X_test_trimmed = trim_timesteps(X_test_norm, config['timesteps'])

            X_train_windowed = apply_time_windows(X_train_trimmed, config['window_size'], config['stride'])
            X_test_windowed = apply_time_windows(X_test_trimmed, config['window_size'], config['stride'])

            # --- 3d. Create low-correlation feature combinations ---
            valid_combinations = create_valid_combinations(X_train_windowed, num_comb=config['max_combinations'],
                                                           fold_idx=fold_idx)
            feature_mappings = [{global_idx: local_idx for local_idx, global_idx in enumerate(subset)} for subset in
                                valid_combinations]

            # --- 3e. Sub-sample the data to create balanced sets for training ---
            # Note: Sampling is now done on indices to save memory
            nominal_flights = int(
                (config['train_anomaly_total'] * config['nominal_proportion']) / (1 - config['nominal_proportion']))
            train_target_total = nominal_flights + config['train_anomaly_total']

            _, y_train_sampled, train_indices = sample_flights_by_class(
                X_train_windowed, y_train, target_total=train_target_total, fold_idx=fold_idx,
                n_classes=config['n_classes'], nominal_proportion=config['nominal_proportion'], balance_classes=False,
                replace=True
            )
            _, y_test_sampled, test_indices = sample_flights_by_class(
                X_test_windowed, y_test, target_total=config['test_total'], fold_idx=fold_idx,
                n_classes=config['n_classes'], nominal_proportion=config['nominal_proportion']
            )

            # Apply sampled indices to all data representations
            X_train_trimmed_sampled = X_train_trimmed[train_indices]
            X_test_trimmed_sampled = X_test_trimmed[test_indices]
            X_train_windowed_sampled = X_train_windowed[train_indices]

            X_test_windowed_sampled = X_test_windowed[test_indices]
            X_test_flat = X_test_windowed_sampled.reshape(len(X_test_windowed_sampled), -1)

            # --- 3f. Process a single fold with the preprocessed data ---
            result = process_single_fold(
                fold_idx=fold_idx,
                X_train_trimmed=X_train_trimmed_sampled,
                y_train=y_train_sampled,
                X_test_trimmed=X_test_trimmed_sampled,
                y_test=y_test_sampled,
                X_train_windowed=X_train_windowed_sampled,
                X_test_windowed=X_test_windowed_sampled,
                valid_combinations=valid_combinations,
                feature_mappings=feature_mappings,
                X_test_flat= X_test_flat,
                config=config
            )
            all_fold_results.append(result)

            log_memory_usage(f"end of fold {fold_idx}")
            gc.collect()

        return all_fold_results

    except Exception as e:
        logger.error(f"Error in run_cross_validation: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def process_single_fold(
        fold_idx: int,
        X_train_trimmed: np.ndarray,
        y_train: np.ndarray,
        X_test_trimmed: np.ndarray,
        y_test: np.ndarray,
        X_train_windowed: np.ndarray,
        X_test_windowed: np.ndarray,
        valid_combinations: List,
        feature_mappings: Dict,
        X_test_flat: np.ndarray,
        config: Dict
) -> Dict[str, Any]:
    """
    Processes all logic for a single fold: pattern extraction, model training, and fusion.
    """
    try:
        # --- STEP 1: Extract Ordinal Patterns ---
        # This is now done once on the trimmed (but not windowed) data.
        # This is much cleaner than the previous reconstruct -> extract logic.
        train_patterns = []
        test_patterns = []
        for comb_idx, combo in enumerate(valid_combinations):
            # Select the 4 variables for this combination
            X_train_combo = X_train_trimmed[:, :, list(combo)]
            X_test_combo = X_test_trimmed[:, :, list(combo)]

            train_pat = extract_ordinal_patterns_for_combination(X_train_combo, valid_combinations, feature_mappings,
                                                                 comb_idx, labels=y_train)
            test_pat = extract_ordinal_patterns_for_combination(X_test_combo, valid_combinations, feature_mappings,
                                                                comb_idx, labels=y_test, is_test=True)
            train_patterns.append(train_pat)
            test_patterns.append(test_pat)

        train_patterns = np.stack(train_patterns, axis=0)
        test_patterns = np.stack(test_patterns, axis=0)
        logger.info(
            f"Fold {fold_idx}: Train patterns shape: {train_patterns.shape}, Test patterns shape: {test_patterns.shape}")

        # --- STEP 2: Create Likelihood Distributions (The "Model Training") ---
        # This uses ONLY the training patterns
        binary_distributions, minority_distributions = [], []
        n_windows_ld = (config['timesteps'] - config['window_size']) // config['stride'] + 1
        ld_config = {
            'num_patterns': config['num_patterns'],
            'window_size': config['window_size'],
            'stride': config['stride'],
            'epsilon': config['epsilon']
        }
        # Binary Stage
        for comb_idx in range(len(valid_combinations)):
            for win_idx in range(n_windows_ld):
                start = win_idx * config['stride']
                end = start + config['window_size']
                patterns_window = train_patterns[comb_idx, :, start:end]

                binary_ld = create_likelihood_distribution(patterns_window, y_train, hierarchy_level=0, **ld_config)
                binary_distributions.append(binary_ld)

        # Minority Stage
        anom_mask = y_train > 0
        anom_train_patterns = train_patterns[:, anom_mask, :]
        anom_y_train = y_train[anom_mask]
        for comb_idx in range(len(valid_combinations)):
            for win_idx in range(n_windows_ld):
                start = win_idx * config['stride']
                end = start + config['window_size']
                patterns_window = anom_train_patterns[comb_idx, :, start:end]

                minority_ld = create_likelihood_distribution(patterns_window, anom_y_train, hierarchy_level=1, **ld_config)
                minority_distributions.append(minority_ld)

        likelihood_distributions_all = [binary_distributions, minority_distributions]

        # --- DIAGNOSTIC STEP: Visualize Likelihoods ---
        # We will run this only for the first fold to get a single, clear plot.
        if fold_idx == 0:
            # Get the distributions for the very first window of the first combination.
            # `binary_distributions` is a list where each element is a dictionary {class: distribution}.
            binary_ld_for_plot = binary_distributions[0]

            # Extract the probability vectors for the first timestep (t=0) within that window.
            # The shape is (window_size, num_patterns), so we take the first row.
            nominal_dist = binary_ld_for_plot[0][0, :]  # Class 0, timestep 0
            anomaly_dist = binary_ld_for_plot[1][0, :]  # Class 1, timestep 0

            # Call the new plotting function we added.
            from likelihood_distribution import plot_likelihood_comparison
            plot_likelihood_comparison(
                dist_nominal=nominal_dist,
                dist_anomaly=anomaly_dist,
                timestep=0,
                save_path="likelihood_comparison_fold0_comb0_win0.png"
            )
            logger.info("Generated diagnostic likelihood comparison plot.")
        # --- END DIAGNOSTIC ---
        # --- STEP 3: Perform Sensor Fusion ---

        # We pass `train_patterns` and `y_train` as the reference set for uncertainty calculation.
        results = perform_sensor_fusion(
            training_data_windowed=X_train_windowed,
            test_data_windowed=X_test_windowed,
            train_patterns=train_patterns,
            test_patterns=test_patterns,
            training_labels=y_train,
            test_labels=y_test,
            likelihood_distributions_all=likelihood_distributions_all,
            valid_combinations=valid_combinations,
            feature_mappings=feature_mappings,

            all_patterns=train_patterns,
            all_labels=y_train,
            fold_idx=fold_idx,
            # Pass relevant config values
            window_size=config['window_size'],
            stride=config['stride'],
            total_timesteps=config['timesteps'],
        )


        results['y_test'] = y_test
        results['features'] = X_test_flat
        results['fold'] = fold_idx
        return results

    except Exception as e:
        logger.error(f"Error in process_single_fold (fold {fold_idx}): {str(e)}")
        logger.error(traceback.format_exc())
        raise


def summarize_results(all_fold_data: List[Dict]) -> None:
    """
    Summarize results across all folds, following Garcia et al.'s evaluation metrics.

    Args:
        all_fold_data (List[Dict]): List of fold results.
    """
    logger.info("=" * 50)
    logger.info("HIERARCHICAL ANOMALY DETECTION SUMMARY")
    logger.info("=" * 50)
    class_names = {0: 'Nominal', 1: 'High Speed', 2: 'High Path', 3: 'Late Flaps'}
    total_f1_micro, total_mcc = 0, 0
    all_y_true, all_final_preds, all_binary_preds, all_binary_probs = [], [], [], []

    for fold_data in all_fold_data:
        # --- Access nested data structures ---
        preds_data = fold_data.get('predictions', {})
        metrics_data = fold_data.get('metrics', {})

        fold = fold_data["fold"]
        y_true = fold_data["y_test"]

        # Corrected key access for predictions
        final_preds = preds_data.get("final")
        binary_preds = preds_data.get("binary")
        binary_probs = preds_data.get("binary_probs")

        # Corrected key access for metrics
        f1_micro = metrics_data.get('f1_micro', 0.0)
        mcc = metrics_data.get('mcc', 0.0)
        precision_per_class = metrics_data.get('per_class_precision', {})
        recall_per_class = metrics_data.get('per_class_recall', {})
        f1_per_class = metrics_data.get('per_class_f1', {})

        # Check for essential data before processing
        if final_preds is None or binary_preds is None or binary_probs is None:
            logger.warning(f"Fold {fold}: Missing essential prediction data. Skipping fold accumulation.")
            continue

        logger.info(f"Fold {fold} - binary_probs structure: {type(binary_probs[0])}, sample: {binary_probs[0]}")
        all_y_true.extend(y_true)
        all_final_preds.extend(final_preds)
        all_binary_preds.extend(binary_preds)
        all_binary_probs.extend(binary_probs)

        # Updated logging to use extracted metric variables
        logger.info(f"Fold {fold} - F1 Micro: {f1_micro:.4f}, MCC: {mcc:.4f}")
        for i, cls in class_names.items():
            # --- FIX: Use array indexing (i.e., []) instead of .get() ---
            if isinstance(precision_per_class, np.ndarray) and i < len(precision_per_class):
                precision = precision_per_class[i]
                recall = recall_per_class[i]
                f1 = f1_per_class[i]
            else:
                # Fallback in case it's not a NumPy array or index is out of bounds
                precision = recall = f1 = 0.0

            logger.info(f"  - {cls}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        total_f1_micro += f1_micro
        total_mcc += mcc

    n_folds = len(all_fold_data)
    # The division may cause a ZeroDivisionError if n_folds is 0 (i.e., all folds were skipped)
    if n_folds > 0:
        logger.info(f"Average F1 Micro: {total_f1_micro / n_folds:.4f}, Average MCC: {total_mcc / n_folds:.4f}")
    else:
        logger.info("No fold results were successfully processed to calculate averages.")

    all_y_true = np.array(all_y_true, dtype=np.int8)
    all_final_preds = np.array(all_final_preds, dtype=np.int8)
    all_binary_preds = np.array(all_binary_preds, dtype=np.int8)

    # Handle the case where all_binary_probs is empty
    if all_binary_probs:
        all_binary_probs_anomaly = np.array([prob['1'] for prob in all_binary_probs], dtype=np.float32)
    else:
        all_binary_probs_anomaly = np.array([])

    logger.info(f"Aggregate F1 Micro: {f1_score(all_y_true, all_final_preds, average='micro', zero_division=0):.4f}")
    logger.info(f"Aggregate MCC: {matthews_corrcoef(all_y_true, all_final_preds):.4f}")

    binary_true = (all_y_true > 0).astype(np.int8)

    # Ensure there's data before calculating CM/Scores
    if len(binary_true) > 0 and len(all_binary_preds) > 0:
        binary_cm = confusion_matrix(binary_true, all_binary_preds, labels=[0, 1])
        logger.info(f"Aggregate Binary Stage Confusion Matrix:\n{binary_cm}")

        precision_agg = precision_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']),
                                        zero_division=0)
        recall_agg = recall_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']),
                                  zero_division=0)
        f1_agg = f1_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']), zero_division=0)
        logger.info("Aggregate Per-Class Metrics:")
        for i, cls in class_names.items():
            logger.info(
                f"  - {cls}: Precision: {precision_agg[i]:.4f}, Recall: {recall_agg[i]:.4f}, F1: {f1_agg[i]:.4f}")

        # Compute and plot aggregate confusion matrix
        aggregate_cm = confusion_matrix(all_y_true, all_final_preds, labels=range(CONFIG['n_classes']))
        class_names_list = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
        plot_confusion_matrix(aggregate_cm, class_names_list, fold_idx='aggregate',
                              title="Hierarchical Fusion (All Folds)",
                              save_path="confusion_matrix_aggregate_hierarchical_fusion.png")
        logger.info(f"Aggregate Confusion Matrix:\n{aggregate_cm}")
        logger.info(f"Saved aggregate confusion matrix to confusion_matrix_aggregate_hierarchical_fusion.png")

        # Compute and plot aggregate ROC curve
        if len(all_binary_probs_anomaly) > 0:
            fpr, tpr, _ = roc_curve(binary_true, all_binary_probs_anomaly)
            aggregate_auc = roc_auc_score(binary_true, all_binary_probs_anomaly)
            plot_roc_curve(binary_true, all_binary_probs_anomaly, fold_idx='aggregate',
                           save_path="roc_curve_aggregate_hierarchical_fusion.png")
            logger.info(f"Aggregate Binary AUC: {aggregate_auc:.4f}")
            logger.info(f"Saved aggregate ROC curve to roc_curve_aggregate_hierarchical_fusion.png")
        else:
            logger.info("Cannot compute ROC curve: No binary probabilities available.")

    else:
        logger.info("Cannot compute aggregate metrics: No data available in aggregated arrays.")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Assuming you have from your results:
# y_true: true labels (N,)
# y_pred: predicted labels (N,)
# m_omega: uncertainty mass (N,)
# xgb_probs: XGBoost class probabilities (N, 4)

# PLOT 1: Uncertainty vs Accuracy
def plot_uncertainty_vs_accuracy(m_omega, y_true, y_pred):
    # Bin by uncertainty deciles
    bins = np.percentile(m_omega, np.arange(0, 101, 10))
    bin_indices = np.digitize(m_omega, bins)

    bin_accuracies = []
    bin_centers = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).mean()
            bin_accuracies.append(acc)
            bin_centers.append((bins[i - 1] + bins[i]) / 2)

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Uncertainty Mass m(Ω)', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title('Calibration: Uncertainty vs Accuracy')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('uncertainty_vs_error.pdf')

    # Compute correlation
    r, p = pearsonr(m_omega, (y_true == y_pred).astype(float))
    print(f"Pearson correlation: r={r:.3f}, p={p:.4f}")


# PLOT 2: Uncertainty by Class
def analyze_uncertainty_by_class(m_omega, y_true, y_pred):
    classes = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
    results = []

    for c in range(4):
        mask = y_true == c
        avg_omega = m_omega[mask].mean()
        std_omega = m_omega[mask].std()
        high_unc_pct = (m_omega[mask] > 0.5).mean() * 100
        accuracy = (y_true[mask] == y_pred[mask]).mean() * 100

        results.append({
            'Class': classes[c],
            'Avg m(Ω)': f"{avg_omega:.2f} ± {std_omega:.2f}",
            '% High Unc': f"{high_unc_pct:.1f}%",
            'Accuracy': f"{accuracy:.1f}%"
        })

    df = pd.DataFrame(results)
    latex_table = df.to_latex(index=False)
    # Using logger.info to ensure the full table is captured in the log file
    logger.info("\nLaTeX Table (Uncertainty by Class):\n" + latex_table)


# PLOT 3: Human-in-the-loop routing
def plot_routing_analysis(
    m_omega,
    y_true,
    y_pred,
    class_ids=(1, 2, 3),
    class_names=('High Speed', 'High Path', 'Late Flaps'),
    thresholds=None,
    human_accuracy=0.90,
    review_cap=0.30,  # annotate best point with review rate <= 30%
    save_path_pdf='Fig/routing_analysis.pdf',
    save_path_png='Fig/routing_analysis.png',
):
    """
    Human-in-the-loop workload frontier:
      x-axis: % of flights sent to human review (overall, via m(Ω) > threshold)
      y-axis: recall for each anomaly class if reviewed cases are corrected by a human with `human_accuracy`

    - Shades the gain above baseline recall
    - Adds Wilson 95% CI ribbons for stability
    - Annotates the best point under `review_cap`
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- helpers ----------
    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan, np.nan)
        p = k / n
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        half  = z * np.sqrt((p*(p-1) + 0.25*z**2/n + p*(1-p)) / n) / denom
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        return (center, lo, hi)

    def percent(x):  # 0-1 to percent
        return 100.0 * np.asarray(x, dtype=float)

    # ---------- inputs ----------
    m_omega = np.asarray(m_omega, dtype=float)
    y_true  = np.asarray(y_true, dtype=int)
    y_pred  = np.asarray(y_pred, dtype=int)
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.80, 13)  # 0.20..0.80 by 0.05

    # overall review mask per threshold -> workload on the x-axis
    # (operationally: safety team workload is driven by total volume flagged)
    review_rates = []
    for t in thresholds:
        review_rates.append((m_omega > t).mean())
    review_rates = np.asarray(review_rates)

    # ---------- compute per-class curves ----------
    # consistent class colors (colorblind-friendly-ish)
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # figure layout: 2x2, three plots + shared legend/policy panel
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    axs = [axes[0,0], axes[0,1], axes[1,0]]
    ax_legend = axes[1,1]
    ax_legend.axis('off')

    legend_handles = []
    summary_lines = []

    for ax, c_id, c_name, color in zip(axs, class_ids, class_names, class_colors):
        mask_c = (y_true == c_id)
        n_c = int(mask_c.sum())
        if n_c == 0:
            ax.text(0.5, 0.5, f'No samples for {c_name}', ha='center', va='center')
            continue

        baseline_k = int((y_pred[mask_c] == c_id).sum())
        baseline_rec, base_lo, base_hi = wilson_ci(baseline_k, n_c)

        # arrays across thresholds
        imp_center, imp_lo, imp_hi = [], [], []

        # compute improved recall at each threshold
        for t in thresholds:
            flagged_overall = (m_omega > t)
            # among class-c items, which are flagged
            flagged_c = mask_c & flagged_overall

            # auto correct = correct predictions among non-flagged class-c items
            auto_correct = ((y_pred == c_id) & mask_c & (~flagged_overall)).sum()

            # human correct = flagged class-c items corrected with human_accuracy
            human_n = int(flagged_c.sum())
            # expectation for recall: add human_accuracy * n_flagged_c
            k = auto_correct + human_accuracy * human_n

            center, lo, hi = wilson_ci(k, n_c)
            imp_center.append(center)
            imp_lo.append(lo)
            imp_hi.append(hi)

        imp_center = np.asarray(imp_center)
        imp_lo     = np.asarray(imp_lo)
        imp_hi     = np.asarray(imp_hi)

        # x-axis: workload as overall review rate
        x = percent(review_rates)
        y = percent(imp_center)
        y_lo = percent(imp_lo)
        y_hi = percent(imp_hi)
        y_base = percent([baseline_rec]*len(x))

        # shade the gain above baseline
        ax.fill_between(x, y_base, y, where=(y > y_base), color=color, alpha=0.10, linewidth=0)

        # improved curve with CI
        line, = ax.plot(x, y, color=color, linewidth=2.2, marker='o', markersize=4, label=c_name)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

        # baseline line
        ax.plot(x, y_base, color='#555555', linestyle='--', linewidth=1.5, label='Baseline recall')

        # choose best point under review cap
        cap_mask = (review_rates <= review_cap)
        if cap_mask.any():
            j = np.nanargmax(imp_center[cap_mask])
            x_star = x[cap_mask][j]
            y_star = y[cap_mask][j]
            ax.scatter([x_star], [y_star], s=48, color=color, edgecolors='black', linewidths=0.8, zorder=5)
            ax.annotate(
                f'Best ≤{int(review_cap*100)}%\n{y_star:.1f}%',
                xy=(x_star, y_star),
                xytext=(6, 10),
                textcoords='offset points',
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor=color),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2)
            )

            # add to summary
            summary_lines.append(
                f'{c_name}: baseline {percent(baseline_rec):.1f}% → {y_star:.1f}% at {x_star:.0f}% review'
            )

        ax.set_title(c_name)
        ax.set_xlabel('Review rate (%)')
        ax.set_ylabel('Recall (%)')
        ax.set_xlim(0, max(50, np.ceil(x.max()/5)*5))  # clamp to a clean right bound
        ax.set_ylim(40, 100)
        ax.grid(alpha=0.3, linestyle=':')

        legend_handles.append(line)

    # shared legend + policy notes
    leg = ax_legend.legend(
        handles=legend_handles,
        labels=class_names,
        loc='upper left',
        framealpha=0.95,
        title='Classes'
    )
    leg.get_title().set_fontweight('bold')

    # policy box
    note = [
        f'Assumption: human accuracy = {int(human_accuracy*100)}%',
        f'Best-under-cap points shown (cap = {int(review_cap*100)}%)',
        'Shaded region: gain over baseline recall'
    ]
    if summary_lines:
        note.append('')
        note.append('Summary at cap:')
        note.extend(summary_lines)

    ax_legend.text(
        0.02, 0.40,
        '\n'.join(note),
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='#888'),
        ha='left', va='top'
    )

    fig.suptitle('Human-in-the-loop workload frontier\n(uncertainty-based routing with DST $m(\\Omega)$)', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path_pdf), exist_ok=True)
    fig.savefig(save_path_pdf, bbox_inches='tight')
    fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)



# PLOT 4: DST uncertainty vs XGBoost confidence
def plot_uncertainty_comparison(m_omega, xgb_probs, y_true, y_pred):
    xgb_confidence = xgb_probs.max(axis=1)

    # 2D histogram showing accuracy in each bin
    bins_conf = np.linspace(0.3, 1.0, 15)
    bins_unc = np.linspace(0.2, 0.6, 15)

    accuracy_grid = np.zeros((len(bins_unc) - 1, len(bins_conf) - 1))

    for i in range(len(bins_unc) - 1):
        for j in range(len(bins_conf) - 1):
            mask = (m_omega >= bins_unc[i]) & (m_omega < bins_unc[i + 1]) & \
                   (xgb_confidence >= bins_conf[j]) & (xgb_confidence < bins_conf[j + 1])
            if mask.sum() > 10:  # Need enough samples
                accuracy_grid[i, j] = (y_true[mask] == y_pred[mask]).mean()
            else:
                accuracy_grid[i, j] = np.nan

    plt.figure(figsize=(10, 7))
    plt.imshow(accuracy_grid, origin='lower', aspect='auto', cmap='RdYlGn',
               vmin=0.7, vmax=1.0, extent=[bins_conf[0], bins_conf[-1],
                                           bins_unc[0], bins_unc[-1]])
    plt.colorbar(label='Accuracy')
    plt.xlabel('XGBoost Confidence (max probability)')
    plt.ylabel('DST Uncertainty m(Ω)')
    plt.title('Orthogonality of Uncertainty Sources')

    # Add correlation
    r, p = pearsonr(xgb_confidence, m_omega)
    plt.text(0.05, 0.95, f'Pearson r = {r:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('uncertainty_comparison.pdf')

    print(f"Correlation: r={r:.3f}, p={p:.4f}")


def plot_uncertainty_calibration(m_omega, y_true, y_pred, logger):
    """
    Generates a bar plot showing prediction accuracy across m(Omega) quartiles with confidence intervals.
    """
    # Compute quartiles
    q = np.percentile(m_omega, [0, 25, 50, 75, 100])

    # Calculate accuracy for each quartile
    accuracies = []
    ci_lower = []
    ci_upper = []
    labels = []
    n_samples = []

    for i in range(len(q) - 1):
        if i == len(q) - 2:
            mask = (m_omega >= q[i]) & (m_omega <= q[i + 1])
        else:
            mask = (m_omega >= q[i]) & (m_omega < q[i + 1])

        # Ensure there are samples in the mask before calculating
        if mask.sum() == 0:
            logger.warning(f"No samples in m(Omega) quartile [{q[i]:.2f}, {q[i + 1]:.2f}]. Skipping.")
            continue

        correct = (y_true[mask] == y_pred[mask]).astype(float)
        acc = correct.mean()
        n = len(correct)

        # 95% CI
        # stats.sem requires at least 2 samples, stats.t.interval requires n-1 > 0
        if n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=acc, scale=stats.sem(correct))
            ci_l, ci_u = acc - ci[0], ci[1] - acc
        else:  # Handle cases with 0 or 1 samples (no meaningful CI)
            ci_l, ci_u = 0.0, 0.0

        accuracies.append(acc)
        ci_lower.append(ci_l)
        ci_upper.append(ci_u)
        labels.append(f'Q{i + 1}\n[{q[i]:.2f}, {q[i + 1]:.2f}]')
        n_samples.append(n)

    # If no valid quartiles were processed, exit
    if not accuracies:
        logger.warning("No valid accuracy data to plot for uncertainty calibration.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(accuracies))

    bars = ax.bar(x, accuracies, yerr=[ci_lower, ci_upper],
                  capsize=8, alpha=0.75, color='steelblue',
                  edgecolor='navy', linewidth=1.5)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('Uncertainty Mass $m(\Omega)$ Quartile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
    # Set ylim dynamically if needed, or based on overall expected accuracy
    min_ylim = min(accuracies) * 0.95
    max_ylim = max(accuracies) * 1.05
    ax.set_ylim([max(0, min_ylim - 0.01), min(1, max_ylim + 0.01)])  # Ensure 0-1 range

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add overall accuracy if y_true is not empty
    if len(y_true) > 0:
        overall_acc = (y_true == y_pred).mean()
        ax.axhline(overall_acc, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Overall accuracy')

    # Add sample counts above bars
    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        height = bar.get_height()
        # Adjust vertical position of text based on error bar upper limit
        text_y_pos = height + ci_upper[i] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01  # Dynamic offset
        ax.text(bar.get_x() + bar.get_width() / 2., text_y_pos,
                f'n={n}', ha='center', va='bottom', fontsize=9, color='gray')

    # Add correlation annotation
    # Ensure m_omega and comparison array have same length and are not empty
    if len(m_omega) > 1 and len(y_true) == len(m_omega):
        r, p = stats.pearsonr(m_omega, (y_true == y_pred).astype(float))
        textstr = f'Pearson $r = {r:.3f}$\n$p < 0.001$' if p < 0.001 else f'Pearson $r = {r:.3f}$\n$p = {p:.3f}$'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
    else:
        logger.warning("Not enough data to compute Pearson correlation for plot annotation.")

    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    # Ensure 'Fig' directory exists
    if not os.path.exists('Fig'):
        os.makedirs('Fig')
    plt.savefig('Fig/uncertainty_calibration.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free up memory
    logger.info("Generated Uncertainty Calibration Plot: Fig/uncertainty_calibration.pdf")

    logger.info(f"\n=== QUARTILE STATISTICS ===")
    for i, (acc, label) in enumerate(zip(accuracies, labels)):
        logger.info(f"{label.replace(chr(10), ' ')}: {acc:.4f} ({acc * 100:.2f}%)")


def perform_dst_analysis(all_fold_data: List[Dict]):
    """
    Aggregates data from all folds and runs DST analysis + visualizations.
    """
    logger.info("=" * 50)
    logger.info("UNCERTAINTY AND ROUTING ANALYSIS")
    logger.info("=" * 50)

    # Import visualization functions
    from visualizations import (
        create_uncertainty_heatmap,
        create_class_uncertainty_profiles,
        plot_reliability_slices
    )

    # 1. Aggregate Data (your existing code)
    all_y_true = []
    all_y_pred = []
    all_m_omega = []
    all_xgb_probs = []

    for fold_data in all_fold_data:
        preds_data = fold_data.get('predictions', {})
        y_true = fold_data.get("y_test")
        y_pred = preds_data.get("final")
        binary_probs = preds_data.get("binary_probs")
        xgb_probs = preds_data.get("xgb_probs")

        if y_true is None or y_pred is None or binary_probs is None or xgb_probs is None:
            logger.warning(f"Skipping fold {fold_data.get('fold', 'N/A')}: Missing data.")
            continue

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_xgb_probs.append(xgb_probs)

        m_omega = [prob.get('Ω', 0.0) for prob in binary_probs]
        all_m_omega.extend(m_omega)

    if not all_y_true:
        logger.error("No valid data for analysis.")
        return

    y_true = np.array(all_y_true, dtype=np.int8)
    y_pred = np.array(all_y_pred, dtype=np.int8)
    m_omega = np.array(all_m_omega, dtype=np.float32)
    xgb_probs = np.concatenate(all_xgb_probs, axis=0)

    # === 2. Generate Show-Stopping Visualizations ===
    try:
        logger.info("\n" + "=" * 70)
        logger.info(" GENERATING AWARD-QUALITY VISUALIZATIONS")
        logger.info("=" * 70)

        if not os.path.exists('Fig'):
            os.makedirs('Fig')

        # Create aggregate results dict for visualization functions
        aggregate_results = {
            'predictions': {
                'final_pignistics': [{'Ω': m} for m in m_omega],
                'xgb_probs': xgb_probs,
                'final': y_pred
            }
        }

        # === SHOW-STOPPER 1: Uncertainty Landscape ===
        logger.info("\n[NEW] Generating 2D Uncertainty Heatmap...")
        create_uncertainty_heatmap(aggregate_results, y_true, fold_idx='aggregate')
        logger.info("✓ Generated: Fig/uncertainty_heatmap.pdf")

        # === SHOW-STOPPER 2: Class Profiles ===
        logger.info("\n[NEW] Generating Class Uncertainty Profiles...")
        create_class_uncertainty_profiles(aggregate_results, y_true, fold_idx='aggregate')
        logger.info("✓ Generated: Fig/class_uncertainty_profiles_fold_aggregate.pdf")

        logger.info("\n[NEW] Generating Reliability Slices (confidence X uncertainty...")
        plot_reliability_slices(m_omega, xgb_probs, y_true, y_pred,save_path='Fig/reliability_slices.pdf')
        logger.info("✓ Generated: Fig/reliability_slices.pdf")

        # === Your existing analyses ===
        logger.info("\n[1] Uncertainty Calibration Plot")
        plot_uncertainty_calibration(m_omega, y_true, y_pred, logger)

        logger.info("\n[2] Uncertainty vs Accuracy")
        plot_uncertainty_vs_accuracy(m_omega, y_true, y_pred)

        logger.info("\n[3] Uncertainty by Class Table")
        analyze_uncertainty_by_class(m_omega, y_true, y_pred)

        logger.info("\n[4] Routing Analysis")
        plot_routing_analysis(m_omega, y_true, y_pred)

        logger.info("\n[5] Uncertainty Comparison")
        plot_uncertainty_comparison(m_omega, xgb_probs, y_true, y_pred)

        logger.info("\n" + "=" * 70)
        logger.info(" ALL VISUALIZATIONS COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        logger.error(traceback.format_exc())

def perform_swc_aviation_validation(all_fold_data: List[Dict]):
    """
    Aggregate per-fold XGBoost + DST-like outputs and run the clean SWC fusion
    validation using SWC.SWC_aviation.run_swc_validation.
    """
    logger.info("=" * 50)
    logger.info("SWC AVIATION VALIDATION (CLEAN SWC FUSION)")
    logger.info("=" * 50)

    xgb_probs_list = []
    dst_support_list = []
    dst_against_list = []
    dst_ignorance_list = []
    labels_list = []

    n_classes = CONFIG['n_classes']

    for fold_data in all_fold_data:
        preds = fold_data.get('predictions', {})
        y_true = fold_data.get('y_test')

        xgb_probs        = preds.get('xgb_probs')
        final_pignistics = preds.get('final_pignistics')

        fold_id = fold_data.get('fold', 'N/A')

        if y_true is None or xgb_probs is None or final_pignistics is None:
            logger.warning(
                f"Skipping fold {fold_id} for SWC validation: "
                f"missing one of [y_true, xgb_probs, final_pignistics]."
            )
            continue

        y_true = np.asarray(y_true)
        xgb_probs = np.asarray(xgb_probs)
        n_samples = len(y_true)

        # Allocate DST mass arrays for this fold
        dst_support   = np.zeros((n_samples, n_classes), dtype=np.float32)
        dst_against   = np.zeros((n_samples, n_classes), dtype=np.float32)
        dst_ignorance = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            pign = final_pignistics[i] or {}
            m_theta = float(pign.get('Ω', 0.0))
            dst_ignorance[i] = m_theta

            # For each class c, treat:
            #   m({c}) = pign.get(str(c), 0)
            #   m({¬c}) = 1 - m({c}) - m(Θ)
            for c in range(n_classes):
                m_h = float(pign.get(str(c), 0.0))
                m_not_h = max(0.0, 1.0 - m_h - m_theta)
                dst_support[i, c] = m_h
                dst_against[i, c] = m_not_h

        xgb_probs_list.append(xgb_probs)
        dst_support_list.append(dst_support)
        dst_against_list.append(dst_against)
        dst_ignorance_list.append(dst_ignorance)
        labels_list.append(y_true)

    if not labels_list:
        logger.error("No folds had the required outputs for SWC validation.")
        return

    # Concatenate across folds
    xgb_probs     = np.concatenate(xgb_probs_list, axis=0)
    dst_support   = np.concatenate(dst_support_list, axis=0)
    dst_against   = np.concatenate(dst_against_list, axis=0)
    dst_ignorance = np.concatenate(dst_ignorance_list, axis=0)
    labels        = np.concatenate(labels_list, axis=0)

    results_from_paper_a = {
        'xgb_probs':     xgb_probs,
        'dst_support':   dst_support,
        'dst_against':   dst_against,
        'dst_ignorance': dst_ignorance,  # (n_samples,)
        'labels':        labels,
    }

    swc_results = run_swc_validation(results_from_paper_a)

    calib   = swc_results['calibration']
    metrics = swc_results['metrics']

    logger.info(
        f"SWC calibration: r = {calib['r']:.3f}, p = {calib['p']:.4f}"
    )
    logger.info(f"SWC vs XGBoost metrics: {metrics}")





def run_experiment(
    file_path: str,
    config: Dict,
    run_dst: bool = True,
    run_swc: bool = True
):
    """
    High-level driver for the full experiment pipeline.
    Other scripts, notebooks, hyperparameter sweeps should call this.
    """

    logger.info("=== Starting experiment ===")

    # Step 1: CV
    all_fold_results = run_cross_validation(file_path, config)

    # Step 2: Summaries
    summarize_results(all_fold_results)

    # Step 3: Optional analyses
    if run_dst:
        perform_dst_analysis(all_fold_results)

    if run_swc:
        perform_swc_aviation_validation(all_fold_results)

    logger.info("=== Experiment complete ===")
    return all_fold_results


if __name__ == '__main__':
    run_experiment(
        file_path="C:/Users/joshp_ya/DASHlink_full_fourclass_raw_comp.npz",
        config=CONFIG,
        run_dst=True,
        run_swc=True
    )


