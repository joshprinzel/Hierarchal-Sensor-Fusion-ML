
import os
import psutil
import traceback
import gc
import logging
import numpy as np
from typing import List, Dict, Tuple, Any

# Import your existing modules
from dataloader import load_file, trim_timesteps, apply_time_windows
from five_fold_cv import create_valid_combinations
from likelihood_distribution import create_likelihood_distribution, validate_likelihood_distribution
from sensor_fusion import perform_sensor_fusion, plot_confusion_matrix, plot_roc_curve
from sample_by_class import sample_flights_by_class
from ordinal_pattern import extract_ordinal_patterns_for_combination

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import entropy

# Configuration dictionary to centralize parameters
CONFIG = {
    'kfold': 5,
    'max_combinations': 20,  # Your setting, can adjust to 20 as per Garcia et al.
    'timesteps': 70,
    'window_size': 10,
    'stride': 5,
    'n_classes': 4,
    'nominal_proportion': 0.8997,  # Your consistent value
    'train_anomaly_total': 760,  # 760 per anomaly class for balanced training
    'test_total': 3000,  # 760 per class for balanced testing
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
            # This is the fix for the normalization data leak.
            # We create a new normalization function that can fit and transform separately.
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
            # This uses only the training data for the fold, which is correct.
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
        # This uses ONLY the training patterns, which is correct.
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
        # This is the fix for the uncertainty data leak.
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
            # Critical Fix: Use training data as the reference for uncertainty
            all_patterns=train_patterns,
            all_labels=y_train,
            fold_idx=fold_idx,
            # Pass relevant config values
            window_size=config['window_size'],
            stride=config['stride'],
            total_timesteps=config['timesteps'],
        )


        results['y_test'] = y_test
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
        fold = fold_data["fold"]
        y_true = fold_data["y_test"]
        final_preds = fold_data["final_preds"]
        binary_preds = fold_data["binary_preds"]
        binary_probs = fold_data["binary_probs"]
        logger.info(f"Fold {fold} - binary_probs structure: {type(binary_probs[0])}, sample: {binary_probs[0]}")
        all_y_true.extend(y_true)
        all_final_preds.extend(final_preds)
        all_binary_preds.extend(binary_preds)
        all_binary_probs.extend(binary_probs)

        logger.info(f"Fold {fold} - F1 Micro: {fold_data['f1_micro']:.4f}, MCC: {fold_data['mcc']:.4f}")
        for i, cls in class_names.items():
            precision = fold_data['precision_per_class'][i]
            recall = fold_data['recall_per_class'][i]
            f1 = fold_data['f1_per_class'][i]
            logger.info(f"  - {cls}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        total_f1_micro += fold_data["f1_micro"]
        total_mcc += fold_data["mcc"]

    n_folds = len(all_fold_data)
    logger.info(f"Average F1 Micro: {total_f1_micro / n_folds:.4f}, Average MCC: {total_mcc / n_folds:.4f}")

    all_y_true = np.array(all_y_true, dtype=np.int8)
    all_final_preds = np.array(all_final_preds, dtype=np.int8)
    all_binary_preds = np.array(all_binary_preds, dtype=np.int8)
    all_binary_probs_anomaly = np.array([prob['1'] for prob in all_binary_probs], dtype=np.float32)
    logger.info(f"Aggregate F1 Micro: {f1_score(all_y_true, all_final_preds, average='micro'):.4f}")
    logger.info(f"Aggregate MCC: {matthews_corrcoef(all_y_true, all_final_preds):.4f}")

    binary_true = (all_y_true > 0).astype(np.int8)
    binary_cm = confusion_matrix(binary_true, all_binary_preds, labels=[0, 1])
    logger.info(f"Aggregate Binary Stage Confusion Matrix:\n{binary_cm}")

    precision_agg = precision_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']),
                                    zero_division=0)
    recall_agg = recall_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']),
                              zero_division=0)
    f1_agg = f1_score(all_y_true, all_final_preds, average=None, labels=range(CONFIG['n_classes']), zero_division=0)
    logger.info("Aggregate Per-Class Metrics:")
    for i, cls in class_names.items():
        logger.info(f"  - {cls}: Precision: {precision_agg[i]:.4f}, Recall: {recall_agg[i]:.4f}, F1: {f1_agg[i]:.4f}")

    # Compute and plot aggregate confusion matrix
    aggregate_cm = confusion_matrix(all_y_true, all_final_preds, labels=range(CONFIG['n_classes']))
    class_names_list = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']
    plot_confusion_matrix(aggregate_cm, class_names_list, fold_idx='aggregate', title="Hierarchical Fusion (All Folds)",
                          save_path="confusion_matrix_aggregate_hierarchical_fusion.png")
    logger.info(f"Aggregate Confusion Matrix:\n{aggregate_cm}")
    logger.info(f"Saved aggregate confusion matrix to confusion_matrix_aggregate_hierarchical_fusion.png")

    # Compute and plot aggregate ROC curve
    fpr, tpr, _ = roc_curve(binary_true, all_binary_probs_anomaly)
    aggregate_auc = roc_auc_score(binary_true, all_binary_probs_anomaly)
    plot_roc_curve(binary_true, all_binary_probs_anomaly, fold_idx='aggregate',
                   save_path="roc_curve_aggregate_hierarchical_fusion.png")
    logger.info(f"Aggregate Binary AUC: {aggregate_auc:.4f}")
    logger.info(f"Saved aggregate ROC curve to roc_curve_aggregate_hierarchical_fusion.png")


logger = setup_logging()

if __name__ == '__main__':
    logger = setup_logging()
    file_path = "C:/Users/joshp_ya/DASHlink_full_fourclass_raw_comp.npz"

    # Run the entire cross-validation process
    all_fold_results = run_cross_validation(file_path, CONFIG)

    # Summarize the results from all folds
    summarize_results(all_fold_results)
