import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix




#Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_kfold_accuracy(all_fold_data):
    """
    Plot the accuracy across different k-folds
    :param all_fold_data:
    :return:
    """
    folds = [data['fold'] for data in all_fold_data]
    accuracies = [data['accuracy'] for data in all_fold_data]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(folds, accuracies)


    avg_accuracy = np.mean(accuracies)
    ax.axhline(y=avg_accuracy, color='r', linestyle='--', alpha=0.5, label=f'Average Accuracy: {avg_accuracy:.4f}')

    ax.set_xlabel('K-Fold Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('K-Fold Accuracy by Fold')
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('kfold_accuracy.png', dpi=300)
    plt.show()
    return fig


def plot_all_fold_confusion_matrices(all_fold_data, class_names=None):
    """
    Plot confusion matrices for all folds and an average confusion matrix

    :param all_fold_data: List of dictionaries containing fold results with y_true and y_pred
    :param class_names: Names of the classes (optional)
    :return: Figure for the average confusion matrix
    """
    # First, create individual confusion matrices for each fold
    num_folds = len(all_fold_data)
    fig = plt.figure(figsize=(15, 5 * ((num_folds + 1) // 2)))

    # Keep track of the sum of confusion matrices for averaging
    sum_cm = None

    for i, fold_data in enumerate(all_fold_data):
        y_true = fold_data['y_test'][:len(fold_data['predicted_classes'])]
        y_pred = fold_data['predicted_classes']
        cm = confusion_matrix(y_true, y_pred)

        # Initialize sum_cm with the right shape if it's the first fold
        if sum_cm is None:
            sum_cm = np.zeros_like(cm, dtype=float)

        # Add to the sum for later averaging
        sum_cm += cm

        # Plot individual fold confusion matrix
        plt.subplot(((num_folds + 1) // 2), 2, i + 1)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Fold {fold_data["fold"]} Confusion Matrix')

        if class_names is not None:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)

    # Plot the average confusion matrix (optional)
    avg_cm = sum_cm / num_folds

    # Create a separate figure for the average confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, cmap='Blues', fmt='.1f')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Confusion Matrix Across All Folds')

    if class_names is not None:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)

    plt.tight_layout()
    plt.savefig('average_confusion_matrix.png', dpi=300)

    # Save the multi-fold figure
    fig.tight_layout()
    fig.savefig('all_fold_confusion_matrices.png', dpi=300)

    plt.show()
    return plt.gcf()

def plot_class_distribution(labels):
    """
    Plot the class distribution
    :param labels:
    :return:
    """
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    class_names = [f'Class {cls}' for cls in unique_labels]

    # Create just the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(class_counts, labels=class_names, autopct='%1.1f%%',
            colors=sns.color_palette("viridis", len(class_names)),
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('Class Distribution (Percentage)', fontsize=14)

    plt.tight_layout()
    plt.savefig('class_distribution_pie.png', dpi=300)
    plt.show()

    return plt.gcf()
