import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def plot_target_distribution(y, title="Target Distribution", labels=None):
    if labels is None:
        labels = ['Class 0', 'Class 1']
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0.05), shadow=True)
    ax2.set_title(f'{title} - Percentage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_train_test_split(y_train, y_test, labels=None):
    if labels is None:
        labels = ['Class 0', 'Class 1']
    train_counts = [np.sum(y_train == i) for i in range(len(labels))]
    test_counts = [np.sum(y_test == i) for i in range(len(labels))]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training Set', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test Set', color='#e67e22', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Train-Test Split Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MODEL TRAINING PLOTS
# ============================================================================

def plot_training_history(history, metrics=['loss'], title="Training History"):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    for idx, metric in enumerate(metrics):
        if metric in history:
            iterations = range(1, len(history[metric]) + 1)
            axes[idx].plot(iterations, history[metric], 
                          linewidth=2, color='#2c3e50', marker='o', 
                          markersize=3, markevery=max(1, len(history[metric])//20))
            axes[idx].set_xlabel('Iteration', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric.capitalize()} over Iterations', 
                               fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            final_value = history[metric][-1]
            axes[idx].annotate(f'Final: {final_value:.4f}',
                              xy=(len(history[metric]), final_value),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                              fontsize=10, fontweight='bold')
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", cmap='Blues', normalize=False):
    if labels is None:
        labels = ['Class 0', 'Class 1']
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        data = cm_normalized
    else:
        fmt = 'd'
        data = cm
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, 
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                       ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    return fig


# ============================================================================
# METRICS COMPARISON
# ============================================================================

def plot_metrics_comparison(metrics_dict, title="Model Performance Metrics"):
    """
    Vẽ biểu đồ so sánh các metrics
    metrics_dict: {'Accuracy': 0.85, 'Precision': 0.82, ...}
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
    bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.4f}',
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_train_test_comparison(train_metrics, test_metrics, 
                               title="Train vs Test Performance"):
    """
    So sánh metrics giữa train và test sets
    """
    metrics = list(train_metrics.keys())
    train_values = list(train_metrics.values())
    test_values = list(test_metrics.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Training Set',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(feature_names, weights, top_n=15, 
                           title="Feature Importance"):
    """
    Vẽ biểu đồ feature importance từ model weights
    """
    # Create list of (name, weight, abs_weight)
    features = [(name, weight, abs(weight)) 
                for name, weight in zip(feature_names, weights)]
    
    # Sort by absolute weight
    features_sorted = sorted(features, key=lambda x: x[2], reverse=True)[:top_n]
    
    names = [f[0] for f in features_sorted]
    weights_vals = [f[1] for f in features_sorted]
    
    # Reverse for plotting (highest at top)
    names = names[::-1]
    weights_vals = weights_vals[::-1]
    
    # Colors based on positive/negative
    colors = ['#e74c3c' if w > 0 else '#2ecc71' for w in weights_vals]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(names, weights_vals, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, weight in zip(bars, weights_vals):
        width = bar.get_width()
        label_x = width + (0.002 if width > 0 else -0.002)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{weight:.4f}',
               ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Increase Churn Risk'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Decrease Churn Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# PREDICTION ANALYSIS
# ============================================================================

def plot_prediction_distribution(y_true, y_pred, y_proba=None,
                                 title="Prediction Distribution"):
    """
    Vẽ phân bố predictions
    """
    if y_proba is None:
        # Only plot classification
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create confusion categories
        categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        counts = [
            np.sum((y_true == 0) & (y_pred == 0)),  # TN
            np.sum((y_true == 0) & (y_pred == 1)),  # FP
            np.sum((y_true == 1) & (y_pred == 0)),  # FN
            np.sum((y_true == 1) & (y_pred == 1))   # TP
        ]
        colors = ['#2ecc71', '#e67e22', '#e74c3c', '#3498db']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        return fig
    
    else:
        # Plot probability distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of probabilities by true class
        ax1.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='Class 0 (True)',
                color='#2ecc71', edgecolor='black')
        ax1.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label='Class 1 (True)',
                color='#e74c3c', edgecolor='black')
        ax1.axvline(x=0.5, color='black', linewidth=2, linestyle='--', label='Threshold')
        ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Probability Distribution by True Class', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [y_proba[y_true == 0], y_proba[y_true == 1]]
        bp = ax2.boxplot(data_to_plot, labels=['Class 0', 'Class 1'],
                        patch_artist=True, widths=0.6)
        
        colors_box = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.axhline(y=0.5, color='black', linewidth=2, linestyle='--', label='Threshold')
        ax2.set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Probability Distribution Boxplot', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    Vẽ ROC Curve
    """
    # Calculate ROC points
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC
    auc = np.trapz(tpr_list, fpr_list)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr_list, tpr_list, linewidth=3, color='#e74c3c', label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], linewidth=2, linestyle='--', color='gray', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Fill area under curve
    ax.fill_between(fpr_list, tpr_list, alpha=0.2, color='#e74c3c')
    
    plt.tight_layout()
    return fig


# ============================================================================
# RESIDUAL PLOTS (for Regression)
# ============================================================================

def plot_residuals(y_true, y_pred, title="Residual Analysis"):
    """
    Vẽ residual plots cho regression models
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, color='#3498db', edgecolors='black')
    axes[0, 0].axhline(y=0, color='red', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Q-Q plot (approximate)
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
    axes[1, 0].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5, 
                      color='#e67e22', edgecolors='black')
    axes[1, 0].plot([-3, 3], [-3*np.std(residuals), 3*np.std(residuals)], 
                   'r--', linewidth=2)
    axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, color='#9b59b6', edgecolors='black')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def save_all_figures(figures_dict, output_dir='../output/figures'):
    """
    Lưu tất cả figures vào folder
    figures_dict: {'filename': fig_object}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, fig in figures_dict.items():
        filepath = os.path.join(output_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filepath}")
