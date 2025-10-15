"""
Advanced Model Evaluation and Visualization Module

This module provides comprehensive evaluation metrics and visualizations for 
cryptocurrency prediction models, suitable for academic research and publication.

Authors: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics."""
    
    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Probabilistic metrics
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    
    # Advanced metrics
    matthews_corrcoef: float
    cohen_kappa: float
    youden_index: float
    
    # Custom financial metrics
    hit_rate: float
    false_alarm_rate: float
    profit_factor: float
    sharpe_ratio: float
    
    # Statistical significance
    mcnemar_statistic: float
    mcnemar_pvalue: float


class AdvancedModelEvaluator:
    """Advanced model evaluation with comprehensive metrics and visualizations."""
    
    def __init__(self, model_name: str = "XGBoost Model"):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.evaluation_results = {}
        
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             threshold: float = 0.5) -> ModelPerformanceMetrics:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            ModelPerformanceMetrics object with all metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Probabilistic metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        logloss = log_loss(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        # Advanced metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        youden = recall + specificity - 1
        
        # Financial metrics
        hit_rate = recall  # Same as sensitivity/recall
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Profit factor (simplified)
        true_positives_return = np.sum((y_true == 1) & (y_pred == 1))
        false_positives_cost = np.sum((y_true == 0) & (y_pred == 1))
        profit_factor = true_positives_return / max(false_positives_cost, 1)
        
        # Sharpe ratio approximation
        returns = np.where(y_pred == 1, np.where(y_true == 1, 1, -1), 0)
        sharpe = np.mean(returns) / max(np.std(returns), 1e-8) if np.std(returns) > 0 else 0
        
        # Statistical significance (McNemar's test)
        # For model comparison - here we use a simplified version
        mcnemar_stat = ((fp - fn) ** 2) / (fp + fn) if (fp + fn) > 0 else 0
        mcnemar_pval = 1 - stats.chi2.cdf(mcnemar_stat, 1) if mcnemar_stat > 0 else 1
        
        return ModelPerformanceMetrics(
            accuracy=accuracy, precision=precision, recall=recall, f1_score=f1,
            specificity=specificity, roc_auc=roc_auc, pr_auc=pr_auc,
            log_loss=logloss, brier_score=brier, matthews_corrcoef=mcc,
            cohen_kappa=kappa, youden_index=youden, hit_rate=hit_rate,
            false_alarm_rate=false_alarm_rate, profit_factor=profit_factor,
            sharpe_ratio=sharpe, mcnemar_statistic=mcnemar_stat,
            mcnemar_pvalue=mcnemar_pval
        )
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          save_path: Optional[str] = None) -> None:
        """Plot ROC and Precision-Recall curves."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {self.model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        ax2.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {self.model_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC and PR curves saved to: {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """Plot threshold analysis for optimal threshold selection."""
        
        thresholds = np.linspace(0.1, 0.9, 100)
        metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'youden_index': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1_score'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            
            # Youden's index
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['youden_index'].append(sensitivity + specificity - 1)
        
        plt.figure(figsize=(12, 8))
        
        for metric, values in metrics.items():
            plt.plot(thresholds, values, label=metric.replace('_', ' ').title(), linewidth=2)
        
        # Find optimal thresholds
        optimal_f1_idx = np.argmax(metrics['f1_score'])
        optimal_youden_idx = np.argmax(metrics['youden_index'])
        
        plt.axvline(x=thresholds[optimal_f1_idx], color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal F1 Threshold: {thresholds[optimal_f1_idx]:.3f}')
        plt.axvline(x=thresholds[optimal_youden_idx], color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal Youden Threshold: {thresholds[optimal_youden_idx]:.3f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title(f'Threshold Analysis - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to: {save_path}")
        
        plt.show()
        
        return thresholds[optimal_f1_idx], thresholds[optimal_youden_idx]
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """Plot prediction probability distributions by class."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        ax1.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Class 0 (No Rise)',
                density=True, color='lightcoral')
        ax1.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Class 1 (Rise)',
                density=True, color='lightblue')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Prediction Probability Distribution - {self.model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
        box_plot = ax2.boxplot(data_to_plot, labels=['Class 0 (No Rise)', 'Class 1 (Rise)'],
                              patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        box_plot['boxes'][1].set_facecolor('lightblue')
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Prediction Probability Box Plot by Class')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, save_path: Optional[str] = None) -> None:
        """Plot calibration curve to assess prediction reliability."""
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.figure(figsize=(10, 8))
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                linewidth=2, label=f'{self.model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot - {self.model_name}\\n'
                 f'Calibration Error: {calibration_error:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to: {save_path}")
        
        plt.show()
        
        return calibration_error
    
    def plot_confusion_matrix_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     save_path: Optional[str] = None) -> None:
        """Plot confusion matrix as heatmap."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annot = []
        for i in range(cm.shape[0]):
            annot_row = []
            for j in range(cm.shape[1]):
                annot_row.append(f'{cm[i, j]}\\n({cm_percent[i, j]:.1f}%)')
            annot.append(annot_row)
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=['No Rise', 'Rise'],
                   yticklabels=['No Rise', 'Rise'])
        
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance_advanced(self, feature_importance_df: pd.DataFrame,
                                       top_n: int = 30, save_path: Optional[str] = None) -> None:
        """Plot advanced feature importance visualization."""
        
        top_features = feature_importance_df.head(top_n).copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Horizontal bar plot
        sns.barplot(data=top_features, y='feature', x='importance', 
                   palette='viridis', ax=ax1)
        ax1.set_title(f'Top {top_n} Feature Importance')
        ax1.set_xlabel('Importance Score')
        
        # Cumulative importance
        top_features['cumulative_importance'] = top_features['importance'].cumsum()
        total_importance = top_features['importance'].sum()
        top_features['cumulative_percentage'] = (top_features['cumulative_importance'] / 
                                                total_importance * 100)
        
        ax2.plot(range(len(top_features)), top_features['cumulative_percentage'], 
                marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
        ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   feature_importance_df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> None:
        """Create interactive Plotly dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Prediction Distribution', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC={pr_auc:.3f})',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Prediction Distribution
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_true == 0], name='Class 0', opacity=0.7,
                        histnorm='probability density'),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_true == 1], name='Class 1', opacity=0.7,
                        histnorm='probability density'),
            row=2, col=1
        )
        
        # Feature Importance
        top_features = feature_importance_df.head(20)
        fig.add_trace(
            go.Bar(y=top_features['feature'], x=top_features['importance'],
                  orientation='h', name='Feature Importance'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Model Evaluation Dashboard - {self.model_name}',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_xaxes(title_text="Importance", row=2, col=2)
        fig.update_yaxes(title_text="Feature", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 metrics: ModelPerformanceMetrics,
                                 save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        
        report = f"""
# Model Evaluation Report: {self.model_name}

## Dataset Summary
- Total Samples: {len(y_true):,}
- Positive Samples: {np.sum(y_true):,} ({np.mean(y_true)*100:.1f}%)
- Negative Samples: {len(y_true) - np.sum(y_true):,} ({(1-np.mean(y_true))*100:.1f}%)

## Classification Performance Metrics

### Basic Metrics
- **Accuracy**: {metrics.accuracy:.4f}
- **Precision**: {metrics.precision:.4f}
- **Recall (Sensitivity)**: {metrics.recall:.4f}
- **F1-Score**: {metrics.f1_score:.4f}
- **Specificity**: {metrics.specificity:.4f}

### Probabilistic Metrics
- **ROC-AUC**: {metrics.roc_auc:.4f}
- **PR-AUC**: {metrics.pr_auc:.4f}
- **Log Loss**: {metrics.log_loss:.4f}
- **Brier Score**: {metrics.brier_score:.4f}

### Advanced Metrics
- **Matthews Correlation Coefficient**: {metrics.matthews_corrcoef:.4f}
- **Cohen's Kappa**: {metrics.cohen_kappa:.4f}
- **Youden's Index**: {metrics.youden_index:.4f}

### Financial Metrics
- **Hit Rate**: {metrics.hit_rate:.4f}
- **False Alarm Rate**: {metrics.false_alarm_rate:.4f}
- **Profit Factor**: {metrics.profit_factor:.4f}
- **Sharpe Ratio**: {metrics.sharpe_ratio:.4f}

### Statistical Significance
- **McNemar Statistic**: {metrics.mcnemar_statistic:.4f}
- **McNemar p-value**: {metrics.mcnemar_pvalue:.4f}

## Model Interpretation

### Performance Assessment
"""

        # Performance interpretation
        if metrics.roc_auc >= 0.9:
            report += "- **Excellent**: ROC-AUC ≥ 0.90 indicates outstanding discriminative ability\\n"
        elif metrics.roc_auc >= 0.8:
            report += "- **Good**: ROC-AUC ≥ 0.80 indicates good discriminative ability\\n"
        elif metrics.roc_auc >= 0.7:
            report += "- **Fair**: ROC-AUC ≥ 0.70 indicates acceptable discriminative ability\\n"
        else:
            report += "- **Poor**: ROC-AUC < 0.70 indicates limited discriminative ability\\n"
        
        if metrics.matthews_corrcoef >= 0.5:
            report += "- **Strong Correlation**: MCC ≥ 0.50 indicates strong positive correlation\\n"
        elif metrics.matthews_corrcoef >= 0.3:
            report += "- **Moderate Correlation**: MCC ≥ 0.30 indicates moderate positive correlation\\n"
        else:
            report += "- **Weak Correlation**: MCC < 0.30 indicates weak correlation\\n"
        
        report += f"""
### Recommendations
1. **Threshold Selection**: Consider optimal threshold based on business requirements
2. **Model Calibration**: {"Model appears well-calibrated" if metrics.brier_score < 0.25 else "Consider calibration techniques"}
3. **Feature Engineering**: {"Current features are effective" if metrics.roc_auc > 0.75 else "Consider additional feature engineering"}
4. **Class Imbalance**: {"Balanced dataset" if 0.3 < np.mean(y_true) < 0.7 else "Consider resampling techniques"}

---
*Report generated automatically by AdvancedModelEvaluator*
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to: {save_path}")
        
        return report


def demonstrate_evaluation():
    """Demonstrate the evaluation capabilities."""
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate model predictions
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Simulate realistic predictions (better for positive class)
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Base probabilities
    y_pred_proba[y_true == 1] += np.random.normal(0.3, 0.2, np.sum(y_true))  # Boost positive class
    y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Ensure valid probabilities
    
    # Create feature importance data
    feature_names = [f'feature_{i:02d}' for i in range(50)]
    importance_scores = np.random.exponential(0.5, 50)
    importance_scores = importance_scores / importance_scores.sum()
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator("Demonstration Model")
    
    # Calculate comprehensive metrics
    metrics = evaluator.evaluate_comprehensive(y_true, y_pred_proba, threshold=0.5)
    
    # Generate all visualizations
    print("Generating evaluation visualizations...")
    
    evaluator.plot_roc_pr_curves(y_true, y_pred_proba)
    evaluator.plot_threshold_analysis(y_true, y_pred_proba)
    evaluator.plot_prediction_distribution(y_true, y_pred_proba)
    evaluator.plot_calibration_curve(y_true, y_pred_proba)
    evaluator.plot_confusion_matrix_heatmap(y_true, (y_pred_proba >= 0.5).astype(int))
    evaluator.plot_feature_importance_advanced(feature_importance_df)
    
    # Generate report
    report = evaluator.generate_evaluation_report(y_true, y_pred_proba, metrics)
    print(report)
    
    print("Evaluation demonstration completed!")


if __name__ == "__main__":
    demonstrate_evaluation()