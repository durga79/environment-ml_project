import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error,
    r2_score, matthews_corrcoef
)
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.metrics_history = []
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray = None,
                                average: str = 'weighted') -> Dict[str, Any]:
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:
                if y_pred_proba.ndim == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                try:
                    metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                           multi_class='ovr', average='weighted')
                except:
                    metrics['auc_roc_ovr'] = None
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        self.metrics_history.append({
            'type': 'classification',
            'metrics': metrics
        })
        
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['mape'] = mape
        
        rss = np.sum((y_true - y_pred) ** 2)
        metrics['rss'] = rss
        
        self.metrics_history.append({
            'type': 'regression',
            'metrics': metrics
        })
        
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    target_names: list = None):
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             labels: list = None, figsize: Tuple[int, int] = (8, 6),
                             normalize: bool = False):
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      figsize: Tuple[int, int] = (8, 6)):
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only available for binary classification")
            return None
        
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        comparison_df = pd.DataFrame(results_dict).T
        
        if self.task_type == 'classification':
            cols_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
        else:
            cols_to_show = ['rmse', 'mae', 'mape', 'r2']
        
        available_cols = [col for col in cols_to_show if col in comparison_df.columns]
        
        return comparison_df[available_cols].round(4)
    
    def plot_metrics_comparison(self, results_dict: Dict[str, Dict], 
                               metrics_to_plot: list = None,
                               figsize: Tuple[int, int] = (12, 6)):
        if metrics_to_plot is None:
            if self.task_type == 'classification':
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            else:
                metrics_to_plot = ['rmse', 'mae', 'r2']
        
        df = pd.DataFrame(results_dict).T
        
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not available_metrics:
            print("No metrics available to plot")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        df[available_metrics].plot(kind='bar', ax=ax)
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig


class CrossValidationEvaluator:
    
    def __init__(self):
        self.cv_results = {}
    
    def evaluate_cv_results(self, cv_scores: Dict[str, np.ndarray], 
                           model_name: str) -> Dict[str, Any]:
        results = {}
        
        for metric_name, scores in cv_scores.items():
            results[f'{metric_name}_mean'] = np.mean(scores)
            results[f'{metric_name}_std'] = np.std(scores)
            results[f'{metric_name}_min'] = np.min(scores)
            results[f'{metric_name}_max'] = np.max(scores)
        
        self.cv_results[model_name] = results
        
        return results
    
    def print_cv_summary(self, cv_scores: Dict[str, np.ndarray], model_name: str):
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results for {model_name}")
        print(f"{'='*60}")
        
        for metric_name, scores in cv_scores.items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")
            print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
    
    def plot_cv_scores(self, cv_scores: Dict[str, np.ndarray], 
                      model_name: str, figsize: Tuple[int, int] = (10, 6)):
        metrics = list(cv_scores.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, scores) in enumerate(cv_scores.items()):
            ax = axes[idx]
            ax.boxplot([scores], labels=[metric_name])
            ax.scatter([1] * len(scores), scores, alpha=0.6, color='red')
            ax.set_ylabel('Score')
            ax.set_title(f'{metric_name}\n(Mean: {np.mean(scores):.3f})')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Cross-Validation Scores: {model_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig

