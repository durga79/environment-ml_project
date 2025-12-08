import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Tuple
import shap
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer


class SHAPInterpreter:
    
    def __init__(self, model: Any, X_train: pd.DataFrame, model_type: str = 'tree'):
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_train)
        elif model_type == 'kernel':
            self.explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        else:
            self.explainer = shap.Explainer(model, X_train)
    
    def calculate_shap_values(self, X_test: pd.DataFrame):
        self.shap_values = self.explainer.shap_values(X_test)
        return self.shap_values
    
    def plot_summary(self, X_test: pd.DataFrame, plot_type: str = 'dot', 
                     max_display: int = 20, figsize: Tuple[int, int] = (10, 8)):
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        plt.figure(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[1], X_test, plot_type=plot_type, 
                            max_display=max_display, show=False)
        else:
            shap.summary_plot(self.shap_values, X_test, plot_type=plot_type, 
                            max_display=max_display, show=False)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_bar(self, X_test: pd.DataFrame, max_display: int = 20, 
                 figsize: Tuple[int, int] = (10, 8)):
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        plt.figure(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[1], X_test, plot_type='bar', 
                            max_display=max_display, show=False)
        else:
            shap.summary_plot(self.shap_values, X_test, plot_type='bar', 
                            max_display=max_display, show=False)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waterfall(self, X_test: pd.DataFrame, sample_idx: int = 0, 
                       figsize: Tuple[int, int] = (10, 8)):
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        plt.figure(figsize=figsize)
        
        if isinstance(self.shap_values, list):
            shap_values_exp = self.explainer(X_test)
            shap.plots.waterfall(shap_values_exp[sample_idx], show=False)
        else:
            shap_values_exp = self.explainer(X_test)
            shap.plots.waterfall(shap_values_exp[sample_idx], show=False)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_force(self, X_test: pd.DataFrame, sample_idx: int = 0):
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if isinstance(self.shap_values, list):
            return shap.force_plot(self.explainer.expected_value[1], 
                                  self.shap_values[1][sample_idx], 
                                  X_test.iloc[sample_idx])
        else:
            return shap.force_plot(self.explainer.expected_value, 
                                  self.shap_values[sample_idx], 
                                  X_test.iloc[sample_idx])
    
    def get_feature_importance(self, X_test: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if isinstance(self.shap_values, list):
            shap_abs = np.abs(self.shap_values[1]).mean(axis=0)
        else:
            shap_abs = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': shap_abs
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class LIMEInterpreter:
    
    def __init__(self, X_train: pd.DataFrame, mode: str = 'classification'):
        self.X_train = X_train
        self.mode = mode
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            mode=mode,
            random_state=42
        )
    
    def explain_instance(self, model: Any, instance: np.ndarray, 
                        num_features: int = 10, top_labels: int = 1):
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba if self.mode == 'classification' else model.predict,
            num_features=num_features,
            top_labels=top_labels
        )
        
        return explanation
    
    def plot_explanation(self, explanation, label: int = 1, 
                        figsize: Tuple[int, int] = (10, 6)):
        fig = explanation.as_pyplot_figure(label=label)
        fig.set_size_inches(figsize)
        plt.tight_layout()
        return fig
    
    def get_explanation_dict(self, explanation, label: int = 1) -> dict:
        return dict(explanation.as_list(label=label))


class LIMETextInterpreter:
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names, random_state=42)
    
    def explain_instance(self, text: str, predict_fn: Any, num_features: int = 10, 
                        top_labels: int = 1):
        explanation = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=num_features,
            top_labels=top_labels
        )
        
        return explanation
    
    def show_in_notebook(self, explanation, label: int = 1):
        return explanation.show_in_notebook(text=True, labels=(label,))
    
    def get_explanation_dict(self, explanation, label: int = 1) -> dict:
        return dict(explanation.as_list(label=label))
    
    def plot_explanation(self, explanation, label: int = 1, 
                        figsize: Tuple[int, int] = (10, 6)):
        fig = explanation.as_pyplot_figure(label=label)
        fig.set_size_inches(figsize)
        plt.tight_layout()
        return fig


class FeatureImportanceAnalyzer:
    
    @staticmethod
    def get_tree_importance(model: Any, feature_names: List[str], 
                           top_n: int = 20) -> pd.DataFrame:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise AttributeError("Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, 
                               title: str = 'Feature Importance',
                               figsize: Tuple[int, int] = (10, 8)):
        plt.figure(figsize=figsize)
        
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def compare_importances(importance_dicts: dict, top_n: int = 15, 
                           figsize: Tuple[int, int] = (14, 8)):
        fig, axes = plt.subplots(1, len(importance_dicts), figsize=figsize)
        
        if len(importance_dicts) == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(importance_dicts.items()):
            ax = axes[idx]
            top_features = importance_df.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature', 
                       ax=ax, palette='viridis')
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=10)
            ax.set_ylabel('Features', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig


class PermutationImportance:
    
    @staticmethod
    def calculate(model: Any, X: pd.DataFrame, y: np.ndarray, 
                 metric_fn: Any, n_repeats: int = 10, 
                 random_state: int = 42) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y, 
            scoring=metric_fn,
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    @staticmethod
    def plot(importance_df: pd.DataFrame, top_n: int = 20, 
            figsize: Tuple[int, int] = (10, 8)):
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'], align='center', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Permutation Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

