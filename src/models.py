import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import xgboost as xgb
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class MLModelTrainer:
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
    
    def initialize_models(self, model_configs: Dict[str, Dict[str, Any]] = None):
        if model_configs is None:
            if self.task_type == 'classification':
                self.models = {
                    'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                    'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
                    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
                    'svm': SVC(random_state=42, probability=True),
                    'knn': KNeighborsClassifier(n_jobs=-1)
                }
            else:
                self.models = {
                    'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
                    'xgboost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                    'linear_regression': LinearRegression(n_jobs=-1),
                    'svr': SVR(),
                    'knn': KNeighborsRegressor(n_jobs=-1)
                }
        else:
            for name, config in model_configs.items():
                model_class = config['model']
                params = config.get('params', {})
                self.models[name] = model_class(**params)
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: np.ndarray):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray):
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            self.train_model(model_name, X_train, y_train)
        
        return self.trained_models
    
    def predict(self, model_name: str, X_test: pd.DataFrame):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        return self.trained_models[model_name].predict(X_test)
    
    def predict_proba(self, model_name: str, X_test: pd.DataFrame):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)
        else:
            raise AttributeError(f"Model {model_name} does not support probability predictions")
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: np.ndarray, 
                            cv: int = 5, scoring: Dict[str, str] = None):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if scoring is None:
            if self.task_type == 'classification':
                scoring = {
                    'accuracy': 'accuracy',
                    'f1_weighted': 'f1_weighted',
                    'precision_weighted': 'precision_weighted',
                    'recall_weighted': 'recall_weighted'
                }
            else:
                scoring = {
                    'neg_mse': 'neg_mean_squared_error',
                    'neg_mae': 'neg_mean_absolute_error',
                    'r2': 'r2'
                }
        
        model = self.models[model_name]
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                    return_train_score=True, n_jobs=-1)
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, 
                             y_train: np.ndarray, param_grid: Dict[str, list], 
                             search_type: str = 'grid', cv: int = 5, 
                             scoring: str = None, n_iter: int = 20):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        model = self.models[model_name]
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_jobs=-1, verbose=1, return_train_score=True
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, scoring=scoring, 
                n_iter=n_iter, n_jobs=-1, verbose=1, 
                random_state=42, return_train_score=True
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")
        
        search.fit(X_train, y_train)
        
        self.best_params[model_name] = search.best_params_
        self.trained_models[model_name] = search.best_estimator_
        
        return search
    
    def get_model(self, model_name: str):
        if model_name in self.trained_models:
            return self.trained_models[model_name]
        elif model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found")


class TextModelTrainer:
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.trained_models = {}
    
    def create_tfidf_vectorizer(self, max_features: int = 5000, 
                               ngram_range: Tuple[int, int] = (1, 2),
                               min_df: int = 2, max_df: float = 0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        return self.vectorizer
    
    def create_count_vectorizer(self, max_features: int = 5000,
                               ngram_range: Tuple[int, int] = (1, 2),
                               min_df: int = 2, max_df: float = 0.95):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        return self.vectorizer
    
    def fit_vectorizer(self, texts: pd.Series):
        if self.vectorizer is None:
            self.create_tfidf_vectorizer()
        
        return self.vectorizer.fit_transform(texts)
    
    def transform_texts(self, texts: pd.Series):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet. Call fit_vectorizer first.")
        
        return self.vectorizer.transform(texts)
    
    def initialize_text_classifiers(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'svm': SVC(kernel='linear', random_state=42, probability=True),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        }
        
        return self.models
    
    def train_model(self, model_name: str, X_train, y_train):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        return model
    
    def predict(self, model_name: str, X_test):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.trained_models[model_name].predict(X_test)
    
    def predict_proba(self, model_name: str, X_test):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)
        else:
            raise AttributeError(f"Model {model_name} does not support probability predictions")
    
    def get_feature_names(self):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet")
        
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features_per_class(self, model_name: str, class_labels: list, 
                                   top_n: int = 20) -> Dict[str, list]:
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        feature_names = self.get_feature_names()
        
        top_features = {}
        
        if hasattr(model, 'coef_'):
            for idx, label in enumerate(class_labels):
                if len(class_labels) == 2 and model.coef_.shape[0] == 1:
                    coef = model.coef_[0] if idx == 1 else -model.coef_[0]
                else:
                    coef = model.coef_[idx]
                
                top_indices = np.argsort(coef)[-top_n:][::-1]
                top_features[label] = [(feature_names[i], coef[i]) for i in top_indices]
        
        return top_features


def get_default_param_grids(model_name: str) -> Dict[str, list]:
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    }
    
    return param_grids.get(model_name, {})

