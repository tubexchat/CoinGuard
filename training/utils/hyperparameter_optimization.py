"""
Advanced Hyperparameter Optimization and Cross-Validation Module

This module provides sophisticated hyperparameter optimization using multiple
algorithms and robust cross-validation strategies for cryptocurrency prediction models.

Authors: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (
    TimeSeriesSplit, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV
)
from sklearn.metrics import make_scorer, roc_auc_score, log_loss, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization method
    method: str = "optuna"  # "optuna", "random_search", "grid_search", "bayesian"
    
    # Search space
    param_space: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization settings
    n_trials: int = 100
    n_jobs: int = -1
    random_state: int = 42
    
    # Cross-validation settings
    cv_method: str = "time_series"  # "time_series", "stratified", "blocked"
    n_splits: int = 5
    test_size: float = 0.2
    
    # Evaluation metrics
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = field(default_factory=lambda: ["log_loss", "f1_score"])
    
    # Early stopping
    early_stopping_rounds: int = 50
    patience: int = 10
    
    # Optuna specific
    optuna_sampler: str = "tpe"  # "tpe", "random", "cmaes"
    optuna_pruner: str = "median"  # "median", "successive_halving"
    
    def __post_init__(self):
        if not self.param_space:
            self.param_space = self._get_default_xgboost_space()
    
    def _get_default_xgboost_space(self) -> Dict[str, Any]:
        """Get default XGBoost parameter space."""
        return {
            'n_estimators': [500, 1000, 1500, 2000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],
            'min_child_weight': [1, 3, 5, 7]
        }


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    cv_results: Dict[str, List[float]]
    optimization_history: List[Dict[str, Any]]
    feature_importance: Optional[pd.DataFrame] = None
    
    # Statistical analysis
    score_mean: float = 0.0
    score_std: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Optimization metadata
    total_trials: int = 0
    best_trial_number: int = 0
    optimization_time: float = 0.0


class TimeSeriesBlockedCV:
    """Time series cross-validation with blocked structure to prevent data leakage."""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap: int = 0):
        """
        Initialize blocked time series CV.
        
        Args:
            n_splits: Number of CV splits
            test_size: Proportion of data for testing in each split
            gap: Number of samples to skip between train and test (to prevent leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """Generate train/test splits."""
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        
        # Calculate fold size
        fold_size = (n_samples - test_size_samples) // self.n_splits
        
        for i in range(self.n_splits):
            # Training data: from start to current fold end
            train_end = fold_size * (i + 1)
            train_indices = np.arange(0, train_end)
            
            # Test data: after gap, for test_size_samples
            test_start = train_end + self.gap
            test_end = min(test_start + test_size_samples, n_samples)
            test_indices = np.arange(test_start, test_end)
            
            # Skip if test set is too small
            if len(test_indices) < test_size_samples // 2:
                continue
                
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimization with multiple algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.study = None
        self.best_model = None
        self.optimization_results = None
        
    def optimize(self, X: pd.DataFrame, y: pd.Series,
                base_model_params: Optional[Dict] = None) -> OptimizationResult:
        """
        Perform hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            base_model_params: Base model parameters to extend
            
        Returns:
            OptimizationResult with best parameters and model
        """
        print(f"Starting hyperparameter optimization using {self.config.method}...")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        start_time = datetime.now()
        
        if base_model_params is None:
            base_model_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config.random_state,
                'n_jobs': 1  # Set to 1 for CV parallelization
            }
        
        # Choose optimization method
        if self.config.method == "optuna":
            result = self._optimize_with_optuna(X, y, base_model_params)
        elif self.config.method == "random_search":
            result = self._optimize_with_random_search(X, y, base_model_params)
        elif self.config.method == "grid_search":
            result = self._optimize_with_grid_search(X, y, base_model_params)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
        
        # Calculate optimization time
        end_time = datetime.now()
        result.optimization_time = (end_time - start_time).total_seconds()
        
        # Train final model with best parameters
        final_params = {**base_model_params, **result.best_params}
        result.best_model = self._train_final_model(X, y, final_params)
        
        # Calculate feature importance
        if hasattr(result.best_model, 'feature_importances_'):
            result.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': result.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.optimization_results = result
        
        print(f"Optimization completed in {result.optimization_time:.2f} seconds")
        print(f"Best {self.config.primary_metric}: {result.best_score:.4f}")
        print(f"Best parameters: {result.best_params}")
        
        return result
    
    def _get_cv_splitter(self):
        """Get cross-validation splitter based on configuration."""
        if self.config.cv_method == "time_series":
            return TimeSeriesSplit(n_splits=self.config.n_splits)
        elif self.config.cv_method == "blocked":
            return TimeSeriesBlockedCV(n_splits=self.config.n_splits, 
                                     test_size=self.config.test_size)
        elif self.config.cv_method == "stratified":
            return StratifiedKFold(n_splits=self.config.n_splits, 
                                 shuffle=True, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown CV method: {self.config.cv_method}")
    
    def _evaluate_model(self, params: Dict, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model with given parameters using cross-validation."""
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Get CV splitter
        cv_splitter = self._get_cv_splitter()
        
        # Calculate class weights
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        model.set_params(scale_pos_weight=scale_pos_weight)
        
        # Evaluate primary metric
        if self.config.primary_metric == "roc_auc":
            primary_scorer = make_scorer(roc_auc_score, needs_proba=True)
        elif self.config.primary_metric == "log_loss":
            primary_scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        elif self.config.primary_metric == "f1_score":
            primary_scorer = make_scorer(f1_score)
        else:
            raise ValueError(f"Unknown primary metric: {self.config.primary_metric}")
        
        # Perform cross-validation
        primary_scores = cross_val_score(
            model, X, y, cv=cv_splitter, scoring=primary_scorer,
            n_jobs=self.config.n_jobs
        )
        
        # Calculate secondary metrics
        secondary_scores = {}
        for metric in self.config.secondary_metrics:
            if metric == "roc_auc":
                scorer = make_scorer(roc_auc_score, needs_proba=True)
            elif metric == "log_loss":
                scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
            elif metric == "f1_score":
                scorer = make_scorer(f1_score)
            else:
                continue
            
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scorer)
            secondary_scores[metric] = np.mean(scores)
        
        return {
            self.config.primary_metric: np.mean(primary_scores),
            f"{self.config.primary_metric}_std": np.std(primary_scores),
            **secondary_scores
        }
    
    def _optimize_with_optuna(self, X: pd.DataFrame, y: pd.Series,
                            base_params: Dict) -> OptimizationResult:
        """Optimize using Optuna."""
        
        def objective(trial):
            # Sample parameters
            params = base_params.copy()
            
            for param_name, param_values in self.config.param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(param_name, 
                                                             min(param_values), 
                                                             max(param_values))
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name,
                                                               min(param_values),
                                                               max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, 
                                                             param_values[0], 
                                                             param_values[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name,
                                                               param_values[0],
                                                               param_values[1])
            
            # Evaluate model
            scores = self._evaluate_model(params, X, y)
            
            # Store additional metrics as user attributes
            for metric, score in scores.items():
                if metric != self.config.primary_metric:
                    trial.set_user_attr(metric, score)
            
            return scores[self.config.primary_metric]
        
        # Create sampler
        if self.config.optuna_sampler == "tpe":
            sampler = TPESampler(seed=self.config.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.random_state)
        
        # Create pruner
        if self.config.optuna_pruner == "median":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create and run study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        self.study.optimize(objective, n_trials=self.config.n_trials, n_jobs=1)
        
        # Extract results
        best_trial = self.study.best_trial
        
        # Get CV results for best parameters
        best_params = {k: v for k, v in best_trial.params.items()}
        final_params = {**base_params, **best_params}
        cv_scores = self._detailed_cv_evaluation(final_params, X, y)
        
        # Create optimization history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    **{k: v for k, v in trial.user_attrs.items()}
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_trial.value,
            best_model=None,  # Will be set later
            cv_results=cv_scores,
            optimization_history=history,
            total_trials=len(self.study.trials),
            best_trial_number=best_trial.number,
            score_mean=np.mean([t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            score_std=np.std([t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        )
    
    def _optimize_with_random_search(self, X: pd.DataFrame, y: pd.Series,
                                   base_params: Dict) -> OptimizationResult:
        """Optimize using RandomizedSearchCV."""
        
        # Prepare parameter distributions
        param_distributions = {}
        for param_name, param_values in self.config.param_space.items():
            if isinstance(param_values, list):
                param_distributions[param_name] = param_values
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # For range tuples, create list
                if isinstance(param_values[0], int):
                    param_distributions[param_name] = list(range(param_values[0], param_values[1] + 1))
                else:
                    param_distributions[param_name] = np.linspace(param_values[0], param_values[1], 20).tolist()
        
        # Create base model
        model = xgb.XGBClassifier(**base_params)
        
        # Setup CV
        cv_splitter = self._get_cv_splitter()
        
        # Create scorer
        if self.config.primary_metric == "roc_auc":
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        elif self.config.primary_metric == "log_loss":
            scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        else:
            scorer = make_scorer(f1_score)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            model, param_distributions,
            n_iter=self.config.n_trials,
            cv=cv_splitter,
            scoring=scorer,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        # Extract results
        best_params = random_search.best_params_
        cv_scores = self._detailed_cv_evaluation({**base_params, **best_params}, X, y)
        
        # Create history
        history = []
        for i, params in enumerate(random_search.cv_results_['params']):
            history.append({
                'trial_number': i,
                'params': params,
                'value': random_search.cv_results_['mean_test_score'][i]
            })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=random_search.best_score_,
            best_model=None,
            cv_results=cv_scores,
            optimization_history=history,
            total_trials=len(random_search.cv_results_['params']),
            best_trial_number=np.argmax(random_search.cv_results_['mean_test_score'])
        )
    
    def _optimize_with_grid_search(self, X: pd.DataFrame, y: pd.Series,
                                 base_params: Dict) -> OptimizationResult:
        """Optimize using GridSearchCV."""
        
        # Prepare parameter grid (reduce size for grid search)
        param_grid = {}
        for param_name, param_values in self.config.param_space.items():
            if isinstance(param_values, list):
                # Limit to first few values for grid search
                param_grid[param_name] = param_values[:3]
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # For range tuples, create small list
                if isinstance(param_values[0], int):
                    param_grid[param_name] = [param_values[0], 
                                            (param_values[0] + param_values[1]) // 2,
                                            param_values[1]]
                else:
                    param_grid[param_name] = [param_values[0],
                                            (param_values[0] + param_values[1]) / 2,
                                            param_values[1]]
        
        # Create base model
        model = xgb.XGBClassifier(**base_params)
        
        # Setup CV
        cv_splitter = self._get_cv_splitter()
        
        # Create scorer
        if self.config.primary_metric == "roc_auc":
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        elif self.config.primary_metric == "log_loss":
            scorer = make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        else:
            scorer = make_scorer(f1_score)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid,
            cv=cv_splitter,
            scoring=scorer,
            n_jobs=self.config.n_jobs,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        # Extract results
        best_params = grid_search.best_params_
        cv_scores = self._detailed_cv_evaluation({**base_params, **best_params}, X, y)
        
        # Create history
        history = []
        for i, params in enumerate(grid_search.cv_results_['params']):
            history.append({
                'trial_number': i,
                'params': params,
                'value': grid_search.cv_results_['mean_test_score'][i]
            })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=grid_search.best_score_,
            best_model=None,
            cv_results=cv_scores,
            optimization_history=history,
            total_trials=len(grid_search.cv_results_['params']),
            best_trial_number=np.argmax(grid_search.cv_results_['mean_test_score'])
        )
    
    def _detailed_cv_evaluation(self, params: Dict, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform detailed cross-validation evaluation."""
        
        model = xgb.XGBClassifier(**params)
        cv_splitter = self._get_cv_splitter()
        
        # Calculate class weights
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        model.set_params(scale_pos_weight=scale_pos_weight)
        
        results = {metric: [] for metric in [self.config.primary_metric] + self.config.secondary_metrics}
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            for metric in results.keys():
                if metric == "roc_auc":
                    score = roc_auc_score(y_val_fold, y_pred_proba)
                elif metric == "log_loss":
                    score = log_loss(y_val_fold, y_pred_proba)
                elif metric == "f1_score":
                    score = f1_score(y_val_fold, y_pred)
                else:
                    continue
                
                results[metric].append(score)
        
        return results
    
    def _train_final_model(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> xgb.XGBClassifier:
        """Train final model with best parameters."""
        
        # Calculate class weights
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        # Create and train model
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
        model.fit(X, y)
        
        return model
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history."""
        
        if self.optimization_results is None:
            print("No optimization results available. Run optimize() first.")
            return
        
        history = self.optimization_results.optimization_history
        
        if not history:
            print("No optimization history available.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Optimization progress
        trial_numbers = [h['trial_number'] for h in history]
        values = [h['value'] for h in history]
        
        ax1.plot(trial_numbers, values, 'b-', alpha=0.6)
        ax1.scatter(trial_numbers, values, c=values, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel(self.config.primary_metric.upper())
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Best score over time
        best_scores = []
        current_best = -np.inf
        for value in values:
            if value > current_best:
                current_best = value
            best_scores.append(current_best)
        
        ax2.plot(trial_numbers, best_scores, 'r-', linewidth=2)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel(f'Best {self.config.primary_metric.upper()}')
        ax2.set_title('Best Score Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Score distribution
        ax3.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(self.optimization_results.best_score, color='red', linestyle='--',
                   label=f'Best Score: {self.optimization_results.best_score:.4f}')
        ax3.set_xlabel(self.config.primary_metric.upper())
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Parameter importance (if using Optuna)
        if hasattr(self, 'study') and self.study is not None:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                if importance:
                    params = list(importance.keys())[:10]  # Top 10
                    importances = [importance[p] for p in params]
                    
                    ax4.barh(params, importances)
                    ax4.set_xlabel('Importance')
                    ax4.set_title('Parameter Importance')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Parameter importance\\nnot available', 
                            ha='center', va='center', transform=ax4.transAxes)
            except:
                ax4.text(0.5, 0.5, 'Parameter importance\\nnot available', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Parameter importance\\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_parameter_correlations(self, save_path: Optional[str] = None) -> None:
        """Plot correlations between parameters and performance."""
        
        if self.optimization_results is None:
            print("No optimization results available. Run optimize() first.")
            return
        
        history = self.optimization_results.optimization_history
        
        if not history:
            print("No optimization history available.")
            return
        
        # Create DataFrame
        data = []
        for h in history:
            row = h['params'].copy()
            row['score'] = h['value']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Not enough numeric parameters for correlation analysis.")
            return
        
        # Calculate correlations
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Parameter and Score Correlations')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter correlation plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results."""
        if self.optimization_results is None:
            print("No optimization results to save.")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.optimization_results, f)
        
        print(f"Optimization results saved to: {filepath}")
    
    def load_results(self, filepath: str) -> OptimizationResult:
        """Load optimization results."""
        with open(filepath, 'rb') as f:
            self.optimization_results = pickle.load(f)
        
        print(f"Optimization results loaded from: {filepath}")
        return self.optimization_results


def demonstrate_optimization():
    """Demonstrate hyperparameter optimization capabilities."""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i:02d}' for i in range(n_features)]
    )
    
    # Create realistic target with some signal
    signal = X['feature_00'] * 0.5 + X['feature_01'] * 0.3 - X['feature_02'] * 0.2
    y = pd.Series((signal + np.random.randn(n_samples) * 0.5 > 0).astype(int))
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Configure optimization
    config = OptimizationConfig(
        method="optuna",
        n_trials=50,  # Reduced for demo
        cv_method="time_series",
        n_splits=3,
        primary_metric="roc_auc"
    )
    
    # Initialize optimizer
    optimizer = AdvancedHyperparameterOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize(X, y)
    
    # Plot results
    optimizer.plot_optimization_history()
    optimizer.plot_parameter_correlations()
    
    print("\\nOptimization demonstration completed!")
    print(f"Best parameters: {results.best_params}")
    print(f"Best score: {results.best_score:.4f}")


if __name__ == "__main__":
    demonstrate_optimization()