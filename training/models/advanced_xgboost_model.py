"""
Advanced XGBoost Model for Cryptocurrency Price Prediction

This module implements a sophisticated XGBoost-based ensemble model for predicting
cryptocurrency price movements with enhanced feature engineering, advanced evaluation
metrics, and comprehensive model analysis suitable for academic research.

Authors: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, log_loss, matthews_corrcoef,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import joblib
from itertools import product
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import scipy.stats as stats
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration class for model parameters and training settings."""
    
    # Data configuration
    input_csv_path: str = "../data/features_crypto_data.csv"
    model_output_path: str = "../data/models/"
    
    # Target variable configuration
    lookahead_hours: int = 6
    prediction_threshold: float = 0.001  # Minimum price change to consider significant
    
    # Data split configuration
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    use_time_series_split: bool = True
    n_splits: int = 5
    
    # Model parameters
    model_params: Dict = None
    
    # Feature engineering
    use_feature_selection: bool = True
    feature_selection_k: int = 50
    feature_selection_method: str = "mutual_info"  # "f_classif", "mutual_info"
    
    # Evaluation
    evaluation_thresholds: List[float] = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'use_label_encoder': False,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'tree_method': 'hist',
                'grow_policy': 'depthwise'
            }
        
        if self.evaluation_thresholds is None:
            self.evaluation_thresholds = [0.4, 0.5, 0.6, 0.7]


class AdvancedFeatureEngineering:
    """Advanced feature engineering for cryptocurrency time series data."""
    
    @staticmethod
    def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features."""
        print("Creating microstructure features...")
        
        # Price impact measures
        df['spread'] = df['high'] - df['low']
        df['spread_ratio'] = df['spread'] / df['close']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['spread']
        
        # Volume-price relationships
        df['vwap'] = (df['volume'] * df['close']).rolling(window=24).sum() / df['volume'].rolling(window=24).sum()
        df['volume_price_trend'] = df['close'] / df['vwap']
        df['volume_oscillator'] = (df['volume'].rolling(12).mean() - df['volume'].rolling(26).mean()) / df['volume'].rolling(26).mean()
        
        return df
    
    @staticmethod
    def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced volatility features."""
        print("Creating volatility features...")
        
        # Realized volatility measures
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        for window in [6, 12, 24, 48]:
            # Realized volatility
            df[f'realized_vol_{window}h'] = df['log_return'].rolling(window).std() * np.sqrt(window)
            
            # Garman-Klass volatility estimator
            df[f'gk_vol_{window}h'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low']) ** 2).rolling(window).mean() -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2).rolling(window).mean()
            )
            
            # Parkinson volatility
            df[f'parkinson_vol_{window}h'] = np.sqrt(
                (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(window).mean()
            )
        
        return df
    
    @staticmethod
    def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features."""
        print("Creating momentum features...")
        
        # Multiple timeframe momentum
        for period in [6, 12, 24, 48, 72]:
            df[f'momentum_{period}h'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}h'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Acceleration features
        df['momentum_acceleration_12h'] = df['momentum_12h'] - df['momentum_12h'].shift(6)
        df['momentum_acceleration_24h'] = df['momentum_24h'] - df['momentum_24h'].shift(12)
        
        # Trend strength
        for window in [12, 24, 48]:
            df[f'trend_strength_{window}h'] = abs(
                df['close'].rolling(window).corr(pd.Series(range(window)))
            )
        
        return df
    
    @staticmethod
    def create_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime identification features."""
        print("Creating regime features...")
        
        # Volatility regimes
        df['vol_regime'] = pd.qcut(df['realized_vol_24h'], q=3, labels=[0, 1, 2])
        
        # Trend regimes based on multiple moving averages
        df['sma_12'] = df['close'].rolling(12).mean()
        df['sma_24'] = df['close'].rolling(24).mean()
        df['sma_48'] = df['close'].rolling(48).mean()
        
        df['trend_regime'] = 0
        df.loc[(df['close'] > df['sma_12']) & (df['sma_12'] > df['sma_24']), 'trend_regime'] = 1
        df.loc[(df['close'] < df['sma_12']) & (df['sma_12'] < df['sma_24']), 'trend_regime'] = -1
        
        return df


class ModelEvaluator:
    """Comprehensive model evaluation with academic-grade metrics."""
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        }
        
        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = np.trapz(precision, recall)
        
        # Calculate Sharpe-like ratio for classification
        precision_array = np.array([precision_score(y_true, (y_pred_proba >= t).astype(int)) 
                                  for t in np.linspace(0.1, 0.9, 9)])
        recall_array = np.array([recall_score(y_true, (y_pred_proba >= t).astype(int)) 
                               for t in np.linspace(0.1, 0.9, 9)])
        
        metrics['precision_stability'] = 1 / (np.std(precision_array) + 1e-8)
        metrics['recall_stability'] = 1 / (np.std(recall_array) + 1e-8)
        
        return metrics
    
    @staticmethod
    def plot_comprehensive_evaluation(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                    save_path: str = None) -> None:
        """Create comprehensive evaluation plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Prediction Distribution
        axes[0, 2].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Class 0', density=True)
        axes[0, 2].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Class 1', density=True)
        axes[0, 2].set_xlabel('Prediction Probability')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Prediction Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Threshold Analysis
        thresholds = np.linspace(0.1, 0.9, 50)
        precision_scores = [precision_score(y_true, (y_pred_proba >= t).astype(int)) for t in thresholds]
        recall_scores = [recall_score(y_true, (y_pred_proba >= t).astype(int)) for t in thresholds]
        f1_scores = [f1_score(y_true, (y_pred_proba >= t).astype(int)) for t in thresholds]
        
        axes[1, 0].plot(thresholds, precision_scores, label='Precision')
        axes[1, 0].plot(thresholds, recall_scores, label='Recall')
        axes[1, 0].plot(thresholds, f1_scores, label='F1-Score')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Threshold Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confusion Matrix Heatmap (for threshold = 0.5)
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_title('Confusion Matrix (threshold=0.5)')
        
        # Calibration Plot
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        axes[1, 2].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        axes[1, 2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 2].set_xlabel('Mean Predicted Probability')
        axes[1, 2].set_ylabel('Fraction of Positives')
        axes[1, 2].set_title('Calibration Plot')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {save_path}")
        
        plt.show()


class AdvancedXGBoostModel:
    """Advanced XGBoost model with comprehensive evaluation and analysis."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_selector = None
        self.scaler = None
        self.feature_names = None
        self.feature_importance_df = None
        self.evaluation_results = {}
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data with enhanced feature engineering."""
        print(f"Loading data from {self.config.input_csv_path}...")
        
        try:
            df = pd.read_csv(self.config.input_csv_path)
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.sort_values(['symbol', 'open_time']).reset_index(drop=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{self.config.input_csv_path}' not found.")
        
        print("Applying advanced feature engineering...")
        
        # Group by symbol and apply feature engineering
        enhanced_dfs = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Apply feature engineering
            symbol_df = AdvancedFeatureEngineering.create_microstructure_features(symbol_df)
            symbol_df = AdvancedFeatureEngineering.create_volatility_features(symbol_df)
            symbol_df = AdvancedFeatureEngineering.create_momentum_features(symbol_df)
            symbol_df = AdvancedFeatureEngineering.create_regime_features(symbol_df)
            
            enhanced_dfs.append(symbol_df)
        
        df = pd.concat(enhanced_dfs, ignore_index=True)
        
        # Create target variable
        df = self._create_target_variable(df)
        
        # Prepare features
        X, y = self._prepare_features(df)
        
        return X, y
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for price direction prediction."""
        print(f"Creating target variable: future {self.config.lookahead_hours}h price direction...")
        
        # Calculate future price change
        future_close = df.groupby('symbol')['close'].shift(-self.config.lookahead_hours)
        price_change = (future_close - df['close']) / df['close']
        
        # Binary classification: 1 if price increases by more than threshold, 0 otherwise
        df['target'] = (price_change > self.config.prediction_threshold).astype(int)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector."""
        
        # Define features to exclude
        exclude_features = [
            'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'target', 'sma_12', 'sma_24', 'sma_48', 'vwap'
        ]
        
        # Create feature matrix
        feature_columns = [col for col in df.columns if col not in exclude_features]
        X = df[feature_columns].copy()
        y = df['target'].copy()
        
        # Handle infinite values and missing data
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(feature_columns)}")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform feature selection."""
        if not self.config.use_feature_selection:
            return X
        
        print(f"Performing feature selection using {self.config.feature_selection_method}...")
        
        if self.config.feature_selection_method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=self.config.feature_selection_k)
        else:  # mutual_info
            selector = SelectKBest(score_func=mutual_info_classif, k=self.config.feature_selection_k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        
        self.feature_selector = selector
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        print("Performing time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        cv_results = {
            'train_auc': [], 'val_auc': [], 'train_logloss': [], 'val_logloss': [],
            'precision': [], 'recall': [], 'f1': [], 'matthews_corrcoef': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold + 1}/{self.config.n_splits}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate class weights
            class_weights = self._calculate_class_weights(y_train_fold)
            
            # Train model
            model = xgb.XGBClassifier(
                **self.config.model_params,
                scale_pos_weight=class_weights
            )
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                eval_metric=['logloss', 'auc'],
                verbose=False
            )
            
            # Predictions
            y_train_pred_proba = model.predict_proba(X_train_fold)[:, 1]
            y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            cv_results['train_auc'].append(roc_auc_score(y_train_fold, y_train_pred_proba))
            cv_results['val_auc'].append(roc_auc_score(y_val_fold, y_val_pred_proba))
            cv_results['train_logloss'].append(log_loss(y_train_fold, y_train_pred_proba))
            cv_results['val_logloss'].append(log_loss(y_val_fold, y_val_pred_proba))
            cv_results['precision'].append(precision_score(y_val_fold, y_val_pred))
            cv_results['recall'].append(recall_score(y_val_fold, y_val_pred))
            cv_results['f1'].append(f1_score(y_val_fold, y_val_pred))
            cv_results['matthews_corrcoef'].append(matthews_corrcoef(y_val_fold, y_val_pred))
        
        # Print CV results
        print("\nCross-Validation Results:")
        for metric, values in cv_results.items():
            print(f"{metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")
        
        return cv_results
    
    def _calculate_class_weights(self, y: pd.Series) -> float:
        """Calculate class weights for imbalanced data."""
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return 1.0
        return class_counts[0] / class_counts[1]
    
    def split_data_temporal(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data temporally."""
        print("Splitting data temporally...")
        
        train_size = int(len(X) * self.config.train_ratio)
        val_size = int(len(X) * self.config.validation_ratio)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        
        print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Train the XGBoost model."""
        print("Training XGBoost model...")
        
        # Calculate class weights
        scale_pos_weight = self._calculate_class_weights(y_train)
        print(f"Using scale_pos_weight: {scale_pos_weight:.3f}")
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            **self.config.model_params,
            scale_pos_weight=scale_pos_weight
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=['logloss', 'auc'],
            verbose=False
        )
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create feature importance dataframe
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training completed.")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        print("Evaluating model...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {}
        
        # Evaluate at different thresholds
        for threshold in self.config.evaluation_thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = ModelEvaluator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )
            results[f"threshold_{threshold}"] = metrics
            
            print(f"\nThreshold {threshold}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        self.evaluation_results = results
        
        return results
    
    def plot_feature_importance(self, top_n: int = 30, save_path: str = None) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(12, 8))
        
        top_features = self.feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str = None) -> str:
        """Save the trained model and metadata."""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"advanced_xgboost_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs(self.config.model_output_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.config.model_output_path, f"{model_name}.pkl")
        joblib.dump(self.model, model_path)
        
        # Save feature names
        feature_path = os.path.join(self.config.model_output_path, f"{model_name}_features.pkl")
        joblib.dump(self.feature_names, feature_path)
        
        # Save configuration
        config_path = os.path.join(self.config.model_output_path, f"{model_name}_config.pkl")
        joblib.dump(self.config, config_path)
        
        # Save evaluation results
        eval_path = os.path.join(self.config.model_output_path, f"{model_name}_evaluation.pkl")
        joblib.dump(self.evaluation_results, eval_path)
        
        # Save feature importance
        importance_path = os.path.join(self.config.model_output_path, f"{model_name}_feature_importance.csv")
        self.feature_importance_df.to_csv(importance_path, index=False)
        
        print(f"Model saved as: {model_name}")
        print(f"Model files saved to: {self.config.model_output_path}")
        
        return model_name


def main():
    """Main training pipeline."""
    
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize model
    model = AdvancedXGBoostModel(config)
    
    # Load and prepare data
    X, y = model.load_and_prepare_data()
    
    # Feature selection
    X = model.feature_selection(X, y)
    
    # Perform cross-validation
    cv_results = model.time_series_cross_validation(X, y)
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = model.split_data_temporal(X, y)
    
    # Train model
    model.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluation_results = model.evaluate_model(X_test, y_test)
    
    # Plot feature importance
    importance_plot_path = os.path.join(config.model_output_path, "advanced_feature_importance.png")
    model.plot_feature_importance(save_path=importance_plot_path)
    
    # Create comprehensive evaluation plots
    y_test_pred_proba = model.model.predict_proba(X_test)[:, 1]
    eval_plot_path = os.path.join(config.model_output_path, "comprehensive_evaluation.png")
    ModelEvaluator.plot_comprehensive_evaluation(y_test, y_test_pred_proba, save_path=eval_plot_path)
    
    # Save model
    model_name = model.save_model()
    
    print(f"\n{'='*60}")
    print("ADVANCED MODEL TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Cross-validation AUC: {np.mean(cv_results['val_auc']):.4f} (±{np.std(cv_results['val_auc']):.4f})")
    print(f"Test AUC: {evaluation_results['threshold_0.5']['roc_auc']:.4f}")
    print(f"Feature importance and evaluation plots saved.")


if __name__ == "__main__":
    main()