"""
模型训练和评估工具函数
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Any


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, 
                       X_val: pd.DataFrame, y_val: pd.Series, 
                       model_params: Dict) -> xgb.XGBClassifier:
    """训练XGBoost模型"""
    print("\n正在训练 XGBoost 模型...")
    target_counts = y_train.value_counts()
    if 1 not in target_counts or target_counts[1] == 0:
        print("警告: 训练集中没有高风险样本，无法计算权重。")
        scale_pos_weight = 1
    else:
        scale_pos_weight = target_counts[0] / target_counts[1]
        print(f"计算得到的 scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**model_params, scale_pos_weight=scale_pos_weight)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("模型训练完成。")
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, 
                  y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """在测试集上以指定阈值评估模型性能"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"\n--- 模型性能评估 (测试集, 预测阈值 = {threshold}) ---")
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n分类报告:")
    report = classification_report(y_test, y_pred, target_names=['低风险 (0)', '高风险 (1)'], digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['低风险 (0)', '高风险 (1)'], digits=4))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC 分数: {auc_score:.4f}")
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'auc_roc': auc_score,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def plot_feature_importance(model: xgb.XGBClassifier, features: pd.DataFrame, 
                          save_path: str = None, top_n: int = 20) -> None:
    """绘制特征重要性图表"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} 特征重要性', fontsize=16)
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图表已保存到: {save_path}")
    plt.show()


def save_model_and_metadata(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, 
                          config: Dict, model_name: str = None) -> Tuple[str, str]:
    """保存训练好的模型和相关元数据"""
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_model_{timestamp}"
    
    # 确保模型目录存在
    model_dir = config["data"]["model_output_path"]
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")
    
    # 保存特征列名
    feature_path = os.path.join(model_dir, f"{model_name}_features.pkl")
    with open(feature_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"特征列名已保存到: {feature_path}")
    
    # 保存配置信息
    config_path = os.path.join(model_dir, f"{model_name}_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"配置信息已保存到: {config_path}")
    
    # 保存训练统计信息
    stats = {
        'training_samples': len(X),
        'features_count': len(X.columns),
        'target_distribution': y.value_counts().to_dict(),
        'model_type': 'XGBoost',
        'training_date': datetime.now().isoformat()
    }
    stats_path = os.path.join(model_dir, f"{model_name}_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"训练统计信息已保存到: {stats_path}")
    
    return model_name, model_path


def load_model(model_path: str) -> xgb.XGBClassifier:
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_model_features(feature_path: str) -> list:
    """加载模型的特征列名"""
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    return features


def load_model_config(config_path: str) -> Dict:
    """加载模型配置"""
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def get_latest_model_info(model_dir: str) -> Dict[str, str]:
    """获取最新的模型文件信息"""
    if not os.path.exists(model_dir):
        return {}
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and not f.endswith('_features.pkl') and not f.endswith('_config.pkl') and not f.endswith('_stats.pkl')]
    
    if not model_files:
        return {}
    
    # 按修改时间排序，获取最新的模型
    model_files_with_time = [(f, os.path.getmtime(os.path.join(model_dir, f))) for f in model_files]
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_model = model_files_with_time[0][0]
    model_name = latest_model.replace('.pkl', '')
    
    return {
        'model_name': model_name,
        'model_path': os.path.join(model_dir, latest_model),
        'feature_path': os.path.join(model_dir, f"{model_name}_features.pkl"),
        'config_path': os.path.join(model_dir, f"{model_name}_config.pkl"),
        'stats_path': os.path.join(model_dir, f"{model_name}_stats.pkl")
    }
