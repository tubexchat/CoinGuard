import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 集中化配置
# ==============================================================================
CONFIG = {
    "data": {
        "input_csv_path": "data/features_crypto_data.csv",
    },
    "target_variable": {
        "lookahead_hours": 1, # 向前看的小时数
        "drop_percent": -0.10   # 定义为风险的下跌百分比 (-10%)
    },
    "data_split": {
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        # 测试集比例将是 1 - train_ratio - validation_ratio
    },
    "model": {
        "params": {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_estimators': 800,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.9,
            'gamma': 0.2,
            'random_state': 42
        },
        "early_stopping_rounds": 50,
    },
    "tuning": {
        # 每个参数的候选集合（可按需调整大小）
        "param_grid": {
            "n_estimators": [800, 1200, 1600, 2000],
            "learning_rate": [0.03, 0.05, 0.07],
            "max_depth": [4, 5, 6],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "gamma": [0.0, 0.1, 0.2]
        },
        # 评价指标：验证集 ROC AUC（越高越好）
        "scoring": "roc_auc",
        # 限制最大组合数量（>0 时启用前 N 个笛卡尔积组合，以控制开销）
        "max_combinations": 60
    },
    "evaluation": {
        "default_threshold": 0.6,
        "thresholds_to_test": [0.6] # 仅使用 0.6 作为预测阈值
    }
}


# ==============================================================================
# 2. 模块化函数
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """从CSV文件加载数据并进行初步处理。"""
    print(f"正在从 {filepath} 加载数据...")
    try:
        df = pd.read_csv(filepath)
        df['open_time'] = pd.to_datetime(df['open_time'])
        return df
    except FileNotFoundError:
        print(f"错误: 输入文件 '{filepath}' 未找到。请确保路径正确。")
        return None

def create_target_variable(df: pd.DataFrame, lookahead: int, drop_percent: float) -> pd.DataFrame:
    """为数据集创建目标变量 y。"""
    print(f"正在创建目标变量 (未来{lookahead}小时内下跌超过 {-drop_percent:.0%})...")
    
    if lookahead == 1:
        # 当只看未来1小时时，不需要滚动窗口，直接shift即可，效率更高
        future_lows = df.groupby('symbol')['low'].shift(-1)
    else:
        future_lows = df.groupby('symbol')['low'].transform(
            lambda x: x.shift(-lookahead).rolling(window=lookahead, min_periods=1).min()
        )
    
    price_drop_ratio = (future_lows / df['close']) - 1
    df['target'] = (price_drop_ratio < drop_percent).astype(int)
    return df

def prepare_data(df: pd.DataFrame):
    """准备特征矩阵X和目标向量y。"""
    features_to_drop = [
        'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'target'
    ]
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['target']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    print("特征 (X) 和目标 (y) 已准备就绪。")
    print(f"特征数量: {len(X.columns)}")
    return X, y

def split_data_temporal(X, y, train_ratio, val_ratio):
    """按时间顺序划分数据集。"""
    print("\n正在按时间顺序划分数据集...")
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]
    X_test, y_test = X.iloc[train_size + val_size:], y.iloc[train_size + val_size:]

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_xgboost_model(X_train, y_train, X_val, y_val, model_params):
    """训练XGBoost模型（为兼容旧版本XGBoost，关闭提前停止）。"""
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

def tune_hyperparameters(X_train, y_train, X_val, y_val, base_params: dict, tuning_cfg: dict):
    """在验证集上进行超参数搜索，返回最佳参数及评分。

    使用简单的网格搜索（受 max_combinations 限制）按验证 ROC AUC 选择最佳组合。
    """
    print("\n正在进行超参数搜索（基于验证集 ROC AUC）...")

    # 计算类别不平衡权重
    target_counts = y_train.value_counts()
    if 1 not in target_counts or target_counts[1] == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = target_counts[0] / target_counts[1]

    param_grid: dict = tuning_cfg.get("param_grid", {})
    keys = list(param_grid.keys())
    values_product = list(product(*[param_grid[k] for k in keys]))

    max_combinations = tuning_cfg.get("max_combinations", 0)
    if isinstance(max_combinations, int) and max_combinations > 0:
        values_product = values_product[:max_combinations]

    best_score = -float('inf')
    best_params = None

    for idx, values in enumerate(values_product, start=1):
        candidate = dict(zip(keys, values))
        params = {**base_params, **candidate}

        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_val_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_val_proba)
        print(f"候选 {idx}/{len(values_product)} => params={candidate} | val ROC AUC={score:.5f}")

        if score > best_score:
            best_score = score
            best_params = candidate

    print("超参数搜索完成。")
    return best_params, best_score

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """在测试集上以指定阈值评估模型性能。"""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"\n--- 模型性能评估 (测试集, 预测阈值 = {threshold}) ---")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['低风险 (0)', '高风险 (1)'], digits=4))
    
    print(f"AUC-ROC 分数: {roc_auc_score(y_test, y_pred_proba):.4f}")

def plot_feature_importance(model, features):
    """绘制特征重要性图表。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis', ax=ax)
    ax.set_title('Top 20 特征重要性', fontsize=16)
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. 主执行流程
# ==============================================================================

def main():
    """主函数，执行整个流程。"""
    df = load_data(CONFIG["data"]["input_csv_path"])
    if df is None:
        return
        
    df = create_target_variable(
        df, 
        CONFIG["target_variable"]["lookahead_hours"], 
        CONFIG["target_variable"]["drop_percent"]
    )
    
    X, y = prepare_data(df)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
        X, y, 
        CONFIG["data_split"]["train_ratio"],
        CONFIG["data_split"]["validation_ratio"]
    )
    
    # 先进行超参数搜索
    best_params, best_score = tune_hyperparameters(
        X_train, y_train, X_val, y_val,
        CONFIG["model"]["params"],
        CONFIG["tuning"]
    )

    print("\n=== 超参数搜索结果 ===")
    print(f"最佳验证 ROC AUC: {best_score:.5f}")
    print(f"最佳参数: {best_params}")

    # 用最佳参数重新训练（与基础参数合并）
    final_params = {**CONFIG["model"]["params"], **(best_params or {})}
    model = train_xgboost_model(X_train, y_train, X_val, y_val, final_params)
    
    # 使用多个不同阈值评估模型
    for threshold in CONFIG["evaluation"]["thresholds_to_test"]:
        evaluate_model(model, X_test, y_test, threshold=threshold)
        
    print("\n**评估解读:**")
    print("关注 '高风险 (1)' 类别的 [Recall] (召回率) 和 [Precision] (精确率)。")
    print("降低阈值通常会提高召回率（抓住更多风险）但降低精确率（更多虚假警报），反之亦然。")
    print("请根据您的风险偏好，选择最佳的阈值平衡点。")
    
    plot_feature_importance(model, X)


if __name__ == "__main__":
    main()
