"""
模型训练配置文件
包含所有训练相关的配置参数
"""

# 数据配置
DATA_CONFIG = {
    "input_csv_path": "../data/processed/features_crypto_data.csv",
    "model_output_path": "../data/models/",
    "feature_importance_plot_path": "../data/models/feature_importance.png"
}

# 目标变量配置
TARGET_CONFIG = {
    "lookahead_hours": 1,  # 向前看的小时数
    "drop_percent": -0.10  # 定义为风险的下跌百分比 (-10%)
}

# 数据分割配置
DATA_SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    # 测试集比例将是 1 - train_ratio - validation_ratio
}

# 模型基础参数
MODEL_BASE_PARAMS = {
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
}

# 超参数调优配置
TUNING_CONFIG = {
    "param_grid": {
        "n_estimators": [800, 1200, 1600, 2000],
        "learning_rate": [0.03, 0.05, 0.07],
        "max_depth": [4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0.0, 0.1, 0.2]
    },
    "scoring": "roc_auc",
    "max_combinations": 60
}

# 评估配置
EVALUATION_CONFIG = {
    "default_threshold": 0.6,
    "thresholds_to_test": [0.6]
}

# 合并所有配置
TRAINING_CONFIG = {
    "data": DATA_CONFIG,
    "target_variable": TARGET_CONFIG,
    "data_split": DATA_SPLIT_CONFIG,
    "model": {
        "params": MODEL_BASE_PARAMS,
        "early_stopping_rounds": 50
    },
    "tuning": TUNING_CONFIG,
    "evaluation": EVALUATION_CONFIG
}
