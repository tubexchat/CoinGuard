import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 配置 ---
INPUT_CSV_FILE = "data/features_crypto_data.csv"
TARGET_LOOKAHEAD_HOURS = 6  # 向前看6小时
TARGET_DROP_PERCENT = -0.10 # 下跌超过10%

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个数据点创建目标变量 y。
    风险定义：未来6小时内，最低价相比当前收盤價下跌超过10%。
    """
    print(f"正在创建目标变量 (未来{TARGET_LOOKAHEAD_HOURS}小时内下跌超过 {-TARGET_DROP_PERCENT:.0%})...")
    
    # 为了找到未来6小时的最低点，我们向前移动6个时间步，然后计算这个窗口内的滚动最小值
    # shift(-TARGET_LOOKAHEAD_HOURS) 会将未来的数据向上移动
    # rolling(...) 会在移动后的数据上计算，从而得到原始数据“未来”的滚动值
    future_lows = df.groupby('symbol')['low'].transform(
        lambda x: x.shift(-TARGET_LOOKAHEAD_HOURS).rolling(window=TARGET_LOOKAHEAD_HOURS, min_periods=1).min()
    )
    
    # 计算未来最低价相比当前收盘价的变化率
    price_drop_ratio = (future_lows / df['close']) - 1
    
    # 如果变化率低于设定的阈值，则标记为1 (高风险)，否则为0
    df['target'] = (price_drop_ratio < TARGET_DROP_PERCENT).astype(int)
    
    return df

def main():
    """主函数，执行数据加载、目标变量创建、模型训练和评估。"""
    
    # 1. 加载特征数据
    print(f"正在从 {INPUT_CSV_FILE} 加载数据...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        df['open_time'] = pd.to_datetime(df['open_time'])
    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。请先运行 feature_engineering.py。")
        return

    # 2. 创建目标变量
    df = create_target_variable(df)

    # 3. 定义特征 (X) 和目标 (y)
    # 移除未来信息、标识符和可能导致数据泄漏的原始价格列
    features_to_drop = [
        'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'target'
    ]
    X = df.drop(columns=features_to_drop)
    y = df['target']
    
    # 清理可能存在的无穷大值
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) # 使用0填充NaN，也可以选择其他策略

    print("特征 (X) 和目标 (y) 已准备就绪。")
    print("特征数量:", len(X.columns))

    # 4. 类别平衡检查
    target_counts = y.value_counts()
    print("\n目标变量分布:")
    print(target_counts)
    if 1 not in target_counts or target_counts[1] == 0:
        print("警告: 数据中没有高风险样本，无法训练模型。请调整风险定义或使用更长的数据周期。")
        return
        
    # 计算类别权重，用于处理不平衡数据
    scale_pos_weight = target_counts[0] / target_counts[1]
    print(f"计算得到的 scale_pos_weight: {scale_pos_weight:.2f}")

    # 5. 时间序列数据划分
    # 绝不使用随机划分！必须按时间顺序。
    print("\n正在按时间顺序划分数据集...")
    # 使用总数据的70%作为训练集，15%作为验证集，15%作为测试集
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]
    X_test, y_test = X.iloc[train_size + val_size:], y.iloc[train_size + val_size:]

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    # 6. 训练 XGBoost 模型
    print("\n正在训练 XGBoost 模型...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=1000,          # 初始设置较大的树数量
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight, # 关键参数：处理类别不平衡
        random_state=42
    )

    # 使用验证集进行提前停止，防止过拟合
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("模型训练完成。")

    # 7. 在测试集上评估模型
    print("\n--- 模型性能评估 (测试集) ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n混淆矩阵:")
    # [[TN, FP],
    #  [FN, TP]]
    print(confusion_matrix(y_test, y_pred))
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['低风险 (0)', '高风险 (1)']))
    
    print(f"AUC-ROC 分数: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n**评估解读:**")
    print("关注 '高风险 (1)' 类别的 [Recall] (召回率) 指标。")
    print("它表示在所有真实发生的高风险事件中，我们的模型成功预测出了多少比例。对于风控系统，这个指标越高越好。")
    
    # 8. 显示特征重要性
    print("\n--- Top 20 特征重要性 ---")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    print(feature_importance)


if __name__ == "__main__":
    main()