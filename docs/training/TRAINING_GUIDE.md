# CoinGuard 训练指南

## 概述

本指南详细介绍 CoinGuard 项目的模型训练流程，包括数据准备、特征工程、模型训练、超参数调优和模型评估。

## 训练流程概览

```
原始数据 → 特征工程 → 数据分割 → 模型训练 → 超参数调优 → 模型评估 → 模型保存
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 目录结构

确保项目目录结构正确：
```
training/
├── train_model.py          # 主训练脚本
├── configs/
│   └── model_config.py     # 配置文件
└── utils/
    ├── data_utils.py       # 数据处理工具
    └── model_utils.py      # 模型工具
```

## 数据准备

### 1. 下载原始数据

```bash
cd data/raw
python download.py
```

这将下载加密货币的K线数据和多空比数据，保存为 `crypto_klines_data.csv`。

### 2. 特征工程

```bash
cd data/processed
python feature_engineering.py
```

这将生成包含所有特征的 `features_crypto_data.csv` 文件。

## 配置说明

### 模型配置 (configs/model_config.py)

```python
# 数据配置
DATA_CONFIG = {
    "input_csv_path": "../data/processed/features_crypto_data.csv",
    "model_output_path": "../data/models/",
}

# 目标变量配置
TARGET_CONFIG = {
    "lookahead_hours": 1,      # 预测未来1小时
    "drop_percent": -0.10      # 下跌超过10%定义为高风险
}

# 数据分割配置
DATA_SPLIT_CONFIG = {
    "train_ratio": 0.7,        # 70% 训练集
    "validation_ratio": 0.15,  # 15% 验证集
    # 15% 测试集
}

# 模型参数
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
```

## 训练步骤

### 1. 启动训练

```bash
cd training
python train_model.py
```

### 2. 训练过程说明

训练过程包括以下步骤：

#### 2.1 数据加载和验证
- 从CSV文件加载特征数据
- 验证数据质量和完整性
- 显示数据统计信息

#### 2.2 目标变量创建
- 根据配置创建目标变量
- 计算未来价格下跌情况
- 生成二分类标签（0=低风险，1=高风险）

#### 2.3 数据分割
- 按时间顺序分割数据
- 确保训练集、验证集、测试集的时间顺序
- 避免数据泄露

#### 2.4 超参数搜索
- 使用网格搜索进行超参数调优
- 基于验证集ROC AUC选择最佳参数
- 支持限制搜索组合数量以控制计算时间

#### 2.5 模型训练
- 使用最佳参数训练最终模型
- 自动计算类别不平衡权重
- 支持早停机制

#### 2.6 模型评估
- 在测试集上评估模型性能
- 计算混淆矩阵、分类报告、AUC-ROC
- 生成特征重要性图表

#### 2.7 模型保存
- 保存训练好的模型（PKL格式）
- 保存特征列名和配置信息
- 保存训练统计信息

## 输出文件

训练完成后，在 `data/models/` 目录下会生成以下文件：

```
data/models/
├── xgboost_model_20231201_100000.pkl      # 训练好的模型
├── xgboost_model_20231201_100000_features.pkl  # 特征列名
├── xgboost_model_20231201_100000_config.pkl    # 训练配置
├── xgboost_model_20231201_100000_stats.pkl     # 训练统计
└── feature_importance.png                       # 特征重要性图表
```

## 性能调优

### 1. 超参数调优

在 `configs/model_config.py` 中调整超参数搜索空间：

```python
TUNING_CONFIG = {
    "param_grid": {
        "n_estimators": [800, 1200, 1600, 2000],
        "learning_rate": [0.03, 0.05, 0.07],
        "max_depth": [4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0.0, 0.1, 0.2]
    },
    "max_combinations": 60  # 限制搜索组合数量
}
```

### 2. 计算资源优化

- **快速迭代**: 设置较小的 `max_combinations` (如 20-30)
- **生产训练**: 使用较大的 `max_combinations` (如 100-200)
- **内存优化**: 减少 `n_estimators` 或增加 `learning_rate`
- **并行处理**: XGBoost 自动使用多核CPU

### 3. 数据质量优化

- 确保数据时间顺序正确
- 检查特征工程是否完整
- 验证目标变量分布是否合理
- 处理缺失值和异常值

## 模型评估指标

### 1. 主要指标

- **AUC-ROC**: 模型整体性能
- **精确率 (Precision)**: 高风险预测的准确性
- **召回率 (Recall)**: 高风险样本的识别率
- **F1分数**: 精确率和召回率的调和平均

### 2. 混淆矩阵解读

```
               预测
实际    低风险    高风险
低风险    TN       FP
高风险    FN       TP
```

- **TN (True Negative)**: 正确预测的低风险
- **FP (False Positive)**: 错误预测的高风险（假警报）
- **FN (False Negative)**: 错误预测的低风险（漏报）
- **TP (True Positive)**: 正确预测的高风险

### 3. 阈值调优

通过调整预测阈值来平衡精确率和召回率：

- **高阈值 (0.7-0.9)**: 高精确率，低召回率
- **中阈值 (0.5-0.7)**: 平衡精确率和召回率
- **低阈值 (0.3-0.5)**: 低精确率，高召回率

## 特征重要性分析

训练完成后会生成特征重要性图表，显示最重要的20个特征：

### 常见重要特征

1. **技术指标**: RSI, MACD, 布林带相关指标
2. **价格特征**: 收益率、价格变化、价格范围
3. **成交量特征**: 成交量变化、成交量比率
4. **时间特征**: 滞后特征、滚动统计特征
5. **市场情绪**: 多空比、持仓比

### 特征重要性解读

- **高重要性特征**: 对模型预测贡献大，需要重点关注
- **低重要性特征**: 可能冗余或噪声，考虑移除
- **特征稳定性**: 检查特征在不同时间段的重要性变化

## 模型部署

### 1. 模型文件管理

```python
# 加载模型
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 加载特征名称
with open('model_features.pkl', 'rb') as f:
    features = pickle.load(f)
```

### 2. 预测流程

```python
# 数据预处理
def preprocess_data(raw_data):
    # 特征工程
    # 数据清洗
    # 特征选择
    return processed_data

# 模型预测
def predict_risk(processed_data):
    probabilities = model.predict_proba(processed_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
```

## 常见问题

### 1. 内存不足

**问题**: 训练过程中内存不足
**解决方案**:
- 减少 `n_estimators`
- 增加 `learning_rate`
- 使用数据采样
- 减少特征数量

### 2. 训练时间过长

**问题**: 超参数搜索时间过长
**解决方案**:
- 减少 `max_combinations`
- 使用更小的参数搜索空间
- 使用早停机制
- 并行化训练

### 3. 模型性能不佳

**问题**: AUC-ROC 分数较低
**解决方案**:
- 检查数据质量
- 增加更多特征
- 调整目标变量定义
- 尝试不同的模型参数

### 4. 类别不平衡

**问题**: 高风险样本过少
**解决方案**:
- 使用 `scale_pos_weight` 参数
- 调整目标变量阈值
- 使用SMOTE等采样技术
- 使用Focal Loss等损失函数

## 最佳实践

### 1. 数据管理

- 定期更新训练数据
- 保持数据版本控制
- 记录数据来源和处理过程
- 验证数据质量

### 2. 模型管理

- 记录每次训练的配置和结果
- 使用版本控制管理模型文件
- 定期评估模型性能
- 建立模型回滚机制

### 3. 实验管理

- 使用实验跟踪工具（如MLflow）
- 记录超参数和性能指标
- 比较不同实验的结果
- 建立实验文档

### 4. 监控和维护

- 监控模型在生产环境中的性能
- 定期重新训练模型
- 跟踪特征分布变化
- 建立模型性能预警机制

## 扩展功能

### 1. 多模型集成

```python
# 训练多个模型
models = []
for config in model_configs:
    model = train_model(config)
    models.append(model)

# 集成预测
def ensemble_predict(models, data):
    predictions = []
    for model in models:
        pred = model.predict_proba(data)[:, 1]
        predictions.append(pred)
    return np.mean(predictions, axis=0)
```

### 2. 在线学习

```python
# 增量训练
def incremental_train(model, new_data):
    model.fit(new_data, partial_fit=True)
    return model
```

### 3. 模型解释

```python
# SHAP值分析
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```
