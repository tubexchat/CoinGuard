# CoinGuard：面向数字资产风险的开源金融机器学习

简体中文 | [English](README.md)

CoinGuard 是一个开源的金融机器学习项目，用于加密数字资产的风险识别与分析。项目基于完善的数据流水线、丰富的特征工程（市场微观结构与技术指标），并采用 XGBoost 在按时间划分的数据集上进行训练与评估，可作为研究基线或生产系统组件。

## 核心特性

- 完整的数据采集与聚合（交易对发现、1 小时 K 线）
- 丰富的特征工程：收益率、波动率、滚动统计、技术指标（RSI、MACD、布林带、ATR）
- 面向时间的目标构造与时间序列切分，避免信息泄露
- 通过 `scale_pos_weight` 处理类别不平衡问题
- 在验证集上以 ROC AUC 进行超参搜索
- 提供混淆矩阵、分类报告与 AUC-ROC 等清晰评估指标

## 仓库结构

- `download.py`：抓取交易对、1h K 线、多空比数据，并合并生成 `data/crypto_klines_data.csv`
- `feature_engineering.py`：构建特征，输出 `data/features_crypto_data.csv`
- `train_model.py`：创建目标变量、时间序列切分、超参搜索与训练、评估、绘制前 20 特征重要性
- `data/`：数据制品
- `reports/`：实验记录与说明

## 快速开始

1）准备环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

要求：Python 3.10+

2）下载市场数据

```bash
python download.py
```

3）构建特征

```bash
python feature_engineering.py
```

4）训练与评估

```bash
python train_model.py
```

运行后将输出控制台指标与特征重要性可视化图表。

## 开仓条件（可选，用于信号部署）

若将 CoinGuard 用于交易信号触发，可增加如下过滤条件：

- 当前费率需小于等于 +0.0100%
- 当日涨幅必须大于 24 小时涨幅

说明：

- 请统一费率与收益率的单位（百分比或小数）
- 在执行前确保数据源中包含所需字段

## 计算预算（Budget）

可通过以下参数控制速度/成本/精度的权衡：

- 超参搜索预算：`train_model.py` 中 `CONFIG["tuning"]["max_combinations"]`（越小越快越省）
- 模型规模与训练时长：降低 `n_estimators`、`max_depth`，或提高 `learning_rate`
- 评估开销：保持 `thresholds_to_test` 较短
- 数据体量：减少下载的交易对或行数（更小数据训练更快）

建议快速迭代时先用较小 `max_combinations`（例如 10–20）与适中 `n_estimators`（例如 800），然后再逐步放大。

## 数据来源

- 交易对列表：`https://api.lewiszhang.top/ticker/24hr`
- K 线数据：`https://api.lewiszhang.top/klines?symbol=BTCUSDT&interval=1h&limit=1000`
- 账户多空比：`https://api.lewiszhang.top/topLongShortAccountRatio`
- 持仓多空比：`https://api.lewiszhang.top/topLongShortPositionRatio`

## 模型配置（默认）

- 算法：XGBoost 二分类模型
- 预测视角（展望期）：1 小时
- 风险定义：下一小时下跌幅度超过 −10%
- 验证指标：ROC AUC
- 分类阈值：0.6（可配置）

可在 `train_model.py` 的 `CONFIG` 中修改相关参数。

## 评估结果（示例）

基于一次代表性运行（阈值 = 0.6）：

- Accuracy（准确率）：0.9935
- AUC-ROC：0.9545
- 混淆矩阵（行=真实，列=预测）：

```
[[56386   282]
 [   89    60]]
```

- 分类报告（precision / recall / f1 / support）
  - 低风险（0）：0.9984 / 0.9950 / 0.9967 / 56668
  - 高风险（1）：0.1754 / 0.4027 / 0.2444 / 149

说明：

- 该任务存在显著类别不平衡，可通过调整决策阈值权衡召回率与精确率。
- 降低阈值通常会提高召回率（捕获更多风险），但会降低精确率（更多误报）。

## 前 20 特征重要性（示例）

训练脚本会绘制并打印前 20 个特征重要性。典型影响较大者包括：

- `BBB_20_2.0`（布林带宽度）
- `number_of_trades`（成交笔数）
- `RSI`
- `BBP_20_2.0`（布林带位置）
- `BBL_20_2.0` / `BBM_20_2.0` / `BBU_20_2.0`
- `return_1h`、`log_return_1h`、`lag_return_*h`
- `close` 的滚动均值与波动率

运行 `train_model.py` 可基于当前数据与配置复现图表。

## 可复现性与合规性

- 目标构造与时间切分按交易对分组并避免前视信息泄露。
- 模型默认设置固定随机种子，保证结果可复现。
- 本仓库仅用于研究与教育，非任何投资建议。
- 使用前请遵守所在司法辖区监管要求及数据提供方服务条款。

## 安全与风险提示

- 使用公共 API，无需提交任何密钥；请勿提交敏感信息至仓库。
- 本项目仅提供分析能力，与任何实盘交易逻辑应严格隔离、沙箱测试。
- 训练前请校验数据完整性，API 结构变更可能导致流程失败。

## 参与贡献

欢迎通过 Issue/PR 参与：

- 对重大改动请先开 Issue 讨论方案
- 提交信息清晰，并在适用时提供改动前后指标
- 补充必要的测试或可运行示例

## 许可协议

请参阅仓库根目录的 `LICENSE` 文件。


