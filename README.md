# CoinGuard
Digital Asset Risk Analytics that based on XGBoost 


## 项目进程

1. 定义风险：未来6小时内下跌超过10%。

2. 获取数据： 数据的下载及获取。

3. 特征工程：使用pandas-ta库，计算至少20-30个技术指标作为特征。

4. 数据划分：严格按照时间顺序划分训练集和验证集。

5. 模型训练：使用LightGBM或XGBoost训练一个二元分类器。

6. 模型评估：重点关注召回率和PR曲线。
