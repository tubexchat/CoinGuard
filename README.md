# CoinGuard: 基于机器学习的加密货币风险预测系统

<img src="static/images/logo_bar.png"/>

CoinGuard 是一个基于机器学习的开源加密货币风险预测系统。它使用工程化的市场微观结构和技术指标，通过XGBoost在时间分割数据上进行训练，来检测加密货币市场的高风险场景。项目采用模块化架构设计，分为训练区域、测试区域、FastAPI区域、数据存储区域和文档区域。核心功能为预测某个数字货币合约未来6小时的涨跌。

## 🚀 主要特性

- **模块化架构**: 清晰的区域划分，便于维护和扩展
- **完整的数据管道**: 从数据下载到特征工程的自动化流程
- **丰富的特征工程**: 收益率、波动率、滚动统计、技术指标（RSI、MACD、布林带、ATR）
- **时间感知的目标创建**: 时间分割避免数据泄露
- **类别不平衡处理**: 通过 `scale_pos_weight` 处理不平衡数据
- **超参数搜索**: 基于验证集ROC AUC的自动调优
- **RESTful API**: 提供模型预测和管理接口
- **完整的测试框架**: 单元测试和集成测试
- **模型持久化**: PKL格式保存，便于部署

## 📁 项目架构

```
CoinGuard/
├── training/                 # 训练区域
│   ├── train_model.py       # 主训练脚本
│   ├── configs/             # 配置文件
│   └── utils/               # 工具函数
├── testing/                 # 测试区域
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   └── fixtures/            # 测试数据
├── fastapi/                 # FastAPI区域
│   ├── main.py              # API主应用
│   ├── models/              # 模型管理
│   └── utils/               # API工具
├── data/                    # 数据存储区域
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   ├── models/              # 模型文件
│   └── data_manager.py      # 数据管理器
├── static/                  # 静态资源区域
│   └── images/              # 图片资源
│       └── logo_bar.png     # CoinGuard logo
├── docs/                    # 文档区域
│   ├── README.md            # 架构文档
│   ├── api/                 # API文档
│   └── training/            # 训练文档
└── run.py                   # 项目启动脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
# 初始化环境
python run.py setup

# 下载数据(大约在3个半小时)
python run.py download

# 生成特征
python run.py features

# 训练模型（data/models/ 目录下生成相应的PKL文件）
python training/train_model.py
# or
python run.py train 

# 启动API服务
python run.py api

# 运行测试
python run.py test

# 查看状态
python run.py status
```

## 持久化模型的分类

1. **{model_name}.pkl**: 训练好的 XGBoost 模型对象，用于预测（predict_proba/predict）。
2. **{model_name}features.pkl**: 训练时使用的特征列名列表（含顺序）。在线推理时按此顺序对齐特征，避免列错位。
3. **{model_name}config.pkl**: 训练配置字典，包含数据路径、模型参数、数据集划分比例、评估阈值等，便于复现实验与审计。
4. **{model_name}stats.pkl**: 训练统计信息，例如样本量、特征数、目标分布、模型类型、训练时间等，用于监控与记录。

FastAPI 服务通过 fastapi/models/model_manager.py 的 load_latest_model() 同时加载这四个文件：模型用于预测；features 保证输入列顺序；config 和 stats 提供元数据与可观测性。