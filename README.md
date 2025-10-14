# CoinGuard: 基于机器学习的加密货币风险预测系统


CoinGuard 是一个基于机器学习的开源加密货币风险预测系统。它使用工程化的市场微观结构和技术指标，通过XGBoost在时间分割数据上进行训练，来检测加密货币市场的高风险场景。项目采用模块化架构设计，分为训练区域、测试区域、FastAPI区域、数据存储区域和文档区域。

<img src="static/images/logo_bar.png"/>

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

# 下载数据
python run.py download

# 生成特征
python run.py features

# 训练模型
python run.py train

# 启动API服务
python run.py api

# 运行测试
python run.py test

# 查看状态
python run.py status
```