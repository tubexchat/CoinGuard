# CoinGuard 项目架构文档

## 项目概述

CoinGuard 是一个基于机器学习的加密货币风险预测系统，采用模块化架构设计，分为训练区域、测试区域、FastAPI区域、数据存储区域和文档区域。

## 项目架构

```
CoinGuard/
├── training/                 # 训练区域
│   ├── train_model.py       # 主训练脚本
│   ├── configs/             # 配置文件
│   │   └── model_config.py  # 模型配置
│   └── utils/               # 工具函数
│       ├── data_utils.py    # 数据处理工具
│       └── model_utils.py   # 模型工具
├── testing/                 # 测试区域
│   ├── unit/                # 单元测试
│   │   ├── test_data_utils.py
│   │   └── test_model_utils.py
│   ├── integration/         # 集成测试
│   │   └── test_model_pipeline.py
│   └── fixtures/            # 测试数据
│       └── sample_data.py
├── fastapi/                 # FastAPI区域
│   ├── main.py              # API主应用
│   ├── models/              # 模型管理
│   │   └── model_manager.py
│   └── utils/               # API工具
│       ├── data_validator.py
│       └── response_formatter.py
├── data/                    # 数据存储区域
│   ├── raw/                 # 原始数据
│   │   └── download.py      # 数据下载脚本
│   ├── processed/           # 处理后数据
│   │   └── feature_engineering.py
│   ├── models/              # 模型文件
│   └── data_manager.py      # 数据管理器
├── docs/                    # 文档区域
│   ├── README.md            # 本文档
│   ├── api/                 # API文档
│   ├── training/            # 训练文档
│   └── testing/             # 测试文档
└── requirements.txt         # 依赖包
```

## 各区域详细说明

### 1. 训练区域 (training/)

负责模型训练和超参数调优。

**主要功能：**
- 数据加载和预处理
- 特征工程
- 模型训练（XGBoost）
- 超参数搜索
- 模型评估
- 模型保存（PKL格式）

**核心文件：**
- `train_model.py`: 主训练脚本，包含完整的训练流程
- `configs/model_config.py`: 集中化配置管理
- `utils/data_utils.py`: 数据处理工具函数
- `utils/model_utils.py`: 模型训练和评估工具

**使用方式：**
```bash
cd training
python train_model.py
```

### 2. 测试区域 (testing/)

提供完整的测试框架，确保代码质量和模型性能。

**测试类型：**
- **单元测试**: 测试各个工具函数的正确性
- **集成测试**: 测试完整的训练和预测流程
- **测试数据**: 提供各种测试场景的样本数据

**核心文件：**
- `unit/test_*.py`: 单元测试文件
- `integration/test_model_pipeline.py`: 集成测试
- `fixtures/sample_data.py`: 测试数据生成器

**运行测试：**
```bash
cd testing
python -m pytest unit/
python -m pytest integration/
```

### 3. FastAPI区域 (fastapi/)

提供RESTful API服务，用于模型预测和模型管理。

**主要功能：**
- 模型预测API
- 模型信息查询
- 健康检查
- 数据验证
- 响应格式化

**核心文件：**
- `main.py`: FastAPI主应用
- `models/model_manager.py`: 模型管理器
- `utils/data_validator.py`: 数据验证器
- `utils/response_formatter.py`: 响应格式化器

**启动API服务：**
```bash
cd fastapi
python main.py
```

**API端点：**
- `GET /`: 根路径
- `GET /health`: 健康检查
- `POST /predict`: 风险预测
- `GET /model/info`: 模型信息
- `POST /model/reload`: 重新加载模型
- `GET /features`: 特征列表

### 4. 数据存储区域 (data/)

统一管理所有CSV数据，确保数据的一致性和可追溯性。

**数据分类：**
- **原始数据** (`raw/`): 从API获取的原始K线数据
- **处理后数据** (`processed/`): 经过特征工程处理的数据
- **模型文件** (`models/`): 训练好的模型文件（PKL格式）

**核心文件：**
- `data_manager.py`: 数据管理器，统一管理所有数据操作
- `raw/download.py`: 数据下载脚本
- `processed/feature_engineering.py`: 特征工程脚本

**数据流程：**
1. 运行 `data/raw/download.py` 下载原始数据
2. 运行 `data/processed/feature_engineering.py` 进行特征工程
3. 训练区域使用处理后的数据进行模型训练
4. 模型保存到 `data/models/` 目录

### 5. 文档区域 (docs/)

提供完整的项目文档和使用说明。

**文档结构：**
- `README.md`: 项目架构总览
- `api/`: API使用文档
- `training/`: 训练流程文档
- `testing/`: 测试指南

## 数据流

```
原始数据 (API) 
    ↓
数据下载 (data/raw/download.py)
    ↓
原始CSV数据 (data/raw/)
    ↓
特征工程 (data/processed/feature_engineering.py)
    ↓
处理后CSV数据 (data/processed/)
    ↓
模型训练 (training/train_model.py)
    ↓
模型文件 (data/models/*.pkl)
    ↓
API预测 (fastapi/main.py)
```

## 模型格式

训练好的模型以PKL格式保存，包含以下文件：
- `{model_name}.pkl`: 训练好的XGBoost模型
- `{model_name}_features.pkl`: 特征列名
- `{model_name}_config.pkl`: 训练配置
- `{model_name}_stats.pkl`: 训练统计信息

## 配置管理

所有配置集中在 `training/configs/model_config.py` 中，包括：
- 数据路径配置
- 模型参数配置
- 超参数搜索配置
- 评估配置

## 测试策略

- **单元测试**: 覆盖所有工具函数
- **集成测试**: 验证完整的训练和预测流程
- **数据验证**: 确保数据质量和一致性
- **API测试**: 验证API接口的正确性

## 部署建议

1. **开发环境**: 使用较小的数据集和较少的超参数组合进行快速迭代
2. **生产环境**: 使用完整数据集和充分的超参数搜索
3. **API部署**: 使用Docker容器化部署，配置负载均衡
4. **监控**: 添加模型性能监控和API使用统计

## 扩展性

该架构设计具有良好的扩展性：
- 可以轻松添加新的特征工程方法
- 可以集成其他机器学习算法
- 可以添加更多的API端点
- 可以扩展测试覆盖范围

## 维护指南

1. **定期更新数据**: 确保使用最新的市场数据
2. **模型重训练**: 定期使用新数据重新训练模型
3. **性能监控**: 监控模型预测性能和API响应时间
4. **日志管理**: 定期清理和归档日志文件
5. **备份策略**: 定期备份模型文件和配置
