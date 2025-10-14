# CoinGuard API 使用指南

## 概述

CoinGuard API 提供基于机器学习的加密货币风险预测服务。API 使用 FastAPI 框架构建，支持 RESTful 接口。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API 版本**: v1.0.0
- **数据格式**: JSON
- **认证**: 无需认证（开发版本）
- **Logo**: `/static/images/logo_bar.png` 或 `/logo`

## 快速开始

### 1. 启动服务

```bash
cd fastapi
python main.py
```

服务启动后，可以通过以下地址访问：
- API 文档: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc

### 2. 健康检查

```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00",
  "model_loaded": true,
  "model_info": {
    "model_name": "xgboost_model_20231201_100000",
    "model_type": "XGBoost",
    "training_date": "2023-12-01T09:30:00",
    "loaded_at": "2023-12-01T10:00:00"
  }
}
```

## API 端点详解

### 1. 风险预测

**端点**: `POST /predict`

**描述**: 对输入的市场数据进行风险预测

**请求体**:
```json
{
  "data": [
    {
      "symbol": "BTCUSDT",
      "open_time": "2023-12-01T09:00:00",
      "open": 42000.0,
      "high": 42500.0,
      "low": 41800.0,
      "close": 42200.0,
      "volume": 1000.0,
      "close_time": "2023-12-01T10:00:00",
      "quote_asset_volume": 42000000.0,
      "number_of_trades": 500,
      "taker_buy_base_asset_volume": 600.0,
      "taker_buy_quote_asset_volume": 25200000.0,
      "long_short_ratio": 1.2,
      "long_short_position_ratio": 1.1
    }
  ],
  "threshold": 0.6
}
```

**参数说明**:
- `data`: 市场数据数组，每个元素包含一个时间点的市场数据
- `threshold`: 预测阈值（0-1），默认 0.6

**响应示例**:
```json
{
  "predictions": [
    {
      "index": 0,
      "symbol": "BTCUSDT",
      "prediction": 0,
      "probability": 0.25,
      "risk_level": "low",
      "confidence": 0.5
    }
  ],
  "model_info": {
    "model_name": "xgboost_model_20231201_100000",
    "model_type": "XGBoost",
    "training_date": "2023-12-01T09:30:00",
    "features_count": 45,
    "training_samples": 100000,
    "loaded_at": "2023-12-01T10:00:00"
  },
  "request_info": {
    "data_count": 1,
    "threshold": 0.6,
    "timestamp": "2023-12-01T10:00:00"
  },
  "prediction_statistics": {
    "total_predictions": 1,
    "high_risk_count": 0,
    "low_risk_count": 1,
    "high_risk_ratio": 0.0,
    "avg_probability": 0.25,
    "avg_confidence": 0.5,
    "min_probability": 0.25,
    "max_probability": 0.25,
    "symbol_statistics": {
      "BTCUSDT": {
        "total": 1,
        "high_risk": 0,
        "low_risk": 1,
        "avg_probability": 0.25,
        "avg_confidence": 0.5
      }
    }
  }
}
```

### 2. 模型信息

**端点**: `GET /model/info`

**描述**: 获取当前加载模型的信息

**响应示例**:
```json
{
  "model_name": "xgboost_model_20231201_100000",
  "model_type": "XGBoost",
  "training_date": "2023-12-01T09:30:00",
  "features_count": 45,
  "training_samples": 100000,
  "target_distribution": {
    "0": 95000,
    "1": 5000
  },
  "loaded_at": "2023-12-01T10:00:00"
}
```

### 3. 重新加载模型

**端点**: `POST /model/reload`

**描述**: 重新加载最新的模型文件

**响应示例**:
```json
{
  "message": "模型重新加载成功",
  "timestamp": "2023-12-01T10:00:00",
  "model_info": {
    "model_name": "xgboost_model_20231201_100000",
    "model_type": "XGBoost",
    "training_date": "2023-12-01T09:30:00",
    "features_count": 45,
    "training_samples": 100000,
    "loaded_at": "2023-12-01T10:00:00"
  }
}
```

### 4. 获取Logo

**端点**: `GET /logo`

**描述**: 获取CoinGuard项目的logo图片

**响应**: 返回PNG格式的logo图片文件

### 5. 特征列表

**端点**: `GET /features`

**描述**: 获取模型使用的特征列表

**响应示例**:
```json
{
  "features": [
    "price_change_1h",
    "return_1h",
    "log_return_1h",
    "volume_change_1h",
    "price_range",
    "candle_body",
    "lag_return_1h",
    "lag_return_2h",
    "lag_return_3h",
    "lag_volume_1h",
    "lag_volume_2h",
    "lag_volume_3h",
    "rolling_mean_close_6h",
    "rolling_std_close_6h",
    "rolling_mean_volume_6h",
    "rolling_mean_close_12h",
    "rolling_std_close_12h",
    "rolling_mean_volume_12h",
    "rolling_mean_close_24h",
    "rolling_std_close_24h",
    "rolling_mean_volume_24h",
    "RSI",
    "ATR",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "BBL_20_2.0",
    "BBM_20_2.0",
    "BBU_20_2.0",
    "BBB_20_2.0",
    "BBP_20_2.0",
    "long_short_ratio",
    "long_short_position_ratio"
  ],
  "feature_count": 33
}
```

## 数据格式说明

### 市场数据字段

| 字段名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| symbol | string | 是 | 交易对符号，如 BTCUSDT |
| open_time | string | 是 | 开盘时间，ISO 格式 |
| open | float | 是 | 开盘价，必须 > 0 |
| high | float | 是 | 最高价，必须 > 0 |
| low | float | 是 | 最低价，必须 > 0 |
| close | float | 是 | 收盘价，必须 > 0 |
| volume | float | 是 | 成交量，必须 >= 0 |
| close_time | string | 是 | 收盘时间，ISO 格式 |
| quote_asset_volume | float | 否 | 成交额，必须 >= 0 |
| number_of_trades | int | 否 | 成交笔数，必须 >= 0 |
| taker_buy_base_asset_volume | float | 否 | 主动买入成交量，必须 >= 0 |
| taker_buy_quote_asset_volume | float | 否 | 主动买入成交额，必须 >= 0 |
| long_short_ratio | float | 否 | 多空比，必须 > 0 |
| long_short_position_ratio | float | 否 | 持仓多空比，必须 > 0 |

### 预测结果字段

| 字段名 | 类型 | 描述 |
|--------|------|------|
| index | int | 数据索引 |
| symbol | string | 交易对符号 |
| prediction | int | 预测结果 (0=低风险, 1=高风险) |
| probability | float | 高风险概率 (0-1) |
| risk_level | string | 风险等级 ("low" 或 "high") |
| confidence | float | 预测置信度 (0-1) |

## 错误处理

### 常见错误码

| 状态码 | 描述 | 解决方案 |
|--------|------|----------|
| 400 | 请求数据格式错误 | 检查请求体格式和字段类型 |
| 404 | 模型未加载 | 等待模型加载完成或重新加载模型 |
| 500 | 服务器内部错误 | 检查服务器日志，联系管理员 |
| 503 | 服务不可用 | 检查模型状态，重新加载模型 |

### 错误响应格式

```json
{
  "error": {
    "message": "错误描述",
    "code": 400,
    "timestamp": "2023-12-01T10:00:00"
  }
}
```

## 使用示例

### Python 示例

```python
import requests
import json

# API 基础 URL
BASE_URL = "http://localhost:8000"

# 准备预测数据
prediction_data = {
    "data": [
        {
            "symbol": "BTCUSDT",
            "open_time": "2023-12-01T09:00:00",
            "open": 42000.0,
            "high": 42500.0,
            "low": 41800.0,
            "close": 42200.0,
            "volume": 1000.0,
            "close_time": "2023-12-01T10:00:00",
            "long_short_ratio": 1.2,
            "long_short_position_ratio": 1.1
        }
    ],
    "threshold": 0.6
}

# 发送预测请求
response = requests.post(
    f"{BASE_URL}/predict",
    json=prediction_data,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    print("预测结果:")
    for pred in result["predictions"]:
        print(f"  {pred['symbol']}: {pred['risk_level']} (概率: {pred['probability']:.3f})")
else:
    print(f"请求失败: {response.status_code}")
    print(response.text)
```

### JavaScript 示例

```javascript
const BASE_URL = "http://localhost:8000";

const predictionData = {
    data: [
        {
            symbol: "BTCUSDT",
            open_time: "2023-12-01T09:00:00",
            open: 42000.0,
            high: 42500.0,
            low: 41800.0,
            close: 42200.0,
            volume: 1000.0,
            close_time: "2023-12-01T10:00:00",
            long_short_ratio: 1.2,
            long_short_position_ratio: 1.1
        }
    ],
    threshold: 0.6
};

fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(predictionData)
})
.then(response => response.json())
.then(data => {
    console.log('预测结果:', data);
    data.predictions.forEach(pred => {
        console.log(`${pred.symbol}: ${pred.risk_level} (概率: ${pred.probability.toFixed(3)})`);
    });
})
.catch(error => {
    console.error('请求失败:', error);
});
```

## 性能优化建议

1. **批量预测**: 一次请求处理多个数据点，减少网络开销
2. **连接复用**: 使用连接池或保持连接
3. **异步处理**: 对于大量数据，考虑异步处理
4. **缓存策略**: 对于相同的数据，可以缓存预测结果

## 监控和日志

- API 请求日志记录在服务器控制台
- 模型加载状态可通过 `/health` 端点监控
- 预测性能统计包含在响应中
- 建议在生产环境中配置专门的日志系统

## 安全考虑

1. **输入验证**: API 对所有输入数据进行严格验证
2. **速率限制**: 建议在生产环境中添加速率限制
3. **认证授权**: 生产环境建议添加 API 密钥认证
4. **HTTPS**: 生产环境必须使用 HTTPS
5. **数据隐私**: 确保敏感数据不被记录在日志中
