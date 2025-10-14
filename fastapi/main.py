"""
FastAPI主应用
提供模型预测API服务
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# 添加项目根目录到路径，优先本地模块
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用相对/本地导入，避免与第三方 fastapi 包冲突
from models.model_manager import ModelManager
from utils.data_validator import DataValidator
from utils.response_formatter import ResponseFormatter
from data.raw.download import (
    KLINE_COLUMNS,
    fetch_klines_for_symbol,
    fetch_account_ratio_for_symbol,
    fetch_position_ratio_for_symbol,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="CoinGuard Risk Prediction API",
    description="基于机器学习的加密货币风险预测API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="../static"), name="static")

# 全局变量
model_manager = None
data_validator = None
response_formatter = None


# Pydantic模型定义
class MarketData(BaseModel):
    """市场数据模型"""
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    open_time: str = Field(..., description="开盘时间，ISO格式")
    open: float = Field(..., gt=0, description="开盘价")
    high: float = Field(..., gt=0, description="最高价")
    low: float = Field(..., gt=0, description="最低价")
    close: float = Field(..., gt=0, description="收盘价")
    volume: float = Field(..., ge=0, description="成交量")
    close_time: str = Field(..., description="收盘时间，ISO格式")
    quote_asset_volume: Optional[float] = Field(None, ge=0, description="成交额")
    number_of_trades: Optional[int] = Field(None, ge=0, description="成交笔数")
    taker_buy_base_asset_volume: Optional[float] = Field(None, ge=0, description="主动买入成交量")
    taker_buy_quote_asset_volume: Optional[float] = Field(None, ge=0, description="主动买入成交额")
    long_short_ratio: Optional[float] = Field(None, gt=0, description="多空比")
    long_short_position_ratio: Optional[float] = Field(None, gt=0, description="持仓多空比")


class PredictionRequest(BaseModel):
    """预测请求模型"""
    data: List[MarketData] = Field(..., description="市场数据列表")
    threshold: Optional[float] = Field(0.6, ge=0, le=1, description="预测阈值")


class PredictionResponse(BaseModel):
    """预测响应模型"""
    predictions: List[Dict[str, Any]] = Field(..., description="预测结果列表")
    model_info: Dict[str, Any] = Field(..., description="模型信息")
    request_info: Dict[str, Any] = Field(..., description="请求信息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="时间戳")
    model_loaded: bool = Field(..., description="模型是否已加载")
    model_info: Optional[Dict[str, Any]] = Field(None, description="模型信息")


class SymbolPredictRequest(BaseModel):
    """按交易对拉取数据并预测的请求模型"""
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    interval: Optional[str] = Field("1h", description="K线周期，如1h")
    limit: Optional[int] = Field(200, ge=10, le=1000, description="拉取数据条数")
    threshold: Optional[float] = Field(0.6, ge=0, le=1, description="预测阈值")


class SymbolPredictionResponse(BaseModel):
    """按交易对预测的精简响应（仅返回最新一条的方向判断）"""
    symbol: str = Field(..., description="交易对")
    interval: str = Field(..., description="K线周期")
    next_hour: str = Field(..., description="方向：up 或 down")
    probability: float = Field(..., ge=0, le=1, description="预测为上涨的概率")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    threshold: float = Field(..., ge=0, le=1, description="使用的阈值")
    latest_open_time: Optional[str] = Field(None, description="最新K线开盘时间（ISO）")
    model_name: Optional[str] = Field(None, description="模型名称")


# 依赖注入
def get_model_manager():
    """获取模型管理器实例"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager


def get_data_validator():
    """获取数据验证器实例"""
    global data_validator
    if data_validator is None:
        data_validator = DataValidator()
    return data_validator


def get_response_formatter():
    """获取响应格式化器实例"""
    global response_formatter
    if response_formatter is None:
        response_formatter = ResponseFormatter()
    return response_formatter


# API路由
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动CoinGuard API服务...")
    
    # 初始化模型管理器
    global model_manager
    model_manager = ModelManager()
    
    # 尝试加载最新模型
    try:
        model_manager.load_latest_model()
        logger.info("模型加载成功")
    except Exception as e:
        logger.warning(f"模型加载失败: {e}")
    
    logger.info("CoinGuard API服务启动完成")


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "CoinGuard Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "logo": "/static/images/logo_bar.png"
    }


@app.get("/logo")
async def get_logo():
    """获取CoinGuard logo"""
    logo_path = os.path.join(os.path.dirname(__file__), "..", "static", "images", "logo_bar.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    model_manager = get_model_manager()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.is_model_loaded(),
        model_info=model_manager.get_model_info() if model_manager.is_model_loaded() else None
    )


 


@app.get("/model/info")
async def get_model_info(model_manager: ModelManager = Depends(get_model_manager)):
    """获取模型信息"""
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=404,
            detail="模型未加载"
        )
    
    return model_manager.get_model_info()


@app.post("/model/reload")
async def reload_model(model_manager: ModelManager = Depends(get_model_manager)):
    """重新加载模型"""
    try:
        model_manager.load_latest_model()
        return {
            "message": "模型重新加载成功",
            "timestamp": datetime.now().isoformat(),
            "model_info": model_manager.get_model_info()
        }
    except Exception as e:
        logger.error(f"模型重新加载失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"模型重新加载失败: {str(e)}"
        )


@app.get("/features")
async def get_feature_list(model_manager: ModelManager = Depends(get_model_manager)):
    """获取模型特征列表"""
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=404,
            detail="模型未加载"
        )
    
    return {
        "features": model_manager.get_feature_names(),
        "feature_count": len(model_manager.get_feature_names())
    }


@app.post("/predict/symbol", response_model=SymbolPredictionResponse)
async def predict_by_symbol(
    request: SymbolPredictRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    response_formatter: ResponseFormatter = Depends(get_response_formatter)
):
    """按交易对自动获取数据并进行预测"""
    try:
        # 检查模型是否已加载
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=503, detail="模型未加载，请稍后重试或联系管理员")

        symbol = request.symbol.upper().strip()
        interval = request.interval
        limit = request.limit

        # 1) 拉取K线数据
        klines = fetch_klines_for_symbol(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            raise HTTPException(status_code=502, detail=f"获取 {symbol} K线数据失败")

        # 转为DataFrame并清洗
        records = [[symbol] + k for k in klines]
        df = pd.DataFrame(records, columns=KLINE_COLUMNS)
        if 'ignore' in df.columns:
            df = df.drop(columns=['ignore'])

        # 数值列转型
        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2) 合并多空比
        ratios = fetch_account_ratio_for_symbol(symbol=symbol, period='1h', limit=limit)
        if ratios:
            ratio_df = pd.DataFrame([
                {
                    'symbol': r.get('symbol', symbol),
                    'timestamp': pd.to_numeric(r.get('timestamp'), errors='coerce'),
                    'longShortRatio': pd.to_numeric(r.get('longShortRatio'), errors='coerce'),
                }
                for r in ratios
            ])
            if not ratio_df.empty:
                df = df.merge(
                    ratio_df[['symbol', 'timestamp', 'longShortRatio']],
                    left_on=['symbol', 'open_time'],
                    right_on=['symbol', 'timestamp'],
                    how='left'
                ).drop(columns=['timestamp'])
                df = df.rename(columns={'longShortRatio': 'long_short_ratio'})
        else:
            df['long_short_ratio'] = np.nan

        # 3) 合并持仓多空比
        pos_ratios = fetch_position_ratio_for_symbol(symbol=symbol, period='1h', limit=limit)
        if pos_ratios:
            pos_ratio_df = pd.DataFrame([
                {
                    'symbol': r.get('symbol', symbol),
                    'timestamp': pd.to_numeric(r.get('timestamp'), errors='coerce'),
                    'longShortRatio': pd.to_numeric(r.get('longShortRatio'), errors='coerce'),
                }
                for r in pos_ratios
            ])
            if not pos_ratio_df.empty:
                df = df.merge(
                    pos_ratio_df[['symbol', 'timestamp', 'longShortRatio']],
                    left_on=['symbol', 'open_time'],
                    right_on=['symbol', 'timestamp'],
                    how='left'
                ).drop(columns=['timestamp'])
                df = df.rename(columns={'longShortRatio': 'long_short_position_ratio'})
        else:
            df['long_short_position_ratio'] = np.nan

        # 确保时间排序，取最新一条
        try:
            df_sorted = df.copy()
            # open_time 可能是毫秒整型，转换为datetime仅用于响应展示
            latest_open_time_iso = None
            if 'open_time' in df_sorted.columns:
                # 保存最新一条的时间戳（毫秒或秒）
                latest_row = df_sorted.sort_values('open_time').iloc[-1]
                try:
                    ts = pd.to_numeric(latest_row['open_time'], errors='coerce')
                    # 兼容毫秒/秒级时间戳
                    if not np.isnan(ts):
                        if ts > 10_000_000_000:  # 毫秒
                            latest_open_time_iso = pd.to_datetime(ts, unit='ms').isoformat()
                        else:  # 秒
                            latest_open_time_iso = pd.to_datetime(ts, unit='s').isoformat()
                except Exception:
                    latest_open_time_iso = None
                df_sorted = df_sorted.sort_values('open_time')
            else:
                df_sorted = df_sorted.reset_index(drop=True)
        except Exception:
            df_sorted = df
            latest_open_time_iso = None

        # 4) 进行预测（对排序后的数据）
        predictions = model_manager.predict(df_sorted, threshold=request.threshold)
        if not predictions:
            raise HTTPException(status_code=500, detail="预测结果为空")

        last_pred = predictions[-1]
        direction = 'up' if int(last_pred.get('prediction', 0)) == 1 else 'down'

        return SymbolPredictionResponse(
            symbol=symbol,
            interval=interval,
            next_hour=direction,
            probability=float(last_pred.get('probability', 0.0)),
            confidence=float(last_pred.get('confidence', 0.0)),
            threshold=float(request.threshold or 0.6),
            latest_open_time=latest_open_time_iso,
            model_name=(model_manager.get_model_info() or {}).get('model_name')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"按symbol预测过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"按symbol预测失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
