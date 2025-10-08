"""
FastAPI主应用
提供模型预测API服务
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.models.model_manager import ModelManager
from fastapi.utils.data_validator import DataValidator
from fastapi.utils.response_formatter import ResponseFormatter

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
        "docs": "/docs"
    }


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


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    data_validator: DataValidator = Depends(get_data_validator),
    response_formatter: ResponseFormatter = Depends(get_response_formatter)
):
    """风险预测API"""
    try:
        # 检查模型是否已加载
        if not model_manager.is_model_loaded():
            raise HTTPException(
                status_code=503,
                detail="模型未加载，请稍后重试或联系管理员"
            )
        
        # 验证输入数据
        validation_result = data_validator.validate_market_data(request.data)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"数据验证失败: {validation_result['errors']}"
            )
        
        # 转换数据格式
        df = pd.DataFrame([item.dict() for item in request.data])
        
        # 进行预测
        predictions = model_manager.predict(df, threshold=request.threshold)
        
        # 格式化响应
        response = response_formatter.format_prediction_response(
            predictions=predictions,
            model_info=model_manager.get_model_info(),
            request_info={
                "data_count": len(request.data),
                "threshold": request.threshold,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"预测过程中发生内部错误: {str(e)}"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
