"""
响应格式化工具
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """响应格式化器类"""
    
    def __init__(self):
        """初始化响应格式化器"""
        pass
    
    def format_prediction_response(
        self, 
        predictions: List[Dict[str, Any]], 
        model_info: Dict[str, Any], 
        request_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        格式化预测响应
        
        Args:
            predictions: 预测结果列表
            model_info: 模型信息
            request_info: 请求信息
            
        Returns:
            Dict: 格式化后的响应
        """
        try:
            # 计算预测统计信息
            prediction_stats = self._calculate_prediction_statistics(predictions)
            
            # 格式化响应
            response = {
                "predictions": predictions,
                "model_info": {
                    "model_name": model_info.get("model_name", "Unknown"),
                    "model_type": model_info.get("model_type", "Unknown"),
                    "training_date": model_info.get("training_date", "Unknown"),
                    "features_count": model_info.get("features_count", 0),
                    "training_samples": model_info.get("training_samples", 0),
                    "loaded_at": model_info.get("loaded_at", "Unknown")
                },
                "request_info": {
                    "data_count": request_info.get("data_count", 0),
                    "threshold": request_info.get("threshold", 0.6),
                    "timestamp": request_info.get("timestamp", "Unknown")
                },
                "prediction_statistics": prediction_stats
            }
            
            return response
            
        except Exception as e:
            logger.error(f"响应格式化失败: {e}")
            raise RuntimeError(f"响应格式化失败: {str(e)}")
    
    def _calculate_prediction_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算预测统计信息
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not predictions:
            return {}
        
        try:
            total_predictions = len(predictions)
            high_risk_count = sum(1 for p in predictions if p.get('prediction', 0) == 1)
            low_risk_count = total_predictions - high_risk_count
            
            probabilities = [p.get('probability', 0) for p in predictions]
            confidences = [p.get('confidence', 0) for p in predictions]
            
            # 按交易对分组统计
            symbol_stats = {}
            for pred in predictions:
                symbol = pred.get('symbol', 'unknown')
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'total': 0,
                        'high_risk': 0,
                        'low_risk': 0,
                        'avg_probability': 0,
                        'avg_confidence': 0
                    }
                
                symbol_stats[symbol]['total'] += 1
                if pred.get('prediction', 0) == 1:
                    symbol_stats[symbol]['high_risk'] += 1
                else:
                    symbol_stats[symbol]['low_risk'] += 1
            
            # 计算每个交易对的平均概率和置信度
            for symbol in symbol_stats:
                symbol_predictions = [p for p in predictions if p.get('symbol') == symbol]
                if symbol_predictions:
                    symbol_probs = [p.get('probability', 0) for p in symbol_predictions]
                    symbol_confs = [p.get('confidence', 0) for p in symbol_predictions]
                    symbol_stats[symbol]['avg_probability'] = sum(symbol_probs) / len(symbol_probs)
                    symbol_stats[symbol]['avg_confidence'] = sum(symbol_confs) / len(symbol_confs)
            
            return {
                'total_predictions': total_predictions,
                'high_risk_count': high_risk_count,
                'low_risk_count': low_risk_count,
                'high_risk_ratio': high_risk_count / total_predictions if total_predictions > 0 else 0,
                'avg_probability': sum(probabilities) / len(probabilities) if probabilities else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'min_probability': min(probabilities) if probabilities else 0,
                'max_probability': max(probabilities) if probabilities else 0,
                'symbol_statistics': symbol_stats
            }
            
        except Exception as e:
            logger.error(f"预测统计计算失败: {e}")
            return {}
    
    def format_error_response(self, error_message: str, error_code: int = 500) -> Dict[str, Any]:
        """
        格式化错误响应
        
        Args:
            error_message: 错误消息
            error_code: 错误代码
            
        Returns:
            Dict: 格式化后的错误响应
        """
        return {
            "error": {
                "message": error_message,
                "code": error_code,
                "timestamp": self._get_current_timestamp()
            }
        }
    
    def format_health_response(
        self, 
        status: str, 
        model_loaded: bool, 
        model_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        格式化健康检查响应
        
        Args:
            status: 服务状态
            model_loaded: 模型是否已加载
            model_info: 模型信息
            
        Returns:
            Dict: 格式化后的健康检查响应
        """
        response = {
            "status": status,
            "timestamp": self._get_current_timestamp(),
            "model_loaded": model_loaded
        }
        
        if model_loaded and model_info:
            response["model_info"] = {
                "model_name": model_info.get("model_name", "Unknown"),
                "model_type": model_info.get("model_type", "Unknown"),
                "training_date": model_info.get("training_date", "Unknown"),
                "loaded_at": model_info.get("loaded_at", "Unknown")
            }
        
        return response
    
    def format_model_info_response(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化模型信息响应
        
        Args:
            model_info: 模型信息
            
        Returns:
            Dict: 格式化后的模型信息响应
        """
        return {
            "model_info": model_info,
            "timestamp": self._get_current_timestamp()
        }
    
    def format_feature_list_response(self, features: List[str]) -> Dict[str, Any]:
        """
        格式化特征列表响应
        
        Args:
            features: 特征列表
            
        Returns:
            Dict: 格式化后的特征列表响应
        """
        return {
            "features": features,
            "feature_count": len(features),
            "timestamp": self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def format_prediction_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        格式化预测摘要
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            Dict: 预测摘要
        """
        if not predictions:
            return {"summary": "无预测结果"}
        
        try:
            # 统计高风险和低风险的数量
            high_risk = [p for p in predictions if p.get('prediction', 0) == 1]
            low_risk = [p for p in predictions if p.get('prediction', 0) == 0]
            
            # 按风险级别分组
            risk_summary = {
                "high_risk": {
                    "count": len(high_risk),
                    "symbols": list(set(p.get('symbol', 'unknown') for p in high_risk)),
                    "avg_probability": sum(p.get('probability', 0) for p in high_risk) / len(high_risk) if high_risk else 0
                },
                "low_risk": {
                    "count": len(low_risk),
                    "symbols": list(set(p.get('symbol', 'unknown') for p in low_risk)),
                    "avg_probability": sum(p.get('probability', 0) for p in low_risk) / len(low_risk) if low_risk else 0
                }
            }
            
            return {
                "summary": f"预测完成: {len(predictions)} 个数据点，{len(high_risk)} 个高风险，{len(low_risk)} 个低风险",
                "risk_summary": risk_summary,
                "total_predictions": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"预测摘要格式化失败: {e}")
            return {"summary": "预测摘要生成失败"}
