"""
数据验证工具
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器类"""
    
    def __init__(self):
        """初始化数据验证器"""
        self.required_fields = [
            'symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'
        ]
        
        self.optional_fields = [
            'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'long_short_ratio', 'long_short_position_ratio'
        ]
    
    def validate_market_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证市场数据
        
        Args:
            data: 市场数据列表
            
        Returns:
            Dict: 验证结果
        """
        errors = []
        warnings = []
        
        try:
            # 检查数据是否为空
            if not data:
                errors.append("数据列表为空")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            # 转换为DataFrame进行验证
            df = pd.DataFrame(data)
            
            # 检查必需字段
            missing_fields = set(self.required_fields) - set(df.columns)
            if missing_fields:
                errors.append(f"缺少必需字段: {list(missing_fields)}")
            
            # 检查数据类型和值
            for field in self.required_fields:
                if field in df.columns:
                    field_errors = self._validate_field(df, field)
                    errors.extend(field_errors)
            
            # 检查可选字段
            for field in self.optional_fields:
                if field in df.columns:
                    field_warnings = self._validate_optional_field(df, field)
                    warnings.extend(field_warnings)
            
            # 检查数据一致性
            consistency_errors = self._validate_data_consistency(df)
            errors.extend(consistency_errors)
            
            # 检查数据质量
            quality_warnings = self._validate_data_quality(df)
            warnings.extend(quality_warnings)
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "data_count": len(df)
            }
            
        except Exception as e:
            logger.error(f"数据验证过程中发生错误: {e}")
            return {
                "valid": False,
                "errors": [f"数据验证失败: {str(e)}"],
                "warnings": warnings
            }
    
    def _validate_field(self, df: pd.DataFrame, field: str) -> List[str]:
        """验证单个字段"""
        errors = []
        
        try:
            if field == 'symbol':
                # 检查symbol字段
                if df[field].isna().any():
                    errors.append(f"{field}: 包含空值")
                if df[field].str.strip().eq('').any():
                    errors.append(f"{field}: 包含空字符串")
            
            elif field in ['open_time', 'close_time']:
                # 检查时间字段
                if df[field].isna().any():
                    errors.append(f"{field}: 包含空值")
                else:
                    # 尝试转换为datetime
                    try:
                        pd.to_datetime(df[field])
                    except:
                        errors.append(f"{field}: 时间格式无效")
            
            elif field in ['open', 'high', 'low', 'close', 'volume']:
                # 检查数值字段
                if df[field].isna().any():
                    errors.append(f"{field}: 包含空值")
                
                # 检查是否为数值类型
                if not pd.api.types.is_numeric_dtype(df[field]):
                    errors.append(f"{field}: 不是数值类型")
                
                # 检查价格字段的合理性
                if field in ['open', 'high', 'low', 'close']:
                    if (df[field] <= 0).any():
                        errors.append(f"{field}: 包含非正值")
                
                # 检查成交量字段
                if field == 'volume':
                    if (df[field] < 0).any():
                        errors.append(f"{field}: 包含负值")
            
        except Exception as e:
            errors.append(f"{field}: 验证失败 - {str(e)}")
        
        return errors
    
    def _validate_optional_field(self, df: pd.DataFrame, field: str) -> List[str]:
        """验证可选字段"""
        warnings = []
        
        try:
            if field in ['quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                # 检查数值字段
                if df[field].isna().any():
                    warnings.append(f"{field}: 包含空值")
                elif not pd.api.types.is_numeric_dtype(df[field]):
                    warnings.append(f"{field}: 不是数值类型")
                elif (df[field] < 0).any():
                    warnings.append(f"{field}: 包含负值")
            
            elif field == 'number_of_trades':
                # 检查交易笔数
                if df[field].isna().any():
                    warnings.append(f"{field}: 包含空值")
                elif not pd.api.types.is_numeric_dtype(df[field]):
                    warnings.append(f"{field}: 不是数值类型")
                elif (df[field] < 0).any():
                    warnings.append(f"{field}: 包含负值")
            
            elif field in ['long_short_ratio', 'long_short_position_ratio']:
                # 检查多空比
                if df[field].isna().any():
                    warnings.append(f"{field}: 包含空值")
                elif not pd.api.types.is_numeric_dtype(df[field]):
                    warnings.append(f"{field}: 不是数值类型")
                elif (df[field] <= 0).any():
                    warnings.append(f"{field}: 包含非正值")
        
        except Exception as e:
            warnings.append(f"{field}: 验证失败 - {str(e)}")
        
        return warnings
    
    def _validate_data_consistency(self, df: pd.DataFrame) -> List[str]:
        """验证数据一致性"""
        errors = []
        
        try:
            # 检查high >= low
            if 'high' in df.columns and 'low' in df.columns:
                invalid_high_low = df['high'] < df['low']
                if invalid_high_low.any():
                    errors.append(f"发现 {invalid_high_low.sum()} 个数据点的high < low")
            
            # 检查价格在high和low之间
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
                invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
                
                if invalid_open.any():
                    errors.append(f"发现 {invalid_open.sum()} 个数据点的open价格超出high-low范围")
                if invalid_close.any():
                    errors.append(f"发现 {invalid_close.sum()} 个数据点的close价格超出high-low范围")
            
            # 检查时间顺序
            if 'open_time' in df.columns and 'close_time' in df.columns:
                try:
                    open_times = pd.to_datetime(df['open_time'])
                    close_times = pd.to_datetime(df['close_time'])
                    invalid_time_order = close_times <= open_times
                    if invalid_time_order.any():
                        errors.append(f"发现 {invalid_time_order.sum()} 个数据点的close_time <= open_time")
                except:
                    pass  # 时间格式错误已在字段验证中处理
        
        except Exception as e:
            errors.append(f"数据一致性验证失败: {str(e)}")
        
        return errors
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """验证数据质量"""
        warnings = []
        
        try:
            # 检查重复数据
            if 'symbol' in df.columns and 'open_time' in df.columns:
                duplicates = df.duplicated(subset=['symbol', 'open_time']).sum()
                if duplicates > 0:
                    warnings.append(f"发现 {duplicates} 个重复的数据点")
            
            # 检查异常值（使用简单的统计方法）
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    # 使用IQR方法检测异常值
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        warnings.append(f"{col}: 发现 {outliers} 个可能的异常值")
            
            # 检查数据完整性
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isna().sum().sum()
            if missing_cells > 0:
                missing_ratio = missing_cells / total_cells
                warnings.append(f"数据完整性: {missing_ratio:.2%} 的单元格为空")
        
        except Exception as e:
            warnings.append(f"数据质量验证失败: {str(e)}")
        
        return warnings
    
    def validate_prediction_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证预测请求
        
        Args:
            request_data: 请求数据
            
        Returns:
            Dict: 验证结果
        """
        errors = []
        warnings = []
        
        try:
            # 检查必需字段
            if 'data' not in request_data:
                errors.append("缺少必需字段: data")
                return {"valid": False, "errors": errors, "warnings": warnings}
            
            # 验证数据字段
            data_validation = self.validate_market_data(request_data['data'])
            if not data_validation['valid']:
                errors.extend(data_validation['errors'])
            warnings.extend(data_validation['warnings'])
            
            # 验证阈值
            if 'threshold' in request_data:
                threshold = request_data['threshold']
                if not isinstance(threshold, (int, float)):
                    errors.append("threshold必须是数值类型")
                elif threshold < 0 or threshold > 1:
                    errors.append("threshold必须在0和1之间")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"预测请求验证失败: {e}")
            return {
                "valid": False,
                "errors": [f"预测请求验证失败: {str(e)}"],
                "warnings": warnings
            }
