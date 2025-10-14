#!/usr/bin/env python3
"""
CoinGuard 项目启动脚本
提供统一的命令行接口来管理整个项目
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(cmd, cwd=None):
    """运行命令并处理错误"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


def setup_environment():
    """初始化项目环境"""
    print("正在初始化项目环境...")
    
    # 创建必要的目录
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "logs",
        "backups"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("项目环境初始化完成!")


def download_data():
    """下载原始数据"""
    print("正在下载原始数据...")
    try:
        # 直接运行下载脚本，不捕获输出以显示进度条
        result = subprocess.run(f"{sys.executable} download.py", shell=True, cwd="data/raw", check=True)
        print("数据下载完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"数据下载失败: {e}")
        return False


def generate_features():
    """生成特征数据"""
    print("正在生成特征数据...")
    success = run_command(f"{sys.executable} feature_engineering.py", cwd="data/processed")
    if success:
        print("特征生成完成!")
    else:
        print("特征生成失败!")
    return success


def train_model():
    """训练模型"""
    print("正在训练模型...")
    success = run_command(f"{sys.executable} train_model.py", cwd="training")
    if success:
        print("模型训练完成!")
    else:
        print("模型训练失败!")
    return success


def start_api():
    """启动API服务"""
    print("正在启动API服务...")
    print("API服务将在 http://localhost:8000 启动")
    print("API文档地址: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务")
    
    try:
        run_command(f"{sys.executable} main.py", cwd="fastapi")
    except KeyboardInterrupt:
        print("\nAPI服务已停止")


def run_tests():
    """运行测试"""
    print("正在运行测试...")
    success = run_command(f"{sys.executable} -m pytest testing/ -v", cwd=".")
    if success:
        print("所有测试通过!")
    else:
        print("测试失败!")
    return success


def show_status():
    """显示项目状态"""
    print("=== CoinGuard 项目状态 ===")
    
    # 检查数据文件
    data_files = {
        "原始数据": "data/raw/crypto_klines_data.csv",
        "特征数据": "data/processed/features_crypto_data.csv"
    }
    
    for name, path in data_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {name}: {path} ({size:.2f} MB)")
        else:
            print(f"✗ {name}: {path} (不存在)")
    
    # 检查模型文件
    models_dir = "data/models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            print(f"✓ 模型文件: {len(model_files)} 个文件")
            for model_file in model_files[:3]:  # 显示前3个
                print(f"  - {model_file}")
            if len(model_files) > 3:
                print(f"  ... 还有 {len(model_files) - 3} 个文件")
        else:
            print("✗ 模型文件: 无")
    else:
        print("✗ 模型文件: 目录不存在")
    
    # 检查API服务状态
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API服务: 运行中")
        else:
            print("✗ API服务: 响应异常")
    except:
        print("✗ API服务: 未运行")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CoinGuard 项目启动脚本")
    parser.add_argument("command", choices=[
        "setup", "download", "features", "train", "api", "test", "status", "pipeline"
    ], help="要执行的命令")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup_environment()
    
    elif args.command == "download":
        download_data()
    
    elif args.command == "features":
        generate_features()
    
    elif args.command == "train":
        train_model()
    
    elif args.command == "api":
        start_api()
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "status":
        show_status()
    
    elif args.command == "pipeline":
        print("正在执行完整的数据处理流程...")
        if download_data():
            if generate_features():
                train_model()
            else:
                print("特征生成失败，停止流程")
        else:
            print("数据下载失败，停止流程")


if __name__ == "__main__":
    main()
