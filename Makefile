# CoinGuard Makefile
# 提供常用的开发和部署命令

.PHONY: help install test train api clean setup

# 默认目标
help:
	@echo "CoinGuard 项目命令:"
	@echo "  make install     - 安装项目依赖"
	@echo "  make setup       - 初始化项目环境"
	@echo "  make test        - 运行所有测试"
	@echo "  make train       - 训练模型"
	@echo "  make api         - 启动API服务"
	@echo "  make download    - 下载数据"
	@echo "  make features    - 生成特征"
	@echo "  make clean       - 清理临时文件"
	@echo "  make docs        - 生成文档"

# 安装依赖
install:
	pip install -r requirements.txt

# 开发环境安装
install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# 初始化项目环境
setup:
	mkdir -p data/raw data/processed data/models logs
	@echo "项目环境初始化完成"

# 运行测试
test:
	python -m pytest testing/ -v --cov=training --cov=fastapi --cov-report=html

# 运行单元测试
test-unit:
	python -m pytest testing/unit/ -v

# 运行集成测试
test-integration:
	python -m pytest testing/integration/ -v

# 训练模型
train:
	cd training && python train_model.py

# 启动API服务
api:
	cd fastapi && python main.py

# 下载数据
download:
	cd data/raw && python download.py

# 生成特征
features:
	cd data/processed && python feature_engineering.py

# 完整的数据处理流程
data-pipeline: download features

# 完整的训练流程
train-pipeline: data-pipeline train

# 清理临时文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# 生成文档
docs:
	@echo "文档生成功能待实现"

# 代码格式化
format:
	black training/ fastapi/ testing/ data/
	isort training/ fastapi/ testing/ data/

# 代码检查
lint:
	flake8 training/ fastapi/ testing/ data/
	mypy training/ fastapi/

# 安全检查
security:
	safety check

# 构建Docker镜像
docker-build:
	docker build -t coinguard:latest .

# 运行Docker容器
docker-run:
	docker run -p 8000:8000 coinguard:latest

# 部署到生产环境
deploy:
	@echo "部署功能待实现"

# 监控服务状态
status:
	@echo "检查服务状态..."
	@curl -s http://localhost:8000/health || echo "API服务未运行"

# 备份数据
backup:
	mkdir -p backups
	tar -czf backups/data_$(shell date +%Y%m%d_%H%M%S).tar.gz data/

# 恢复数据
restore:
	@echo "请指定备份文件: make restore BACKUP_FILE=backups/data_20231201_100000.tar.gz"
	@if [ -n "$(BACKUP_FILE)" ]; then \
		tar -xzf $(BACKUP_FILE); \
		echo "数据恢复完成"; \
	else \
		echo "请指定BACKUP_FILE参数"; \
	fi
