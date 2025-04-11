# src/online_learning/__init__.py
"""
在线学习模块用于接收新数据并增量更新推荐模型。

这个模块提供了以下功能:
1. 数据接收API端点
2. 数据缓冲区管理
3. 增量更新ALS协同过滤模型
4. 增量更新TF-IDF内容模型
5. 性能指标记录和分析
"""

import logging
import os
from pathlib import Path

# 创建必要的目录
Path("results").mkdir(parents=True, exist_ok=True)
Path("models/online").mkdir(parents=True, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("online_learning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("online_learning")

# 配置常量
METRICS_FILE = 'results/online_metrics.json'
UPDATE_INTERVAL = 300  # 5分钟更新一次模型
BUFFER_SIZE_THRESHOLD = 100  # 当缓冲区达到100条记录时触发更新
MODEL_CHECKPOINT_DIR = 'models/online'
CF_WEIGHT = 0.7  # 协同过滤权重

# 版本信息
__version__ = '1.0.0'