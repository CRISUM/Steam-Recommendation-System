# src/online_learning/main.py
"""
在线学习系统的主程序入口。
"""

import argparse
import logging
import os
import sys

from . import logger
from .api import app
from .models import initialize_models
from .updater import start_update_thread, stop_update_thread, is_update_thread_running


def start_flask_server(port=5000, host='0.0.0.0', data_path='data', model_path='models/als_model'):
    """
    启动Flask API服务器

    Args:
        port (int): 服务器端口号
        host (str): 服务器主机地址
        data_path (str): 数据目录路径
        model_path (str): 模型文件路径
    """
    logger.info(f"正在启动在线学习服务 (端口: {port})...")

    # 初始化模型
    initialize_models(data_path, model_path)

    # 启动更新线程
    if not is_update_thread_running():
        start_update_thread()

    # 启动Flask服务
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务...")
        stop_update_thread()
        logger.info("服务已关闭")
    except Exception as e:
        logger.error(f"启动服务时出错: {e}")
        stop_update_thread()
        sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Steam游戏推荐系统在线学习服务")
    parser.add_argument('--port', type=int, default=5000, help='API服务端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API服务主机地址')
    parser.add_argument('--data-path', type=str, default='data', help='数据目录路径')
    parser.add_argument('--model-path', type=str, default='models/als_model', help='模型文件路径')
    parser.add_argument('--use-s3', action='store_true', help='从S3加载数据和模型')

    args = parser.parse_args()

    # 如果使用S3，调整路径
    if args.use_s3:
        if not args.data_path.startswith('s3a://'):
            args.data_path = f"steam-project-data-976193243904/{args.data_path.lstrip('/')}"
        if not args.model_path.startswith('s3a://'):
            args.model_path = f"steam-project-data-976193243904/{args.model_path.lstrip('/')}"

    return args


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()

    # 显示启动信息
    logger.info("=== Steam游戏推荐系统在线学习服务 ===")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"模型路径: {args.model_path}")

    # 启动服务器
    start_flask_server(
        port=args.port,
        host=args.host,
        data_path=args.data_path,
        model_path=args.model_path
    )