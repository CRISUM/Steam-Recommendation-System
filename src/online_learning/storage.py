# src/online_learning/storage.py
"""
模型和指标存储管理。
"""

import os
import json
import time
import pickle
import logging
import numpy as np
import boto3
import shutil
from pathlib import Path
from datetime import datetime

from . import METRICS_FILE, MODEL_CHECKPOINT_DIR
from src.utils.aws_utils import get_storage_path, is_emr_cluster_mode

# 设置日志
logger = logging.getLogger("online_learning.storage")


def get_adjusted_path(path):
    """获取根据环境调整的路径"""
    if is_emr_cluster_mode():
        return get_storage_path(path)
    return path


def save_metrics(metrics, bucket_name="steam-project-data-976193243904"):
    """
    保存指标到本地文件和S3

    Args:
        metrics (dict): 性能指标字典
        bucket_name (str): S3存储桶名称
    """
    # 获取适当的路径
    metrics_file_path = get_adjusted_path(METRICS_FILE)

    # 添加时间戳（如果没有）
    if 'timestamp' not in metrics:
        metrics['timestamp'] = datetime.now().isoformat()

    # 处理本地或S3路径
    if metrics_file_path.startswith("s3://"):
        # S3路径处理
        # 先从S3下载现有数据（如果有）
        s3_path = metrics_file_path.replace("s3://", "")
        s3_bucket = s3_path.split("/")[0]
        s3_key = "/".join(s3_path.split("/")[1:])

        existing_metrics = []
        temp_file = "temp_metrics.json"

        try:
            # 尝试从S3下载现有文件
            s3_client = boto3.client('s3')
            s3_client.download_file(s3_bucket, s3_key, temp_file)

            # 读取现有数据
            with open(temp_file, 'r') as f:
                existing_metrics = json.load(f)
        except Exception as e:
            logger.info(f"无法从S3下载现有指标文件 (可能不存在): {e}")

        # 确保是列表
        if not isinstance(existing_metrics, list):
            existing_metrics = []

        # 添加新指标
        existing_metrics.append(metrics)

        # 限制历史记录大小
        if len(existing_metrics) > 100:
            existing_metrics = existing_metrics[-100:]

        # 保存到临时文件
        with open(temp_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)

        # 上传到S3
        try:
            s3_client.upload_file(temp_file, s3_bucket, s3_key)
            logger.info(f"指标已上传到S3: {metrics_file_path}")

            # 删除临时文件
            os.remove(temp_file)
        except Exception as e:
            logger.error(f"上传指标到S3出错: {e}")
    else:
        # 本地文件处理
        # 确保目录存在
        Path(os.path.dirname(metrics_file_path)).mkdir(parents=True, exist_ok=True)

        # 从文件加载现有指标
        existing_metrics = []
        if os.path.exists(metrics_file_path):
            try:
                with open(metrics_file_path, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.error(f"读取指标文件时出错: {e}")

        # 确保是列表
        if not isinstance(existing_metrics, list):
            existing_metrics = []

        # 添加新指标
        existing_metrics.append(metrics)

        # 限制历史记录大小
        if len(existing_metrics) > 100:
            existing_metrics = existing_metrics[-100:]

        # 保存到文件
        try:
            with open(metrics_file_path, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            logger.info(f"指标已保存到 {metrics_file_path}")
        except Exception as e:
            logger.error(f"保存指标到文件时出错: {e}")


def save_model_checkpoint(model_type, model_obj, metrics, bucket_name="steam-project-data-976193243904"):
    """
    保存模型检查点到本地和S3

    Args:
        model_type (str): 模型类型，'als'或'tfidf'
        model_obj: 模型对象
        metrics (dict): 性能指标字典
        bucket_name (str): S3存储桶名称
    """
    # 获取适当的路径
    checkpoint_dir_base = get_adjusted_path(MODEL_CHECKPOINT_DIR)
    checkpoint_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if checkpoint_dir_base.startswith("s3://"):
        # S3路径处理
        # 先在本地创建临时目录
        temp_checkpoint_dir = f"temp_{model_type}_{checkpoint_timestamp}"
        os.makedirs(temp_checkpoint_dir, exist_ok=True)

        # 保存模型到临时目录
        if model_type == "als" and model_obj is not None:
            try:
                model_obj.save(f"{temp_checkpoint_dir}/model")
                logger.info(f"ALS模型已临时保存到 {temp_checkpoint_dir}")
            except Exception as e:
                logger.error(f"保存ALS模型到临时目录出错: {e}")

        elif model_type == "tfidf" and model_obj is not None:
            try:
                # 保存TF-IDF向量化器
                with open(f"{temp_checkpoint_dir}/tfidf_vectorizer.pkl", "wb") as f:
                    pickle.dump(model_obj, f)

                # 保存相似度矩阵和索引
                from .models import cosine_sim, indices
                if cosine_sim is not None:
                    np.save(f"{temp_checkpoint_dir}/cosine_sim.npy", cosine_sim)

                if indices is not None:
                    with open(f"{temp_checkpoint_dir}/indices.pkl", "wb") as f:
                        pickle.dump(indices, f)

                logger.info(f"TF-IDF模型已临时保存到 {temp_checkpoint_dir}")
            except Exception as e:
                logger.error(f"保存TF-IDF模型到临时目录出错: {e}")

        # 保存指标
        with open(f"{temp_checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 解析S3路径
        s3_path = checkpoint_dir_base.replace("s3://", "")
        s3_bucket = s3_path.split("/")[0]
        s3_prefix = "/".join(s3_path.split("/")[1:]) if len(s3_path.split("/")) > 1 else ""

        # 构建完整的S3目标路径
        if s3_prefix:
            s3_target_prefix = f"{s3_prefix}/{model_type}_{checkpoint_timestamp}"
        else:
            s3_target_prefix = f"{model_type}_{checkpoint_timestamp}"

        # 上传到S3
        try:
            s3_client = boto3.client('s3')
            for root, dirs, files in os.walk(temp_checkpoint_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_checkpoint_dir)
                    s3_key = f"{s3_target_prefix}/{relative_path}"

                    try:
                        s3_client.upload_file(local_path, s3_bucket, s3_key)
                    except Exception as e:
                        logger.error(f"上传文件 {local_path} 到S3出错: {e}")

            logger.info(f"检查点已上传到S3 {checkpoint_dir_base}/{model_type}_{checkpoint_timestamp}")

            # 清理临时目录
            shutil.rmtree(temp_checkpoint_dir)
        except Exception as e:
            logger.error(f"上传检查点到S3出错: {e}")
            # 尝试清理临时目录
            if os.path.exists(temp_checkpoint_dir):
                shutil.rmtree(temp_checkpoint_dir)
    else:
        # 本地保存处理
        # 创建检查点目录
        checkpoint_dir = f"{checkpoint_dir_base}/{model_type}_{checkpoint_timestamp}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 保存模型
        if model_type == "als" and model_obj is not None:
            try:
                model_obj.save(f"{checkpoint_dir}/model")
                logger.info(f"ALS模型已保存到 {checkpoint_dir}")
            except Exception as e:
                logger.error(f"保存ALS模型出错: {e}")

        elif model_type == "tfidf" and model_obj is not None:
            try:
                # 保存TF-IDF向量化器
                with open(f"{checkpoint_dir}/tfidf_vectorizer.pkl", "wb") as f:
                    pickle.dump(model_obj, f)

                # 保存相似度矩阵和索引
                from .models import cosine_sim, indices
                if cosine_sim is not None:
                    np.save(f"{checkpoint_dir}/cosine_sim.npy", cosine_sim)

                if indices is not None:
                    with open(f"{checkpoint_dir}/indices.pkl", "wb") as f:
                        pickle.dump(indices, f)

                logger.info(f"TF-IDF模型已保存到 {checkpoint_dir}")
            except Exception as e:
                logger.error(f"保存TF-IDF模型出错: {e}")

        # 保存指标
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


def load_model_from_s3(s3_path, local_path="temp_model", bucket_name="steam-project-data-976193243904"):
    """
    从S3加载模型

    Args:
        s3_path (str): S3中的模型路径
        local_path (str): 模型临时存储的本地路径
        bucket_name (str): S3存储桶名称

    Returns:
        str: 本地模型路径
    """
    # 解析S3路径
    if s3_path.startswith('s3://'):
        parts = s3_path.replace('s3://', '').split('/', 1)
        if len(parts) > 1:
            bucket_name = parts[0]
            key_prefix = parts[1]
        else:
            key_prefix = ""
    else:
        key_prefix = s3_path

    # 创建临时目录
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path)

    logger.info(f"从S3 {bucket_name}/{key_prefix} 下载模型...")

    try:
        # 下载模型文件
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')

        download_count = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=key_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # 计算本地文件路径
                    rel_path = os.path.relpath(key, key_prefix) if key_prefix else key
                    local_file = os.path.join(local_path, rel_path)

                    # 创建目录
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    # 下载文件
                    s3_client.download_file(bucket_name, key, local_file)
                    download_count += 1

        logger.info(f"从S3下载了 {download_count} 个模型文件到 {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"从S3加载模型出错: {e}")
        raise


def clean_old_checkpoints(max_checkpoints=5):
    """
    清理旧的检查点，保留最新的N个

    Args:
        max_checkpoints (int): 每种类型要保留的最大检查点数
    """
    # 获取调整后的路径
    checkpoint_dir_path = get_adjusted_path(MODEL_CHECKPOINT_DIR)

    # 如果是S3路径，跳过清理（S3存储成本低，可以保留更多历史）
    if checkpoint_dir_path.startswith("s3://"):
        logger.info("检查点存储在S3，跳过本地清理")
        return

    try:
        # 获取检查点目录
        checkpoint_dir = Path(checkpoint_dir_path)
        if not checkpoint_dir.exists():
            return

        # 按类型分组检查点
        als_checkpoints = []
        tfidf_checkpoints = []

        for item in checkpoint_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name.startswith("als_"):
                als_checkpoints.append(item)
            elif item.name.startswith("tfidf_"):
                tfidf_checkpoints.append(item)

        # 按修改时间排序
        als_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        tfidf_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # 删除过多的检查点
        for old_cp in als_checkpoints[max_checkpoints:]:
            logger.info(f"删除旧的ALS检查点: {old_cp}")
            shutil.rmtree(old_cp)

        for old_cp in tfidf_checkpoints[max_checkpoints:]:
            logger.info(f"删除旧的TF-IDF检查点: {old_cp}")
            shutil.rmtree(old_cp)

    except Exception as e:
        logger.error(f"清理旧检查点时出错: {e}")