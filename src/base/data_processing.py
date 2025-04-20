# src/data_processing.py
import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
import os
import io
import boto3
from src.utils.aws_utils import is_emr_cluster_mode


def initialize_spark(app_name="SteamRecommendationSystem"):
    """初始化Spark会话，适用于本地和集群环境"""
    # 创建builder对象
    builder = SparkSession.builder.appName(app_name)

    # 检查是否在EMR集群上运行 - 在集群模式下不要指定master
    if not is_emr_cluster_mode():
        builder = builder.master("local[*]")

    # 添加S3访问配置
    spark = builder \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    return spark


def load_data_from_s3(bucket_name, prefix):
    """从S3加载数据文件"""
    print(f"正在从S3 {bucket_name}/{prefix} 加载数据...")

    s3_client = boto3.client('s3')

    # 读取CSV文件
    try:
        # 加载游戏数据
        games_obj = s3_client.get_object(Bucket=bucket_name, Key=f"{prefix}games.csv")
        games_df = pd.read_csv(io.BytesIO(games_obj['Body'].read()))
        print(f"已从S3加载 {len(games_df)} 个游戏")
    except Exception as e:
        print(f"从S3加载游戏数据时出错: {e}")
        games_df = pd.DataFrame()

    try:
        # 加载用户数据
        users_obj = s3_client.get_object(Bucket=bucket_name, Key=f"{prefix}users.csv")
        users_df = pd.read_csv(io.BytesIO(users_obj['Body'].read()))
        print(f"已从S3加载 {len(users_df)} 个用户")
    except Exception as e:
        print(f"从S3加载用户数据时出错: {e}")
        users_df = pd.DataFrame()

    try:
        # 加载评价数据
        recommendations_obj = s3_client.get_object(Bucket=bucket_name, Key=f"{prefix}recommendations.csv")
        recommendations_df = pd.read_csv(io.BytesIO(recommendations_obj['Body'].read()))
        print(f"已从S3加载 {len(recommendations_df)} 条评价")
    except Exception as e:
        print(f"从S3加载评价数据时出错: {e}")
        recommendations_df = pd.DataFrame()

    # 读取JSON元数据
    try:
        metadata_obj = s3_client.get_object(Bucket=bucket_name, Key=f"{prefix}games_metadata.json")
        metadata_content = metadata_obj['Body'].read().decode('utf-8')
        metadata_list = []

        for line in metadata_content.split("\n"):
            line = line.strip()
            if line:
                try:
                    metadata_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        metadata_df = pd.DataFrame(metadata_list)
        print(f"已从S3加载 {len(metadata_df)} 条游戏元数据")
    except Exception as e:
        print(f"从S3加载元数据时出错: {e}")
        metadata_df = pd.DataFrame()

    print(f"从S3数据加载完成")
    return games_df, users_df, recommendations_df, metadata_df


def load_data(data_path=None):
    """加载CSV和JSON数据，支持从本地或S3加载"""

    # 检查是否需要从S3加载
    if data_path and data_path.startswith('s3://'):
        # 解析S3路径
        bucket_name = data_path.replace('s3://', '').split('/')[0]
        prefix = '/'.join(data_path.replace('s3://', '').split('/')[1:])
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        # 从S3加载
        return load_data_from_s3(bucket_name, prefix)

    elif data_path:
        # 从本地加载（保留原有逻辑）
        print(f"正在从 {data_path} 加载数据...")

        # 读取CSV文件
        games_df = pd.read_csv(f"{data_path}/games.csv")
        users_df = pd.read_csv(f"{data_path}/users.csv")
        recommendations_df = pd.read_csv(f"{data_path}/recommendations.csv")

        # 读取JSON元数据
        metadata_list = []
        with open(f"{data_path}/games_metadata.json", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    metadata_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        metadata_df = pd.DataFrame(metadata_list)

        print(f"已加载 {len(games_df)} 个游戏, {len(users_df)} 个用户, "
              f"{len(recommendations_df)} 条评价, {len(metadata_df)} 条游戏元数据")

        return games_df, users_df, recommendations_df, metadata_df