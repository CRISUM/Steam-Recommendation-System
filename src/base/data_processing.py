# src/data_processing.py
import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession
import os
import io
import boto3

# 修改 initialize_spark 函数
def initialize_spark(app_name="SteamRecommendationSystem"):
    """初始化Spark会话"""
    spark = SparkSession.builder \
        .appName(app_name) \
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
        # 原有的本地文件加载代码...

    else:
        # 默认从S3加载当前项目的数据
        return load_data_from_s3("steam-project-data", "")


def load_data_from_s3(bucket_name, prefix=""):
    """从S3加载数据"""
    s3 = boto3.client('s3')
    print(f"正在从S3桶 {bucket_name}/{prefix} 加载数据...")

    # 读取CSV文件
    games_obj = s3.get_object(Bucket=bucket_name, Key=f"{prefix}games.csv")
    games_df = pd.read_csv(io.BytesIO(games_obj['Body'].read()))

    users_obj = s3.get_object(Bucket=bucket_name, Key=f"{prefix}users.csv")
    users_df = pd.read_csv(io.BytesIO(users_obj['Body'].read()))

    recommendations_obj = s3.get_object(Bucket=bucket_name, Key=f"{prefix}recommendations.csv")
    recommendations_df = pd.read_csv(io.BytesIO(recommendations_obj['Body'].read()))

    # 读取JSON元数据
    metadata_list = []
    metadata_obj = s3.get_object(Bucket=bucket_name, Key=f"{prefix}games_metadata.json")
    content = metadata_obj['Body'].read().decode('utf-8')

    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                metadata_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"解析JSON行时出错: {e}")
                continue

    metadata_df = pd.DataFrame(metadata_list)

    print(f"已加载 {len(games_df)} 个游戏, {len(users_df)} 个用户, "
          f"{len(recommendations_df)} 条评价, {len(metadata_df)} 条游戏元数据")

    return games_df, users_df, recommendations_df, metadata_df

def process_tags(x):
    if isinstance(x, list):
        return x
    else:
        return []

def preprocess_data(games_df, users_df, recommendations_df, metadata_df, spark=None):
    """数据预处理"""
    print("开始数据预处理...")

    # 数据类型转换
    recommendations_df['hours'] = recommendations_df['hours'].astype(float)

    # 将字符串转换为布尔值（如果需要）
    if recommendations_df['is_recommended'].dtype == 'object':
        recommendations_df['is_recommended'] = recommendations_df['is_recommended'].map(
            {'TRUE': True, 'FALSE': False}
        )

    # 合并游戏信息和元数据
    games_with_metadata = pd.merge(games_df, metadata_df, on='app_id', how='left')

    # 处理缺失的描述
    if 'description' in games_with_metadata.columns:
        games_with_metadata['description'] = games_with_metadata['description'].fillna('')
    else:
        games_with_metadata['description'] = ''

    # 处理标签数据
    def process_tags(x):
        if isinstance(x, list):
            return x
        elif pd.isna(x) or x is None:
            return []
        else:
            return []

    # 如果tags列存在，则处理它；如果不存在，则创建一个空列表列
    if 'tags' in games_with_metadata.columns:
        games_with_metadata['tags'] = games_with_metadata['tags'].apply(process_tags)
    else:
        games_with_metadata['tags'] = [[] for _ in range(len(games_with_metadata))]

    # 创建用于协同过滤的评分数据
    # 将游戏时长和是否推荐转换为评分
    recommendations_df['rating'] = recommendations_df.apply(
        lambda row: min(10.0, row['hours'] / 10) * (1.5 if row['is_recommended'] else 0.5),
        axis=1
    )

    # 转换为Spark DataFrame（如果提供了Spark会话）
    spark_ratings = None
    if spark is not None:
        try:
            # 尝试创建Spark DataFrame
            spark_ratings = spark.createDataFrame(
                recommendations_df[['user_id', 'app_id', 'rating']]
            )
        except Exception as e:
            print(f"创建Spark DataFrame时出错: {e}")
            print("将使用None代替Spark DataFrame")
    else:
        print("未提供Spark会话，跳过Spark DataFrame转换")

    print("数据预处理完成")
    return games_with_metadata, spark_ratings, recommendations_df

def split_data(data, test_ratio=0.2, random_state=42):
    """将数据分割为训练集和测试集"""
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        data, test_size=test_ratio, random_state=random_state
    )
    return train_data, test_data