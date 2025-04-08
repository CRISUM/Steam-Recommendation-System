# src/data_processing.py
import pandas as pd
import numpy as np
import json
from pyspark.sql import SparkSession


def initialize_spark(app_name="SteamRecommendationSystem"):
    """初始化Spark会话"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark


def load_data(data_path):
    """加载CSV和JSON数据"""
    print(f"正在从 {data_path} 加载数据...")

    # 读取CSV文件
    games_df = pd.read_csv(f"{data_path}/games.csv")
    users_df = pd.read_csv(f"{data_path}/users.csv")
    recommendations_df = pd.read_csv(f"{data_path}/recommendations.csv")

    print(f"已加载 {len(games_df)} 个游戏, {len(users_df)} 个用户, {len(recommendations_df)} 条评价")

    # 读取JSON元数据
    metadata_list = []
    try:
        with open(f"{data_path}/games_metadata.json", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metadata_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行时出错: {e}")
                        continue
    except Exception as e:
        print(f"读取元数据文件时出错: {e}")

    metadata_df = pd.DataFrame(metadata_list)
    print(f"已加载 {len(metadata_df)} 条游戏元数据")

    return games_df, users_df, recommendations_df, metadata_df


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
    games_with_metadata['description'] = games_with_metadata['description'].fillna('')

    # 处理标签数据
    games_with_metadata['tags'] = games_with_metadata['tags'].apply(
        lambda x: [] if pd.isna(x) else x
    )

    # 创建用于协同过滤的评分数据
    # 将游戏时长和是否推荐转换为评分
    recommendations_df['rating'] = recommendations_df.apply(
        lambda row: min(10.0, row['hours'] / 10) * (1.5 if row['is_recommended'] else 0.5),
        axis=1
    )

    # 转换为Spark DataFrame（如果提供了Spark会话）
    if spark is not None:
        spark_ratings = spark.createDataFrame(
            recommendations_df[['user_id', 'app_id', 'rating']]
        )
    else:
        spark_ratings = None
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