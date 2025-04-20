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
    if is_emr_cluster_mode():
        builder = builder.master("yarn")
    else:
        builder = builder.master("local[*]")

    # 添加S3访问配置
    spark = builder \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "16g") \
        .config("spark.driver.maxResultSize", "8g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.yarn.executor.memoryOverhead", "4g") \
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


def preprocess_data(games_df, users_df, recommendations_df, metadata_df, spark=None):
    """预处理数据，合并游戏元数据和转换评分"""
    print("预处理数据...")

    # 处理游戏数据与元数据
    if not games_df.empty and not metadata_df.empty:
        # 合并游戏数据和元数据
        games_with_metadata = pd.merge(
            games_df,
            metadata_df[['app_id', 'description', 'tags']],
            on='app_id',
            how='left'
        )

        # 填充缺失值
        games_with_metadata['description'] = games_with_metadata['description'].fillna('')

        # 增加批处理逻辑，分段处理tags
        batch_size = 10000
        total_batches = len(games_with_metadata) // batch_size + 1

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(games_with_metadata))
            print(f"处理标签批次 {i + 1}/{total_batches} ({start_idx}:{end_idx})")

            # 只处理当前批次
            games_with_metadata.loc[start_idx:end_idx, 'tags'] = \
                games_with_metadata.loc[start_idx:end_idx, 'tags'].apply(process_tags)

        print(f"处理后的游戏数据: {len(games_with_metadata)} 条记录")
    else:
        games_with_metadata = games_df.copy() if not games_df.empty else pd.DataFrame()
        print("警告: 游戏数据或元数据为空")

    # 处理推荐数据 - 同样分批处理
    if not recommendations_df.empty:
        # 将is_recommended和hours转换为评分
        if 'rating' not in recommendations_df.columns:
            # 分批处理评分
            batch_size = 1000000  # 100万条记录一批
            total_batches = len(recommendations_df) // batch_size + 1

            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(recommendations_df))
                print(f"处理评分批次 {i + 1}/{total_batches} ({start_idx}:{end_idx})")

                recommendations_df.loc[start_idx:end_idx, 'rating'] = \
                    recommendations_df.loc[start_idx:end_idx].apply(
                        lambda row: min(10.0, row['hours'] / 10) * (1.5 if row['is_recommended'] else 0.5),
                        axis=1
                    )

        # 处理后的推荐数据
        processed_recommendations = recommendations_df.copy()
        print(f"处理后的推荐数据: {len(processed_recommendations)} 条记录")
    else:
        processed_recommendations = pd.DataFrame()
        print("警告: 推荐数据为空")

    # 如果提供了Spark会话，转换为Spark DataFrame
    spark_ratings = None
    if spark is not None and not processed_recommendations.empty:
        try:
            # 创建Spark DataFrame - 如果数据较大，分批转换
            if len(processed_recommendations) > 10000000:  # 1000万条记录以上
                # 分批转换为Spark DataFrame
                print("分批转换为Spark DataFrame...")
                batch_size = 2000000  # 500万条记录一批
                total_batches = len(processed_recommendations) // batch_size + 1

                for i in range(total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(processed_recommendations))
                    print(f"转换批次 {i + 1}/{total_batches} ({start_idx}:{end_idx})")

                    batch_df = processed_recommendations.iloc[start_idx:end_idx]
                    batch_spark = spark.createDataFrame(
                        batch_df[['user_id', 'app_id', 'rating']]
                    )

                    if spark_ratings is None:
                        spark_ratings = batch_spark
                    else:
                        spark_ratings = spark_ratings.union(batch_spark)

                print(f"创建Spark评分数据: {spark_ratings.count()} 条记录")
            else:
                # 直接转换
                spark_ratings = spark.createDataFrame(
                    processed_recommendations[['user_id', 'app_id', 'rating']]
                )
                print(f"创建Spark评分数据: {spark_ratings.count()} 条记录")
        except Exception as e:
            print(f"创建Spark DataFrame时出错: {e}")

    return games_with_metadata, spark_ratings, processed_recommendations


def process_tags(x):
    # 首先，如果输入是数组或Series，直接转换为列表并返回第一个元素（如果有）
    if hasattr(x, '__iter__') and not isinstance(x, (str, dict)):
        if hasattr(x, 'tolist'):  # numpy数组转换方法
            items = x.tolist()
        else:
            items = list(x)
        # 返回第一个元素，或空列表
        return items[0] if items else []

        # 处理单个值
    if pd.isna(x) or x is None:
        return []

    # 若已经是列表，直接返回
    if isinstance(x, list):
        return x

    # 若是字符串且可能是JSON，尝试解析
    if isinstance(x, str):
        if x.strip() == '':
            return []
        try:
            parsed = json.loads(x.strip())
            return parsed if isinstance(parsed, list) else []
        except:
            # 不是有效的JSON，尝试将其作为单个标签
            return [x]

    # 其他情况返回空列表
    return []

def split_data(data, test_ratio=0.2, random_state=42):
    """将数据分割为训练集和测试集"""
    np.random.seed(random_state)

    # 按用户划分，确保每个用户的数据不会同时出现在训练集和测试集中
    user_ids = data['user_id'].unique()
    test_users = np.random.choice(
        user_ids,
        size=int(len(user_ids) * test_ratio),
        replace=False
    )

    # 划分数据
    test_data = data[data['user_id'].isin(test_users)]
    train_data = data[~data['user_id'].isin(test_users)]

    print(f"训练集: {len(train_data)} 条记录, 测试集: {len(test_data)} 条记录")
    return train_data, test_data