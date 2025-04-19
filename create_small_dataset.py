# create_small_dataset.py
import pandas as pd
import numpy as np
import os
import json
import shutil
import boto3
import io
from src.utils.aws_utils import get_storage_path, is_emr_cluster_mode


def create_small_dataset(original_data_path="data", small_data_path="data_small",
                         n_games=1000, n_users=500, n_ratings=5000):
    """从原始数据集创建一个小型测试数据集"""
    print(f"创建小型测试数据集: {n_games} 个游戏, {n_users} 个用户, {n_ratings} 条评价")

    # 检查是否在EMR集群上运行
    cluster_mode = is_emr_cluster_mode()

    # 调整路径为S3路径（如果在集群上）
    if cluster_mode:
        original_data_path = get_storage_path(original_data_path)
        small_data_path = get_storage_path(small_data_path)

    # 创建目录（仅本地模式）
    if not cluster_mode:
        os.makedirs(small_data_path, exist_ok=True)

    # 检查原始数据是否存在
    if not cluster_mode and not os.path.exists(original_data_path):
        print(f"错误: 原始数据路径 {original_data_path} 不存在")
        return

    try:
        # 根据环境加载数据
        if original_data_path.startswith("s3://"):
            # 从S3加载
            s3_bucket = original_data_path.replace("s3://", "").split("/")[0]
            s3_prefix = "/".join(original_data_path.replace("s3://", "").split("/")[1:])
            if s3_prefix and not s3_prefix.endswith("/"):
                s3_prefix += "/"

            s3_client = boto3.client('s3')

            # 加载游戏数据
            try:
                games_obj = s3_client.get_object(Bucket=s3_bucket, Key=f"{s3_prefix}games.csv")
                games_df = pd.read_csv(io.BytesIO(games_obj['Body'].read()))
                print(f"从S3加载了 {len(games_df)} 个游戏")
            except Exception as e:
                print(f"从S3加载游戏数据时出错: {e}")
                games_df = pd.DataFrame()

            # 加载用户数据
            try:
                users_obj = s3_client.get_object(Bucket=s3_bucket, Key=f"{s3_prefix}users.csv")
                users_df = pd.read_csv(io.BytesIO(users_obj['Body'].read()))
                print(f"从S3加载了 {len(users_df)} 个用户")
            except Exception as e:
                print(f"从S3加载用户数据时出错: {e}")
                users_df = pd.DataFrame()

            # 加载评价数据
            try:
                recommendations_obj = s3_client.get_object(Bucket=s3_bucket, Key=f"{s3_prefix}recommendations.csv")
                recommendations_df = pd.read_csv(io.BytesIO(recommendations_obj['Body'].read()))
                print(f"从S3加载了 {len(recommendations_df)} 条评价")
            except Exception as e:
                print(f"从S3加载评价数据时出错: {e}")
                recommendations_df = pd.DataFrame()

            # 加载元数据
            try:
                metadata_obj = s3_client.get_object(Bucket=s3_bucket, Key=f"{s3_prefix}games_metadata.json")
                metadata_content = metadata_obj['Body'].read().decode('utf-8')
                metadata_list = []

                for line in metadata_content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            metadata_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                print(f"从S3加载了 {len(metadata_list)} 条游戏元数据")
            except Exception as e:
                print(f"从S3加载元数据时出错: {e}")
                metadata_list = []
        else:
            # 从本地加载
            # 加载游戏数据并取子集
            games_file = f"{original_data_path}/games.csv"
            if os.path.exists(games_file):
                games_df = pd.read_csv(games_file)
                print(f"从本地加载了 {len(games_df)} 个游戏")
            else:
                print(f"警告: 找不到游戏数据文件 {games_file}")
                games_df = pd.DataFrame()

            # 加载用户数据并取子集
            users_file = f"{original_data_path}/users.csv"
            if os.path.exists(users_file):
                users_df = pd.read_csv(users_file)
                print(f"从本地加载了 {len(users_df)} 个用户")
            else:
                print(f"警告: 找不到用户数据文件 {users_file}")
                users_df = pd.DataFrame()

            # 加载评价数据
            recommendations_file = f"{original_data_path}/recommendations.csv"
            if os.path.exists(recommendations_file):
                recommendations_df = pd.read_csv(recommendations_file)
                print(f"从本地加载了 {len(recommendations_df)} 条评价")
            else:
                print(f"警告: 找不到评价数据文件 {recommendations_file}")
                recommendations_df = pd.DataFrame()

            # 加载元数据
            metadata_file = f"{original_data_path}/games_metadata.json"
            metadata_list = []

            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            metadata_list.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                print(f"从本地加载了 {len(metadata_list)} 条游戏元数据")
            else:
                print(f"警告: 找不到元数据文件 {metadata_file}")

        # 处理数据 - 创建小样本
        # 游戏样本
        if not games_df.empty:
            games_sample = games_df.sample(min(n_games, len(games_df)), random_state=42)
        else:
            games_sample = pd.DataFrame()

        # 用户样本
        if not users_df.empty:
            users_sample = users_df.sample(min(n_users, len(users_df)), random_state=42)
        else:
            users_sample = pd.DataFrame()

        # 评价样本
        if not recommendations_df.empty and not games_sample.empty and not users_sample.empty:
            # 过滤出我们样本中的游戏和用户
            filtered_recommendations = recommendations_df[
                recommendations_df['app_id'].isin(games_sample['app_id']) &
                recommendations_df['user_id'].isin(users_sample['user_id'])
                ]

            # 如果过滤后的数据不足，则直接从原始数据中随机抽样
            if len(filtered_recommendations) < n_ratings:
                recommendations_sample = recommendations_df.sample(
                    min(n_ratings, len(recommendations_df)),
                    random_state=42
                )
            else:
                recommendations_sample = filtered_recommendations.sample(
                    min(n_ratings, len(filtered_recommendations)),
                    random_state=42
                )
        else:
            recommendations_sample = pd.DataFrame()

        # 元数据样本
        if metadata_list and not games_sample.empty:
            game_ids = set(games_sample['app_id'])
            selected_metadata = []

            for item in metadata_list:
                if item.get('app_id') in game_ids:
                    selected_metadata.append(item)
        else:
            selected_metadata = []

        # 保存处理后的数据
        if small_data_path.startswith("s3://"):
            # 保存到S3
            s3_bucket = small_data_path.replace("s3://", "").split("/")[0]
            s3_prefix = "/".join(small_data_path.replace("s3://", "").split("/")[1:])
            if s3_prefix and not s3_prefix.endswith("/"):
                s3_prefix += "/"

            # 保存游戏数据
            if not games_sample.empty:
                csv_buffer = io.StringIO()
                games_sample.to_csv(csv_buffer, index=False)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{s3_prefix}games.csv",
                    Body=csv_buffer.getvalue()
                )
                print(f"已保存 {len(games_sample)} 个游戏样本到S3")

            # 保存用户数据
            if not users_sample.empty:
                csv_buffer = io.StringIO()
                users_sample.to_csv(csv_buffer, index=False)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{s3_prefix}users.csv",
                    Body=csv_buffer.getvalue()
                )
                print(f"已保存 {len(users_sample)} 个用户样本到S3")

            # 保存评价数据
            if not recommendations_sample.empty:
                csv_buffer = io.StringIO()
                recommendations_sample.to_csv(csv_buffer, index=False)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{s3_prefix}recommendations.csv",
                    Body=csv_buffer.getvalue()
                )
                print(f"已保存 {len(recommendations_sample)} 条评价样本到S3")

            # 保存元数据
            if selected_metadata:
                json_lines = "\n".join([json.dumps(item) for item in selected_metadata])
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=f"{s3_prefix}games_metadata.json",
                    Body=json_lines
                )
                print(f"已保存 {len(selected_metadata)} 条游戏元数据到S3")
        else:
            # 保存到本地文件系统
            # 保存游戏数据
            if not games_sample.empty:
                games_sample.to_csv(f"{small_data_path}/games.csv", index=False)
                print(f"已保存 {len(games_sample)} 个游戏样本")

            # 保存用户数据
            if not users_sample.empty:
                users_sample.to_csv(f"{small_data_path}/users.csv", index=False)
                print(f"已保存 {len(users_sample)} 个用户样本")

            # 保存评价数据
            if not recommendations_sample.empty:
                recommendations_sample.to_csv(f"{small_data_path}/recommendations.csv", index=False)
                print(f"已保存 {len(recommendations_sample)} 条评价样本")

            # 保存元数据
            if selected_metadata:
                with open(f"{small_data_path}/games_metadata.json", 'w', encoding='utf-8') as f:
                    for item in selected_metadata:
                        f.write(json.dumps(item) + '\n')
                print(f"已保存 {len(selected_metadata)} 条游戏元数据")

        print(f"小型测试数据集创建完成，保存在 {small_data_path}")

    except Exception as e:
        print(f"创建小型数据集时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_small_dataset()