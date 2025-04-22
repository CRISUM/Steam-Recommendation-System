# create_small_dataset_s.py
import pandas as pd
import numpy as np
import os
import json
import boto3
import io
from src.utils.aws_utils import get_storage_path, is_emr_cluster_mode


def create_small_dataset_s(original_data_path="s3://steam-project-data-976193243904",
                           small_data_path="s3://steam-project-data-976193243904/data_small_s",
                           n_games=200, n_users=100, n_ratings=1000):
    """从原始数据集创建一个小型测试数据集 - 减少版"""
    print(f"创建极小型训练数据集: {n_games} 个游戏, {n_users} 个用户, {n_ratings} 条评价")

    # 检查是否在EMR集群上运行
    cluster_mode = is_emr_cluster_mode()

    # 确保路径格式正确
    if not original_data_path.startswith("s3://"):
        original_data_path = f"s3://{original_data_path}"
    if not small_data_path.startswith("s3://"):
        small_data_path = f"s3://{small_data_path}"

    try:
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

        # 处理数据 - 创建小样本
        # 游戏样本 - 优先选择热门游戏
        if not games_df.empty:
            if 'user_reviews' in games_df.columns:
                # 按评论数排序选择热门游戏
                games_sample = games_df.sort_values('user_reviews', ascending=False).head(n_games)
            else:
                games_sample = games_df.sample(min(n_games, len(games_df)), random_state=42)
        else:
            games_sample = pd.DataFrame()

        # 用户样本
        if not users_df.empty:
            # 优先选择有更多产品和评论的活跃用户
            if 'products' in users_df.columns and 'reviews' in users_df.columns:
                # 创建活跃度指标
                users_df['activity_score'] = users_df['products'] + users_df['reviews'] * 3
                users_sample = users_df.sort_values('activity_score', ascending=False).head(n_users)
            else:
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
                # 先获取热门游戏的评价
                pop_games_recs = recommendations_df[
                    recommendations_df['app_id'].isin(games_sample['app_id'])
                ]

                if len(pop_games_recs) >= n_ratings:
                    recommendations_sample = pop_games_recs.sample(
                        min(n_ratings, len(pop_games_recs)), random_state=42
                    )
                else:
                    recommendations_sample = recommendations_df.sample(
                        min(n_ratings, len(recommendations_df)), random_state=42
                    )
            else:
                recommendations_sample = filtered_recommendations.sample(
                    min(n_ratings, len(filtered_recommendations)), random_state=42
                )
        else:
            recommendations_sample = pd.DataFrame()

        # 获取样本中涉及到的游戏和用户ID
        unique_game_ids = set(recommendations_sample['app_id'].unique())
        unique_user_ids = set(recommendations_sample['user_id'].unique())

        # 确保包含这些游戏和用户
        if not games_df.empty:
            missing_games = unique_game_ids - set(games_sample['app_id'])
            if missing_games:
                missing_games_df = games_df[games_df['app_id'].isin(missing_games)]
                games_sample = pd.concat([games_sample, missing_games_df], ignore_index=True)

        if not users_df.empty:
            missing_users = unique_user_ids - set(users_sample['user_id'])
            if missing_users:
                missing_users_df = users_df[users_df['user_id'].isin(missing_users)]
                users_sample = pd.concat([users_sample, missing_users_df], ignore_index=True)

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
        # 解析目标S3路径
        target_s3_bucket = small_data_path.replace("s3://", "").split("/")[0]
        target_s3_prefix = "/".join(small_data_path.replace("s3://", "").split("/")[1:])
        if target_s3_prefix and not target_s3_prefix.endswith("/"):
            target_s3_prefix += "/"

        # 保存游戏数据
        if not games_sample.empty:
            csv_buffer = io.StringIO()
            games_sample.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=target_s3_bucket,
                Key=f"{target_s3_prefix}games.csv",
                Body=csv_buffer.getvalue()
            )
            print(f"已保存 {len(games_sample)} 个游戏样本到S3")

        # 保存用户数据
        if not users_sample.empty:
            csv_buffer = io.StringIO()
            users_sample.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=target_s3_bucket,
                Key=f"{target_s3_prefix}users.csv",
                Body=csv_buffer.getvalue()
            )
            print(f"已保存 {len(users_sample)} 个用户样本到S3")

        # 保存评价数据
        if not recommendations_sample.empty:
            csv_buffer = io.StringIO()
            recommendations_sample.to_csv(csv_buffer, index=False)
            s3_client.put_object(
                Bucket=target_s3_bucket,
                Key=f"{target_s3_prefix}recommendations.csv",
                Body=csv_buffer.getvalue()
            )
            print(f"已保存 {len(recommendations_sample)} 条评价样本到S3")

        # 保存元数据
        if selected_metadata:
            json_lines = "\n".join([json.dumps(item) for item in selected_metadata])
            s3_client.put_object(
                Bucket=target_s3_bucket,
                Key=f"{target_s3_prefix}games_metadata.json",
                Body=json_lines
            )
            print(f"已保存 {len(selected_metadata)} 条游戏元数据到S3")

        print(f"小型数据集创建完成，保存在 {small_data_path}")

        # 打印数据减少比例
        original_counts = {
            "games": len(games_df),
            "users": len(users_df),
            "recommendations": len(recommendations_df),
            "metadata": len(metadata_list)
        }

        sample_counts = {
            "games": len(games_sample),
            "users": len(users_sample),
            "recommendations": len(recommendations_sample),
            "metadata": len(selected_metadata)
        }

        print("\n数据减少比例:")
        for key in original_counts:
            if original_counts[key] > 0:
                reduction = original_counts[key] / max(1, sample_counts[key])
                print(f"{key}: 原始 {original_counts[key]} -> 小型 {sample_counts[key]} (减少了 {reduction:.1f} 倍)")

    except Exception as e:
        print(f"创建小型数据集时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_small_dataset_s()