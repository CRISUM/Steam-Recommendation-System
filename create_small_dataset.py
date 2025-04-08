# create_small_dataset.py
import pandas as pd
import numpy as np
import os
import json
import shutil


def create_small_dataset(original_data_path="data", small_data_path="data_small",
                         n_games=1000, n_users=500, n_ratings=5000):
    """从原始数据集创建一个小型测试数据集"""
    print(f"创建小型测试数据集: {n_games} 个游戏, {n_users} 个用户, {n_ratings} 条评价")

    # 创建目录
    os.makedirs(small_data_path, exist_ok=True)

    # 检查原始数据是否存在
    if not os.path.exists(original_data_path):
        print(f"错误: 原始数据路径 {original_data_path} 不存在")
        return

    try:
        # 加载游戏数据并取子集
        games_file = f"{original_data_path}/games.csv"
        if os.path.exists(games_file):
            games_df = pd.read_csv(games_file)
            games_sample = games_df.sample(min(n_games, len(games_df)), random_state=42)
            games_sample.to_csv(f"{small_data_path}/games.csv", index=False)
            print(f"已保存 {len(games_sample)} 个游戏样本")
        else:
            print(f"警告: 找不到游戏数据文件 {games_file}")
            games_sample = pd.DataFrame()

        # 加载用户数据并取子集
        users_file = f"{original_data_path}/users.csv"
        if os.path.exists(users_file):
            users_df = pd.read_csv(users_file)
            users_sample = users_df.sample(min(n_users, len(users_df)), random_state=42)
            users_sample.to_csv(f"{small_data_path}/users.csv", index=False)
            print(f"已保存 {len(users_sample)} 个用户样本")
        else:
            print(f"警告: 找不到用户数据文件 {users_file}")
            users_sample = pd.DataFrame()

        # 加载并过滤评价数据
        recommendations_file = f"{original_data_path}/recommendations.csv"
        if os.path.exists(recommendations_file) and not games_sample.empty and not users_sample.empty:
            recommendations_df = pd.read_csv(recommendations_file)

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

            recommendations_sample.to_csv(f"{small_data_path}/recommendations.csv", index=False)
            print(f"已保存 {len(recommendations_sample)} 条评价样本")
        else:
            print(f"警告: 找不到评价数据文件或游戏/用户样本为空")

        # 处理元数据文件
        metadata_file = f"{original_data_path}/games_metadata.json"
        if os.path.exists(metadata_file) and not games_sample.empty:
            game_ids = set(games_sample['app_id'])
            selected_metadata = []

            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if item.get('app_id') in game_ids:
                            selected_metadata.append(item)
                    except json.JSONDecodeError:
                        continue

            with open(f"{small_data_path}/games_metadata.json", 'w', encoding='utf-8') as f:
                for item in selected_metadata:
                    f.write(json.dumps(item) + '\n')

            print(f"已保存 {len(selected_metadata)} 条游戏元数据")
        else:
            print(f"警告: 找不到元数据文件 {metadata_file} 或游戏样本为空")

        print(f"小型测试数据集创建完成，保存在 {small_data_path}")

    except Exception as e:
        print(f"创建小型数据集时出错: {e}")


if __name__ == "__main__":
    create_small_dataset()