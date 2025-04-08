# src/cold_start.py
import pandas as pd
import numpy as np


def build_popularity_model(ratings_df, games_df):
    """构建基于流行度的推荐模型（用于新用户冷启动）"""
    # 计算每个游戏的评分统计
    game_stats = ratings_df.groupby('app_id').agg({
        'rating': ['mean', 'count', 'sum']
    }).reset_index()

    # 重命名列
    game_stats.columns = ['app_id', 'avg_rating', 'rating_count', 'rating_sum']

    # 计算流行度分数
    # 结合平均评分和评分数量
    game_stats['popularity_score'] = (
            game_stats['avg_rating'] *
            np.log1p(game_stats['rating_count'])  # log1p = log(1+x) 避免评分数为0的问题
    )

    def get_popular_recommendations(top_n=10):
        """获取最流行的游戏"""
        top_games = game_stats.sort_values('popularity_score', ascending=False).head(top_n)
        recommended_games = games_df[games_df['app_id'].isin(top_games['app_id'])].copy()

        # 合并流行度分数
        recommended_games = pd.merge(
            recommended_games,
            top_games[['app_id', 'popularity_score']],
            on='app_id'
        )

        return recommended_games.sort_values('popularity_score', ascending=False)

    return get_popular_recommendations


def build_content_based_cold_start(cosine_sim, indices, games_df):
    """构建基于内容的冷启动推荐（用于新游戏冷启动）"""

    def recommend_similar_games(app_id, top_n=10):
        """为特定游戏推荐相似游戏"""
        # 检查游戏是否在索引中
        if app_id not in indices:
            print(f"警告: 游戏ID {app_id} 未在索引中找到")
            return pd.DataFrame()

        # 获取游戏索引
        idx = indices[app_id]

        # 获取相似度分数
        sim_scores = list(enumerate(cosine_sim[idx]))

        # 根据相似度排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 排除自己
        sim_scores = sim_scores[1:top_n + 1]

        # 获取游戏索引
        game_indices = [i[0] for i in sim_scores]

        # 返回推荐结果
        recommendations = games_df.iloc[game_indices].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]

        return recommendations

    return recommend_similar_games


def get_user_segment(user_id, user_ratings, games_df):
    """根据用户历史确定用户细分"""
    # 如果用户是新用户
    if user_id not in user_ratings or len(user_ratings[user_id]) == 0:
        return "new_user"

    # 获取用户评分的游戏
    user_games = list(user_ratings[user_id].keys())
    rated_games = games_df[games_df['app_id'].isin(user_games)]

    # 统计用户评分游戏的标签
    tag_counts = {}
    for _, game in rated_games.iterrows():
        if 'tags' in game and game['tags']:
            for tag in game['tags']:
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1

    # 找出最受欢迎的标签
    if tag_counts:
        top_tag = max(tag_counts.items(), key=lambda x: x[1])[0]
        return f"tag_{top_tag}"

    # 如果没有足够信息，根据用户评分数量返回一个细分
    num_ratings = len(user_ratings[user_id])
    if num_ratings < 5:
        return "casual_gamer"
    elif num_ratings < 20:
        return "regular_gamer"
    else:
        return "hardcore_gamer"