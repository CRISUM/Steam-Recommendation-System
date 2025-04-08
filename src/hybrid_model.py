# src/hybrid_model.py
import pandas as pd
import numpy as np


def build_hybrid_recommender(cf_model, cosine_sim, indices, games_df, user_ratings_df,
                             cf_weight=0.7, spark=None):
    """构建混合推荐系统"""
    if spark is None:
        raise ValueError("必须提供Spark会话才能使用协同过滤模型")

    # 创建用户-游戏评分字典
    user_ratings = user_ratings_df.groupby('user_id').apply(
        lambda x: dict(zip(x['app_id'], x['rating']))
    ).to_dict()

    def get_recommendations(user_id, top_n=10):
        """为指定用户生成混合推荐"""
        # 检查用户是否有评分历史
        is_new_user = user_id not in user_ratings or not user_ratings[user_id]

        # 如果是新用户，返回热门游戏
        if is_new_user:
            print(f"用户 {user_id} 是新用户，返回热门游戏推荐")
            return get_popular_recommendations(user_ratings_df, games_df, top_n)

        # 协同过滤推荐
        user_df = spark.createDataFrame([(user_id,)], ["user_id"])
        cf_recommendations = cf_model.recommendForUserSubset(user_df, top_n * 2)

        # 如果协同过滤无法给出推荐
        if cf_recommendations.count() == 0:
            print(f"协同过滤无法为用户 {user_id} 提供推荐，使用内容推荐")
            return get_content_recommendations_for_user(
                user_id, user_ratings, indices, cosine_sim, games_df, top_n
            )

        # 获取协同过滤推荐结果
        cf_recs = cf_recommendations.collect()[0].recommendations
        cf_scores = {rec.app_id: rec.rating for rec in cf_recs}

        # 获取用户评分过的游戏
        user_games = list(user_ratings[user_id].keys())

        # 基于内容的推荐
        content_scores = get_content_scores_for_user(
            user_id, user_games, user_ratings, indices, cosine_sim, games_df, top_n
        )

        # 归一化分数
        if content_scores:
            max_content = max(content_scores.values())
            content_scores = {k: v / max_content for k, v in content_scores.items()}

        if cf_scores:
            max_cf = max(cf_scores.values())
            cf_scores = {k: v / max_cf for k, v in cf_scores.items()}

        # 混合两种推荐
        hybrid_scores = {}
        all_games = set(list(cf_scores.keys()) + list(content_scores.keys()))

        for game in all_games:
            if game in user_games:  # 跳过用户已评分的游戏
                continue

            cf_score = cf_scores.get(game, 0)
            content_score = content_scores.get(game, 0)
            hybrid_scores[game] = cf_weight * cf_score + (1 - cf_weight) * content_score

        # 返回TOP-N推荐
        recommended_ids = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommended_games = games_df[games_df['app_id'].isin([x[0] for x in recommended_ids])].copy()

        # 添加推荐分数
        score_dict = dict(recommended_ids)
        recommended_games['recommendation_score'] = recommended_games['app_id'].map(score_dict)

        # 按推荐分数排序
        recommended_games = recommended_games.sort_values('recommendation_score', ascending=False)

        return recommended_games

    return get_recommendations


def get_popular_recommendations(ratings_df, games_df, top_n=10):
    """获取热门游戏推荐（用于冷启动）"""
    # 计算游戏的平均评分和评分数量
    game_stats = ratings_df.groupby('app_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    # 整理列名
    game_stats.columns = ['app_id', 'avg_rating', 'rating_count']

    # 计算置信度区间下限的Wilson得分
    # 这是一种比简单平均评分更好的排序方式，可以平衡评分和评分数量
    def wilson_score(pos, n):
        """计算Wilson得分区间下限"""
        if n == 0:
            return 0
        z = 1.96  # 95% 置信度的z分数
        phat = pos / n
        return (phat + z * z / (2 * n) - z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

    # 将评分映射到0-1区间
    game_stats['normalized_rating'] = game_stats['avg_rating'] / 10.0

    # 计算Wilson得分
    game_stats['wilson_score'] = game_stats.apply(
        lambda x: wilson_score(x['normalized_rating'] * x['rating_count'], x['rating_count']),
        axis=1
    )

    # 排序并获取前N个游戏
    top_games = game_stats.sort_values('wilson_score', ascending=False).head(top_n)

    # 合并游戏详情
    recommended_games = games_df[games_df['app_id'].isin(top_games['app_id'])].copy()

    # 添加推荐分数
    recommended_games = pd.merge(
        recommended_games,
        top_games[['app_id', 'wilson_score']],
        on='app_id'
    )

    return recommended_games.sort_values('wilson_score', ascending=False)


def get_content_recommendations_for_user(user_id, user_ratings, indices, cosine_sim, games_df, top_n=10):
    """基于用户历史评分，使用内容过滤提供推荐"""
    user_games = list(user_ratings[user_id].keys())
    content_scores = get_content_scores_for_user(
        user_id, user_games, user_ratings, indices, cosine_sim, games_df
    )

    # 排序并获取前N个游戏
    recommended_ids = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommended_games = games_df[games_df['app_id'].isin([x[0] for x in recommended_ids])].copy()

    # 添加推荐分数
    score_dict = dict(recommended_ids)
    recommended_games['recommendation_score'] = recommended_games['app_id'].map(score_dict)

    # 按推荐分数排序
    recommended_games = recommended_games.sort_values('recommendation_score', ascending=False)

    return recommended_games


def get_content_scores_for_user(user_id, user_games, user_ratings, indices, cosine_sim, games_df, top_n=20):
    """计算用户可能喜欢的游戏的内容分数"""
    content_scores = {}

    for game_id in user_games:
        # 检查游戏是否在索引中
        if game_id not in indices:
            continue

        # 获取相似游戏
        idx = indices[game_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        # 用户对当前游戏的评分
        user_rating = user_ratings[user_id][game_id]

        # 加权累加相似度分数
        for i, score in sim_scores:
            game_index = i
            app_id = games_df.iloc[game_index]['app_id']

            if app_id not in content_scores:
                content_scores[app_id] = 0

            # 权重 = 用户评分 * 相似度
            content_scores[app_id] += user_rating * score

    return content_scores