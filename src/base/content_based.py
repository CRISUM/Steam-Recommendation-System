# src/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def prepare_content_data(games_df):
    """准备用于内容推荐的数据"""
    # 确保描述列存在且没有NaN值
    if 'description' not in games_df.columns:
        games_df['description'] = ''
    else:
        games_df['description'] = games_df['description'].fillna('')

    # 处理标签数据
    if 'tags' in games_df.columns:
        # 确保标签是列表并转换为字符串
        games_df['tags_str'] = games_df['tags'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
    else:
        games_df['tags_str'] = ''

    # 合并特征文本
    games_df['content_features'] = games_df['description'] + ' ' + games_df['tags_str']

    return games_df


def build_tfidf_model(games_df, max_features=10000):
    """构建TF-IDF模型"""
    # 准备内容数据
    content_df = prepare_content_data(games_df)

    # 创建TF-IDF向量化器
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )

    # 生成TF-IDF矩阵
    tfidf_matrix = tfidf.fit_transform(content_df['content_features'])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 创建游戏索引映射
    indices = pd.Series(content_df.index, index=content_df['app_id'])

    return tfidf, cosine_sim, indices, content_df


def get_content_recommendations(app_id, cosine_sim, indices, games_df, top_n=10):
    """根据内容相似度获取推荐"""
    # 检查app_id是否在索引中
    if app_id not in indices:
        print(f"警告: 游戏ID {app_id} 未在索引中找到")
        return pd.DataFrame()

    # 获取游戏索引
    idx = indices[app_id]

    # 获取相似度分数
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 根据相似度排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 获取前N个相似游戏（除了自己）
    sim_scores = sim_scores[1:top_n + 1]

    # 获取游戏索引
    game_indices = [i[0] for i in sim_scores]

    # 返回推荐结果
    recommendations = games_df.iloc[game_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return recommendations[['app_id', 'title', 'similarity_score']]


def get_similar_games_by_tags(app_id, games_df, top_n=10):
    """根据标签相似度获取推荐（备用方法）"""
    # 获取目标游戏的标签
    target_game = games_df[games_df['app_id'] == app_id]

    if len(target_game) == 0:
        print(f"警告: 游戏ID {app_id} 未找到")
        return pd.DataFrame()

    target_tags = target_game['tags'].iloc[0]

    if not target_tags:
        print(f"警告: 游戏ID {app_id} 没有标签")
        return pd.DataFrame()

    # 计算每个游戏与目标游戏的标签重叠度
    def calculate_tag_similarity(game_tags):
        if not game_tags:
            return 0
        # Jaccard相似度: 交集大小/并集大小
        intersection = len(set(target_tags) & set(game_tags))
        union = len(set(target_tags) | set(game_tags))
        return intersection / union if union > 0 else 0

    # 计算所有游戏的相似度
    games_df['tag_similarity'] = games_df['tags'].apply(calculate_tag_similarity)

    # 排除目标游戏本身
    similar_games = games_df[games_df['app_id'] != app_id].sort_values(
        'tag_similarity', ascending=False
    ).head(top_n)

    return similar_games[['app_id', 'title', 'tag_similarity']]