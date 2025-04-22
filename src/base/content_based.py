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


def build_tfidf_model(games_df, max_features=5000, max_items=15000):
    """构建TF-IDF模型，限制最大项目数"""
    # 如果游戏太多，采样减少
    if len(games_df) > max_items:
        print(f"游戏数量太多({len(games_df)})，随机采样 {max_items} 个游戏")
        # 按游戏热度排序，保留最热门的游戏
        if 'user_reviews' in games_df.columns:
            # 按评论数排序
            games_sample = games_df.sort_values('user_reviews', ascending=False).head(max_items)
        else:
            # 随机采样
            games_sample = games_df.sample(max_items, random_state=42)
        # 使用采样后的数据
        content_df = prepare_content_data(games_sample)
    else:
        # 准备内容数据
        content_df = prepare_content_data(games_df)

    # 创建TF-IDF向量化器
    tfidf = TfidfVectorizer(
        max_features=max_features,  # 减少特征数量
        stop_words='english',
        ngram_range=(1, 2)
    )

    # 生成TF-IDF矩阵
    print(f"生成TF-IDF矩阵 (max_features={max_features})...")
    tfidf_matrix = tfidf.fit_transform(content_df['content_features'])
    print(f"TF-IDF矩阵大小: {tfidf_matrix.shape}")

    # 计算余弦相似度 - 使用分块计算
    print("计算余弦相似度矩阵（分块处理以减少内存使用）...")
    cosine_sim = chunked_cosine_similarity(tfidf_matrix)

    # 创建游戏索引映射
    indices = pd.Series(content_df.index, index=content_df['app_id'])

    return tfidf, cosine_sim, indices, content_df
# 分块计算余弦相似度
def chunked_cosine_similarity(X, chunk_size=1000):
    """
    使用分块法和更小的数据类型计算余弦相似度
    """
    from scipy import sparse
    import numpy as np
    import tempfile
    import os

    n_samples = X.shape[0]
    # 使用临时文件来存储结果
    temp_dir = tempfile.mkdtemp()
    result_file = os.path.join(temp_dir, "cosine_sim.npy")

    try:
        # 如果X不是稀疏矩阵，转换为稀疏矩阵以节省内存
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # 预分配一个内存映射文件，使用float32而不是float64
        # 这会将内存需求减半
        cosine_sim = np.memmap(
            result_file,
            dtype='float32',
            mode='w+',
            shape=(n_samples, n_samples)
        )

        # 分块计算
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            chunk = X[i:end]

            # 计算这个块与所有数据的余弦相似度
            # normalize=True ensures vectors are normalized before computing the dot product
            chunk_sims = sparse.csr_matrix(chunk.dot(X.T).toarray(), dtype='float32')

            # 归一化 - 对每一行根据其范数进行归一化
            # 这相当于计算余弦相似度
            norms1 = np.sqrt((chunk.multiply(chunk)).sum(axis=1))
            norms2 = np.sqrt((X.multiply(X)).sum(axis=1))
            norm_mat = np.outer(norms1, norms2)
            norm_mat[norm_mat == 0] = 1.0  # 避免除以零

            # 归一化得到余弦相似度
            chunk_sims = chunk_sims.toarray() / norm_mat

            # 存储结果到内存映射数组
            cosine_sim[i:end] = chunk_sims.astype('float32')

            # 打印进度
            print(f"计算余弦相似度: {end}/{n_samples} 完成 ({end / n_samples:.1%})")

            # 强制清理内存
            del chunk_sims
            import gc
            gc.collect()

        # 完成后从磁盘读取整个结果
        return np.array(cosine_sim, dtype='float32')

    except Exception as e:
        print(f"计算余弦相似度时出错: {e}")
        # 尝试使用更小的空间计算基本相似度
        return calculate_fallback_similarity(X)
    finally:
        # 无论如何都要清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def calculate_fallback_similarity(X, max_items=20000):
    """
    当内存不足时的备用方法:
    1. 限制项目数量
    2. 使用更简单的相似度计算
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    n_samples = X.shape[0]

    if n_samples > max_items:
        print(f"警告: 项目数量太多 ({n_samples})，截断为 {max_items}")
        # 仅使用前max_items项
        X = X[:max_items]
        n_samples = max_items

    # 使用更小的数据类型
    cosine_sim = np.zeros((n_samples, n_samples), dtype='float32')

    # 使用较小的分块
    chunk_size = 500
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunk = X[i:end]

        # 计算当前块与所有数据的相似度
        similarity = cosine_similarity(chunk, X)
        cosine_sim[i:end] = similarity.astype('float32')

        print(f"备用相似度计算: {end}/{n_samples} 完成 ({end / n_samples:.1%})")

    return cosine_sim

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