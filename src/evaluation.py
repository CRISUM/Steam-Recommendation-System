# src/evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def calculate_precision_recall(recommended_items, relevant_items):
    """计算准确率和召回率"""
    if not recommended_items or not relevant_items:
        return 0, 0

    true_positives = len(set(recommended_items) & set(relevant_items))
    precision = true_positives / len(recommended_items) if recommended_items else 0
    recall = true_positives / len(relevant_items) if relevant_items else 0

    return precision, recall


def calculate_ndcg(recommended_items, item_relevance, k=None):
    """计算NDCG (归一化折损累积增益)"""
    if not recommended_items or not item_relevance:
        return 0

    if k is not None:
        recommended_items = recommended_items[:k]

    # 计算DCG
    dcg = 0
    for i, item in enumerate(recommended_items):
        if item in item_relevance:
            # 使用实际评分作为相关性分数
            rel = item_relevance.get(item, 0)
            dcg += rel / np.log2(i + 2)  # i+2 是因为 log2(1) = 0

    # 计算IDCG (理想DCG)
    sorted_relevance = sorted(item_relevance.values(), reverse=True)
    if k is not None:
        sorted_relevance = sorted_relevance[:k]

    idcg = 0
    for i, rel in enumerate(sorted_relevance):
        idcg += rel / np.log2(i + 2)

    # 计算NDCG
    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg


def calculate_diversity(recommended_items, item_features):
    """计算推荐结果的多样性"""
    if len(recommended_items) <= 1:
        return 0

    # 获取推荐项目的特征
    features = [item_features.get(item, []) for item in recommended_items]

    # 计算特征集合的平均Jaccard距离
    total_distance = 0
    count = 0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            set_i = set(features[i])
            set_j = set(features[j])

            if not set_i or not set_j:
                continue

            # Jaccard距离 = 1 - Jaccard相似度
            jaccard_sim = len(set_i & set_j) / len(set_i | set_j)
            jaccard_dist = 1 - jaccard_sim

            total_distance += jaccard_dist
            count += 1

    # 平均距离
    avg_distance = total_distance / count if count > 0 else 0

    return avg_distance


def evaluate_recommendations(recommender_func, test_users, games_df, test_data, top_n=10):
    """评估推荐系统性能"""
    metrics = {
        'precision': [],
        'recall': [],
        'ndcg': [],
        'diversity': []
    }

    # 创建游戏标签字典
    game_tags = dict(zip(games_df['app_id'], games_df['tags']))

    for user_id in test_users:
        # 获取用户实际喜欢的游戏（评分高于平均值）
        user_ratings = test_data[test_data['user_id'] == user_id]

        if len(user_ratings) == 0:
            continue

        avg_rating = user_ratings['rating'].mean()
        relevant_items = list(user_ratings[user_ratings['rating'] > avg_rating]['app_id'])

        if len(relevant_items) == 0:
            continue

        # 创建用户评分字典
        item_relevance = dict(zip(user_ratings['app_id'], user_ratings['rating']))

        # 获取推荐结果
        try:
            recommendations = recommender_func(user_id, top_n)
            recommended_items = list(recommendations['app_id'])
        except Exception as e:
            print(f"为用户 {user_id} 生成推荐时出错: {e}")
            continue

        if len(recommended_items) == 0:
            continue

        # 计算指标
        precision, recall = calculate_precision_recall(recommended_items, relevant_items)
        ndcg = calculate_ndcg(recommended_items, item_relevance, top_n)
        diversity = calculate_diversity(recommended_items, game_tags)

        # 存储指标
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['ndcg'].append(ndcg)
        metrics['diversity'].append(diversity)

        # 计算平均值
    avg_metrics = {
        'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
        'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
        'ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0,
        'diversity': np.mean(metrics['diversity']) if metrics['diversity'] else 0,
        'f1': 0  # 稍后计算
    }

    # 计算F1分数
    if avg_metrics['precision'] > 0 and avg_metrics['recall'] > 0:
        avg_metrics['f1'] = 2 * (avg_metrics['precision'] * avg_metrics['recall']) / (
                    avg_metrics['precision'] + avg_metrics['recall'])

    return avg_metrics


def compare_recommenders(recommenders, test_users, games_df, test_data, top_n=10):
    """比较多个推荐器的性能"""
    results = {}

    for name, recommender in recommenders.items():
        print(f"评估推荐器: {name}")
        metrics = evaluate_recommendations(recommender, test_users, games_df, test_data, top_n)
        results[name] = metrics
        print(f"结果: {metrics}")

    return results


def visualize_comparison(results, save_path=None):
    """可视化比较结果"""
    # 准备数据
    metric_names = ['precision', 'recall', 'f1', 'ndcg', 'diversity']
    recommenders = list(results.keys())

    # 创建图表
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))

    for i, metric in enumerate(metric_names):
        values = [results[rec][metric] for rec in recommenders]

        # 创建条形图
        sns.barplot(x=recommenders, y=values, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} 比较')
        axes[i].set_ylim(0, 1)

        # 添加数值标签
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center')

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")

    plt.show()


def save_evaluation_results(results, file_path):
    """保存评估结果到JSON文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 将NumPy值转换为Python标准类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

    # 转换结果
    serializable_results = {}
    for model, metrics in results.items():
        serializable_results[model] = {k: convert_to_serializable(v) for k, v in metrics.items()}

    # 保存到文件
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    print(f"评估结果已保存至: {file_path}")