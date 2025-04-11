# src/online_learning/models.py
"""
模型加载、更新和管理。
"""

import logging
import os
import time
import pickle
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator

from . import MODEL_CHECKPOINT_DIR

# 设置日志
logger = logging.getLogger("online_learning.models")

# 全局模型变量
spark = None
als_model = None
tfidf_vectorizer = None
cosine_sim = None
indices = None
games_df = None
users_df = None
recommendations_df = None
metadata_df = None
processed_games = None


def initialize_spark():
    """初始化Spark会话，支持从S3读取数据"""
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName("OnlineSteamRecommender") \
            .master("local[*]") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", "")) \
            .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", "")) \
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        logger.info("Spark会话已初始化")
    return spark


def initialize_models(data_path='data', model_path='models/als_model'):
    """
    从磁盘或S3加载初始数据和模型

    Args:
        data_path (str): 数据目录路径
        model_path (str): 模型文件路径
    """
    global als_model, tfidf_vectorizer, cosine_sim, indices, games_df, users_df, recommendations_df, metadata_df, processed_games

    logger.info("正在初始化在线学习系统...")
    initialize_spark()

    # 加载数据
    logger.info(f"从 {data_path} 加载数据...")
    try:
        from src import load_data, preprocess_data
        games_df, users_df, recommendations_df, metadata_df = load_data(data_path)
        processed_games, _, _ = preprocess_data(games_df, users_df, recommendations_df, metadata_df, spark)
    except Exception as e:
        logger.error(f"加载数据出错: {e}")
        raise

    # 加载ALS模型
    logger.info(f"从 {model_path} 加载ALS模型...")
    try:
        als_model = ALSModel.load(model_path)
    except Exception as e:
        logger.error(f"加载ALS模型出错: {e}")
        raise

    # 构建TF-IDF模型
    logger.info("构建TF-IDF模型...")
    try:
        from src import build_tfidf_model
        tfidf_vectorizer, cosine_sim, indices, _ = build_tfidf_model(processed_games)

        # 保存TF-IDF模型词汇表用于增量更新
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
        with open(f"{MODEL_CHECKPOINT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
    except Exception as e:
        logger.error(f"构建TF-IDF模型出错: {e}")
        raise

    logger.info("系统初始化完成")


def get_global_models_and_data():
    """
    获取全局模型和数据

    Returns:
        tuple: 包含所有全局模型和数据的元组
    """
    return (als_model, tfidf_vectorizer, cosine_sim, indices,
            games_df, users_df, recommendations_df, metadata_df, processed_games)


def update_model_data(buffer_data):
    """
    使用缓冲区数据更新全局数据

    Args:
        buffer_data (dict): 缓冲区数据字典

    Returns:
        tuple: 更新后的数据集元组 (games_df, users_df, recommendations_df, metadata_df, processed_games)
    """
    global games_df, users_df, recommendations_df, metadata_df, processed_games

    # 更新游戏数据
    if buffer_data['games']:
        new_games_df = pd.DataFrame(buffer_data['games'])
        if games_df is not None:
            games_df = pd.concat([games_df, new_games_df], ignore_index=True)
            games_df = games_df.drop_duplicates(subset=['app_id'], keep='last')
        else:
            games_df = new_games_df

    # 更新用户数据
    if buffer_data['users']:
        new_users_df = pd.DataFrame(buffer_data['users'])
        if users_df is not None:
            users_df = pd.concat([users_df, new_users_df], ignore_index=True)
            users_df = users_df.drop_duplicates(subset=['user_id'], keep='last')
        else:
            users_df = new_users_df

    # 更新元数据
    if buffer_data['metadata']:
        new_metadata_df = pd.DataFrame(buffer_data['metadata'])
        if metadata_df is not None:
            metadata_df = pd.concat([metadata_df, new_metadata_df], ignore_index=True)
            metadata_df = metadata_df.drop_duplicates(subset=['app_id'], keep='last')
        else:
            metadata_df = new_metadata_df

    # 处理推荐数据
    if buffer_data['recommendations']:
        # 转换为评分
        for rec in buffer_data['recommendations']:
            if 'is_recommended' in rec and 'hours' in rec and 'rating' not in rec:
                rec['rating'] = min(10.0, float(rec['hours']) / 10) * (1.5 if rec['is_recommended'] else 0.5)

        # 添加到全局数据
        new_ratings_df = pd.DataFrame(buffer_data['recommendations'])
        if recommendations_df is not None:
            recommendations_df = pd.concat([recommendations_df, new_ratings_df], ignore_index=True)
        else:
            recommendations_df = new_ratings_df

    # 如果游戏数据或元数据有更新，重新处理游戏特征
    if buffer_data['games'] or buffer_data['metadata']:
        try:
            from src import preprocess_data
            processed_games, _, _ = preprocess_data(games_df, users_df, recommendations_df, metadata_df, None)
        except Exception as e:
            logger.error(f"重新处理游戏数据时出错: {e}")

    return (games_df, users_df, recommendations_df, metadata_df, processed_games)


def update_als_model(new_data):
    """
    增量更新ALS模型

    Args:
        new_data (list): 新的用户评分数据列表

    Returns:
        tuple: (更新后的模型, 性能指标字典)
    """
    global als_model, spark

    if als_model is None:
        logger.error("ALS模型未初始化，无法更新")
        return als_model, {}

    start_time = time.time()

    # 将新数据转换为Spark DataFrame
    schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("app_id", IntegerType(), True),
        StructField("rating", FloatType(), True)
    ])

    new_ratings = []
    for row in new_data:
        try:
            # 确保数据类型正确
            if 'user_id' in row and 'app_id' in row:
                # 使用已计算的评分或默认值
                rating = float(row.get('rating', 5.0))

                new_ratings.append((
                    int(row['user_id']),
                    int(row['app_id']),
                    float(rating)
                ))
        except (ValueError, TypeError) as e:
            logger.warning(f"处理评分数据时出错: {e}, 数据: {row}")
            continue

    if not new_ratings:
        logger.warning("没有新的有效评分数据，ALS模型未更新")
        return als_model, {}

    new_ratings_df = spark.createDataFrame(new_ratings, schema)
    logger.info(f"为ALS更新准备了 {len(new_ratings)} 条新评分数据")

    # 获取现有模型的超参数
    current_params = {
        'rank': als_model.rank,
        'regParam': als_model.getRegParam(),
        'alpha': als_model.getAlpha(),
        'maxIter': als_model.getMaxIter()
    }

    # 增量训练ALS模型 - 使用较少的迭代次数进行增量更新
    incremental_als = ALS(
        maxIter=2,  # 减少迭代次数，加快增量更新
        rank=current_params['rank'],
        regParam=current_params['regParam'],
        alpha=current_params['alpha'],
        userCol="user_id",
        itemCol="app_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=True,
        seed=42
    )

    # 设置初始化因子（如果可用）- 重用现有的用户和物品因子加速训练
    try:
        # 从现有模型获取用户和物品因子
        userFactors = als_model.userFactors
        itemFactors = als_model.itemFactors

        # 在新模型中设置初始因子
        incremental_als.setUserFactors(userFactors)
        incremental_als.setItemFactors(itemFactors)
        logger.info("使用现有因子初始化增量ALS模型")
    except Exception as e:
        logger.error(f"设置初始化因子时出错: {e}")

    # 训练模型
    try:
        updated_model = incremental_als.fit(new_ratings_df)
        training_time = time.time() - start_time

        # 简单评估 - 计算新数据上的RMSE
        predictions = updated_model.transform(new_ratings_df)

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )

        rmse = evaluator.evaluate(predictions)

        metrics = {
            'training_time': training_time,
            'rmse': rmse,
            'num_samples': len(new_ratings),
            'update_type': 'incremental'
        }

        logger.info(f"ALS模型更新完成，RMSE: {rmse:.4f}, 用时: {training_time:.2f}秒")

        # 更新全局模型
        als_model = updated_model

        return updated_model, metrics

    except Exception as e:
        logger.error(f"更新ALS模型时出错: {e}")
        return als_model, {}


def update_tfidf_model(new_games):
    """
    增量更新TF-IDF模型

    Args:
        new_games (list): 新游戏数据列表

    Returns:
        tuple: (更新后的TF-IDF向量化器, 相似度矩阵, 索引, 性能指标字典)
    """
    global tfidf_vectorizer, cosine_sim, indices, processed_games

    if tfidf_vectorizer is None or processed_games is None:
        logger.error("TF-IDF模型或游戏数据未初始化，无法更新")
        return tfidf_vectorizer, cosine_sim, indices, {}

    start_time = time.time()

    # 过滤出新的游戏
    new_game_ids = [game['app_id'] for game in new_games if 'app_id' in game]
    existing_game_ids = set(processed_games['app_id'].values)
    actual_new_game_ids = [gid for gid in new_game_ids if gid not in existing_game_ids]

    if not actual_new_game_ids:
        logger.warning("没有新的游戏数据，TF-IDF模型未更新")
        return tfidf_vectorizer, cosine_sim, indices, {}

    logger.info(f"处理 {len(actual_new_game_ids)} 个新游戏以更新TF-IDF模型")

    # 准备新游戏数据
    new_game_df = pd.DataFrame(new_games)

    # 确保标签列格式正确
    if 'tags' in new_game_df.columns:
        new_game_df['tags'] = new_game_df['tags'].apply(
            lambda x: x if isinstance(x, list) else
            (json.loads(x) if isinstance(x, str) and x.startswith('[') else [])
        )
    else:
        new_game_df['tags'] = [[] for _ in range(len(new_game_df))]

    # 预处理新游戏数据
    try:
        from src import prepare_content_data
        new_processed_games = prepare_content_data(new_game_df)

        # 使用现有词汇表转换新内容
        existing_features = tfidf_vectorizer.transform(processed_games['content_features'])
        new_features = tfidf_vectorizer.transform(new_processed_games['content_features'])

        # 合并特征矩阵和游戏数据
        combined_features = vstack([existing_features, new_features])
        combined_games = pd.concat([processed_games, new_processed_games], ignore_index=True)

        # 重新计算余弦相似度矩阵
        new_cosine_sim = cosine_similarity(combined_features, combined_features)

        # 更新索引
        new_indices = pd.Series(combined_games.index, index=combined_games['app_id'])

        # 更新全局变量
        processed_games = combined_games
        cosine_sim = new_cosine_sim
        indices = new_indices

        # 计算处理时间
        processing_time = time.time() - start_time

        metrics = {
            'processing_time': processing_time,
            'num_new_games': len(actual_new_game_ids),
            'total_games': len(combined_games),
            'update_type': 'incremental_tfidf'
        }

        logger.info(
            f"TF-IDF模型增量更新完成，新增 {len(actual_new_game_ids)} 个游戏，总计 {len(combined_games)} 个游戏，用时: {processing_time:.2f}秒")

        return tfidf_vectorizer, new_cosine_sim, new_indices, metrics

    except Exception as e:
        logger.error(f"增量更新TF-IDF模型时出错: {e}")

        # 如果增量更新失败，尝试完全重建
        try:
            logger.info("尝试完全重建TF-IDF模型")
            from src import build_tfidf_model

            # 合并游戏数据
            combined_games = pd.concat([processed_games, new_processed_games], ignore_index=True)

            # 重建模型
            new_tfidf, new_cosine_sim, new_indices, _ = build_tfidf_model(combined_games)

            # 更新全局变量
            tfidf_vectorizer = new_tfidf
            cosine_sim = new_cosine_sim
            indices = new_indices
            processed_games = combined_games

            processing_time = time.time() - start_time

            metrics = {
                'processing_time': processing_time,
                'num_new_games': len(actual_new_game_ids),
                'total_games': len(combined_games),
                'update_type': 'full_rebuild_tfidf'
            }

            logger.info(f"TF-IDF模型完全重建完成，总计 {len(combined_games)} 个游戏，用时: {processing_time:.2f}秒")

            return new_tfidf, new_cosine_sim, new_indices, metrics

        except Exception as e2:
            logger.error(f"重建TF-IDF模型时出错: {e2}")
            return tfidf_vectorizer, cosine_sim, indices, {}