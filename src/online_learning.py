# src/online_learning.py
import os
import time
import json
import threading

import boto3
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, BooleanType
import logging
from collections import deque
import pickle
from pathlib import Path
import shutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("online_learning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("online_learning")

# 创建Flask应用
app = Flask(__name__)

# 全局变量
data_buffer = {
    'games': [],
    'users': [],
    'recommendations': [],
    'metadata': []
}

buffer_lock = threading.Lock()
update_thread = None
stop_event = threading.Event()
metrics_history = deque(maxlen=100)  # 保存最近100次更新的指标
last_update_time = None
update_count = 0

# 模型全局变量
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

# 配置
UPDATE_INTERVAL = 300  # 5分钟更新一次模型
BUFFER_SIZE_THRESHOLD = 100  # 当缓冲区达到100条记录时触发更新
MODEL_CHECKPOINT_DIR = 'models/online'
METRICS_FILE = 'results/online_metrics.json'
CF_WEIGHT = 0.7  # 协同过滤权重
INCREMENTAL_LEARNING_ENABLED = True


def initialize_spark():
    """初始化Spark会话"""
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName("OnlineSteamRecommender") \
            .master("local[*]") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        logger.info("Spark会话已初始化")


def initialize_system(data_path='data', model_path='models/als_model'):
    """从磁盘加载初始数据和模型"""
    global als_model, tfidf_vectorizer, cosine_sim, indices, games_df, users_df, recommendations_df, metadata_df, processed_games

    logger.info("正在初始化在线学习系统...")
    initialize_spark()

    # 加载数据
    logger.info(f"从 {data_path} 加载数据...")
    try:
        from src.data_processing import load_data, preprocess_data
        games_df, users_df, recommendations_df, metadata_df = load_data(data_path)
        processed_games, _, _ = preprocess_data(games_df, users_df, recommendations_df, metadata_df, spark)
    except Exception as e:
        logger.error(f"加载数据出错: {e}")
        raise

    # 加载ALS模型
    logger.info(f"从 {model_path} 加载ALS模型...")
    try:
        from pyspark.ml.recommendation import ALSModel
        als_model = ALSModel.load(model_path)
    except Exception as e:
        logger.error(f"加载ALS模型出错: {e}")
        raise

    # 构建TF-IDF模型
    logger.info("构建TF-IDF模型...")
    try:
        from src.content_based import build_tfidf_model
        tfidf_vectorizer, cosine_sim, indices, _ = build_tfidf_model(processed_games)

        # 保存TF-IDF模型词汇表用于增量更新
        Path(MODEL_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{MODEL_CHECKPOINT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
    except Exception as e:
        logger.error(f"构建TF-IDF模型出错: {e}")
        raise

    logger.info("系统初始化完成")


def save_metrics(metrics, bucket_name="steam-project-data"):
    """保存指标到文件"""
    Path(os.path.dirname(METRICS_FILE)).mkdir(parents=True, exist_ok=True)
    metrics['timestamp'] = datetime.now().isoformat()
    metrics_history.append(metrics)

    # 保存完整的指标历史
    with open(METRICS_FILE, 'w') as f:
        json.dump(list(metrics_history), f, indent=2)

    logger.info(f"指标已保存到 {METRICS_FILE}")

    # 添加S3上传
    try:
        s3_client = boto3.client('s3')
        with open(METRICS_FILE, 'rb') as f:
            s3_client.upload_fileobj(f, bucket_name, "results/online_metrics.json")
        logger.info(f"指标已上传到S3 {bucket_name}/results/online_metrics.json")
    except Exception as e:
        logger.error(f"上传指标到S3出错: {e}")

def save_model_checkpoint(model_type, model_obj, metrics, bucket_name="steam-project-data"):
    """保存模型检查点"""
    checkpoint_dir = f"{MODEL_CHECKPOINT_DIR}/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if model_type == "als":
        # 保存ALS模型
        try:
            model_obj.save(f"{checkpoint_dir}/model")
            logger.info(f"ALS模型已保存到 {checkpoint_dir}")
        except Exception as e:
            logger.error(f"保存ALS模型出错: {e}")

    elif model_type == "tfidf":
        # 保存TF-IDF模型
        try:
            with open(f"{checkpoint_dir}/tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(model_obj, f)

            np.save(f"{checkpoint_dir}/cosine_sim.npy", cosine_sim)
            with open(f"{checkpoint_dir}/indices.pkl", "wb") as f:
                pickle.dump(indices, f)

            logger.info(f"TF-IDF模型已保存到 {checkpoint_dir}")
        except Exception as e:
            logger.error(f"保存TF-IDF模型出错: {e}")

    # 保存指标
    with open(f"{checkpoint_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 添加S3上传
    try:
        s3_client = boto3.client('s3')
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, MODEL_CHECKPOINT_DIR)
                s3_key = f"online_checkpoints/{model_type}/{relative_path}"
                s3_client.upload_file(local_path, bucket_name, s3_key)
        logger.info(f"检查点已上传到S3 {bucket_name}/online_checkpoints/{model_type}")
    except Exception as e:
        logger.error(f"上传检查点到S3出错: {e}")


def update_als_model(new_data):
    """增量更新ALS模型"""
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

    new_ratings = [
        (int(row['user_id']), int(row['app_id']), float(row['rating']))
        for row in new_data if 'user_id' in row and 'app_id' in row and 'rating' in row
    ]

    if not new_ratings:
        logger.warning("没有新的有效评分数据，ALS模型未更新")
        return als_model, {}

    new_ratings_df = spark.createDataFrame(new_ratings, schema)

    # 获取现有模型的超参数
    current_params = {
        'rank': als_model.rank,
        'regParam': als_model.getRegParam(),
        'alpha': als_model.getAlpha(),
        'maxIter': als_model.getMaxIter()
    }

    # 增量训练ALS模型 - 使用较少的迭代次数进行增量更新
    from pyspark.ml.recommendation import ALS

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

        from pyspark.ml.evaluation import RegressionEvaluator
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
        return updated_model, metrics

    except Exception as e:
        logger.error(f"更新ALS模型时出错: {e}")
        return als_model, {}


def update_tfidf_model(new_games):
    """增量更新TF-IDF模型"""
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

    # 预处理新游戏数据
    from src.content_based import prepare_content_data
    new_processed_games = prepare_content_data(new_game_df)

    # 合并新旧游戏数据
    combined_games = pd.concat([processed_games, new_processed_games], ignore_index=True)

    # 只增量更新TF-IDF，不完全重建
    try:
        # 选项1: 使用现有词汇表转换新内容
        existing_features = tfidf_vectorizer.transform(processed_games['content_features'])
        new_features = tfidf_vectorizer.transform(new_processed_games['content_features'])

        # 合并特征矩阵
        from scipy.sparse import vstack
        combined_features = vstack([existing_features, new_features])

        # 重新计算余弦相似度矩阵
        new_cosine_sim = cosine_similarity(combined_features, combined_features)

        # 更新索引
        new_indices = pd.Series(combined_games.index, index=combined_games['app_id'])

        # 计算处理时间
        processing_time = time.time() - start_time

        metrics = {
            'processing_time': processing_time,
            'num_new_games': len(actual_new_game_ids),
            'total_games': len(combined_games),
            'update_type': 'incremental_tfidf'
        }

        logger.info(
            f"TF-IDF模型更新完成，新增 {len(actual_new_game_ids)} 个游戏，总计 {len(combined_games)} 个游戏，用时: {processing_time:.2f}秒")

        return tfidf_vectorizer, new_cosine_sim, new_indices, metrics

    except Exception as e:
        logger.error(f"增量更新TF-IDF模型时出错: {e}")

        # 选项2（备选）: 如果增量更新失败，则完全重建TF-IDF模型
        try:
            logger.info("尝试完全重建TF-IDF模型")
            from src.content_based import build_tfidf_model
            new_tfidf, new_cosine_sim, new_indices, _ = build_tfidf_model(combined_games)

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


def update_models():
    """根据缓冲区数据更新模型"""
    global als_model, tfidf_vectorizer, cosine_sim, indices, games_df, users_df, recommendations_df, metadata_df, last_update_time, update_count, processed_games

    with buffer_lock:
        # 复制并清空缓冲区
        new_games = data_buffer['games'].copy()
        new_users = data_buffer['users'].copy()
        new_recommendations = data_buffer['recommendations'].copy()
        new_metadata = data_buffer['metadata'].copy()

        data_buffer['games'] = []
        data_buffer['users'] = []
        data_buffer['recommendations'] = []
        data_buffer['metadata'] = []

    if not new_recommendations and not new_games:
        logger.info("缓冲区为空，跳过模型更新")
        return

    logger.info(f"开始更新模型，新数据: {len(new_recommendations)} 条评价, {len(new_games)} 个游戏")
    update_count += 1

    all_metrics = {}

    # 1. 更新游戏和用户数据
    if new_games:
        # 将新游戏添加到全局数据
        new_games_df = pd.DataFrame(new_games)
        if games_df is not None:
            games_df = pd.concat([games_df, new_games_df], ignore_index=True)
            games_df = games_df.drop_duplicates(subset=['app_id'], keep='last')

    if new_users:
        # 将新用户添加到全局数据
        new_users_df = pd.DataFrame(new_users)
        if users_df is not None:
            users_df = pd.concat([users_df, new_users_df], ignore_index=True)
            users_df = users_df.drop_duplicates(subset=['user_id'], keep='last')

    if new_metadata:
        # 将新元数据添加到全局数据
        new_metadata_df = pd.DataFrame(new_metadata)
        if metadata_df is not None:
            metadata_df = pd.concat([metadata_df, new_metadata_df], ignore_index=True)
            metadata_df = metadata_df.drop_duplicates(subset=['app_id'], keep='last')

    # 2. 处理新的推荐数据并转换为评分
    if new_recommendations:
        # 转换为评分
        for rec in new_recommendations:
            if 'is_recommended' in rec and 'hours' in rec:
                rec['rating'] = min(10.0, float(rec['hours']) / 10) * (1.5 if rec['is_recommended'] else 0.5)
            else:
                rec['rating'] = 5.0  # 默认中等评分

        # 将新评价添加到全局数据
        new_ratings_df = pd.DataFrame(new_recommendations)
        if recommendations_df is not None:
            recommendations_df = pd.concat([recommendations_df, new_ratings_df], ignore_index=True)

    # 3. 更新游戏处理数据
    if new_games or new_metadata:
        # 如果游戏数据更新，需要重新处理以准备内容特征
        from src.data_processing import preprocess_data
        processed_games, _, _ = preprocess_data(games_df, users_df, recommendations_df, metadata_df, None)

    # 4. 更新协同过滤模型
    if new_recommendations:
        updated_als, als_metrics = update_als_model(new_recommendations)
        if als_metrics:
            als_model = updated_als
            all_metrics.update(als_metrics)
            save_model_checkpoint("als", als_model, als_metrics)

    # 5. 更新内容过滤模型
    if new_games or new_metadata:
        updated_tfidf, updated_cosine_sim, updated_indices, tfidf_metrics = update_tfidf_model(new_games)
        if tfidf_metrics:
            tfidf_vectorizer = updated_tfidf
            cosine_sim = updated_cosine_sim
            indices = updated_indices
            all_metrics.update(tfidf_metrics)
            save_model_checkpoint("tfidf", tfidf_vectorizer, tfidf_metrics)

    # 记录所有指标
    if all_metrics:
        all_metrics['update_id'] = update_count
        save_metrics(all_metrics)

    last_update_time = datetime.now()
    logger.info(f"模型更新完成，更新ID: {update_count}")


def update_thread_function():
    """模型更新线程函数"""
    logger.info("模型更新线程已启动")

    while not stop_event.is_set():
        try:
            # 检查是否有足够的数据或达到时间间隔
            buffer_size = 0
            with buffer_lock:
                buffer_size = len(data_buffer['recommendations']) + len(data_buffer['games'])

            current_time = time.time()
            time_since_update = float('inf')
            if last_update_time:
                time_since_update = current_time - time.mktime(last_update_time.timetuple())

            if buffer_size >= BUFFER_SIZE_THRESHOLD or time_since_update >= UPDATE_INTERVAL:
                logger.info(f"触发模型更新: 缓冲区大小 = {buffer_size}, 距上次更新 = {time_since_update:.2f}秒")
                update_models()

            # 等待下一次检查
            time.sleep(10)  # 每10秒检查一次

        except Exception as e:
            logger.error(f"更新线程发生错误: {e}")
            time.sleep(30)  # 发生错误时，等待30秒后继续


def start_update_thread():
    """启动模型更新线程"""
    global update_thread, stop_event

    if update_thread is not None and update_thread.is_alive():
        logger.info("更新线程已经在运行")
        return

    stop_event.clear()
    update_thread = threading.Thread(target=update_thread_function)
    update_thread.daemon = True
    update_thread.start()
    logger.info("模型更新线程已启动")


def stop_update_thread():
    """停止模型更新线程"""
    global update_thread, stop_event

    if update_thread is not None and update_thread.is_alive():
        stop_event.set()
        update_thread.join(timeout=5)
        logger.info("模型更新线程已停止")


@app.route('/api/data', methods=['POST'])
def receive_data():
    """接收新数据的API端点"""
    if not INCREMENTAL_LEARNING_ENABLED:
        return jsonify({"status": "error", "message": "增量学习功能已禁用"}), 403

    try:
        data = request.get_json()

        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "无效的数据格式"}), 400

        # 检查并添加到对应的缓冲区
        with buffer_lock:
            if 'games' in data and isinstance(data['games'], list):
                data_buffer['games'].extend(data['games'])

            if 'users' in data and isinstance(data['users'], list):
                data_buffer['users'].extend(data['users'])

            if 'recommendations' in data and isinstance(data['recommendations'], list):
                data_buffer['recommendations'].extend(data['recommendations'])

            if 'metadata' in data and isinstance(data['metadata'], list):
                data_buffer['metadata'].extend(data['metadata'])

            buffer_size = len(data_buffer['recommendations']) + len(data_buffer['games'])

        logger.info(f"接收到新数据: {len(data.get('recommendations', []))} 条评价, "
                    f"{len(data.get('games', []))} 个游戏, 当前缓冲区大小: {buffer_size}")

        # 可选: 如果缓冲区太大，立即触发更新
        if buffer_size >= BUFFER_SIZE_THRESHOLD:
            # 创建线程触发更新，不阻塞API响应
            update_thread = threading.Thread(target=update_models)
            update_thread.daemon = True
            update_thread.start()

        return jsonify({
            "status": "success",
            "message": "数据已接收",
            "buffer_size": buffer_size
        })

    except Exception as e:
        logger.error(f"处理数据请求时出错: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态的API端点"""
    with buffer_lock:
        buffer_size = {
            'recommendations': len(data_buffer['recommendations']),
            'games': len(data_buffer['games']),
            'users': len(data_buffer['users']),
            'metadata': len(data_buffer['metadata']),
            'total': len(data_buffer['recommendations']) + len(data_buffer['games'])
        }

    status = {
        'incremental_learning_enabled': INCREMENTAL_LEARNING_ENABLED,
        'buffer_size': buffer_size,
        'last_update_time': last_update_time.isoformat() if last_update_time else None,
        'update_count': update_count,
        'update_thread_running': update_thread is not None and update_thread.is_alive()
    }

    return jsonify(status)


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """获取性能指标的API端点"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    else:
        return jsonify([])


@app.route('/api/toggle_learning', methods=['POST'])
def toggle_learning():
    """切换增量学习状态"""
    global INCREMENTAL_LEARNING_ENABLED

    try:
        data = request.get_json()
        if 'enabled' in data:
            INCREMENTAL_LEARNING_ENABLED = bool(data['enabled'])

            if INCREMENTAL_LEARNING_ENABLED:
                start_update_thread()
            else:
                stop_update_thread()

            return jsonify({
                "status": "success",
                "enabled": INCREMENTAL_LEARNING_ENABLED
            })
        else:
            return jsonify({"status": "error", "message": "参数错误"}), 400

    except Exception as e:
        logger.error(f"切换学习状态时出错: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def start_flask_server(port=5000):
    """启动Flask服务器"""
    initialize_system()
    start_update_thread()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    start_flask_server()