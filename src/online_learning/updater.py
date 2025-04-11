# src/online_learning/updater.py
"""
模型更新线程和控制逻辑。
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime

from . import BUFFER_SIZE_THRESHOLD, UPDATE_INTERVAL
from .buffer import extract_buffer_data, get_buffer_size
from .models import (update_als_model, update_tfidf_model, get_global_models_and_data,
                     update_model_data)
from .storage import save_metrics, save_model_checkpoint

# 设置日志
logger = logging.getLogger("online_learning.updater")

# 全局变量
update_thread = None
stop_event = threading.Event()
metrics_history = deque(maxlen=100)  # 保存最近100次更新的指标
last_update_time = None
update_count = 0
INCREMENTAL_LEARNING_ENABLED = True


def update_models():
    """
    根据缓冲区数据更新模型

    执行步骤:
    1. 从缓冲区提取数据
    2. 更新全局数据
    3. 更新协同过滤模型
    4. 更新内容过滤模型
    5. 记录性能指标

    Returns:
        dict: 更新指标和结果
    """
    global last_update_time, update_count

    # 从缓冲区提取数据
    buffer_data = extract_buffer_data()

    # 如果没有新数据，跳过更新
    if not buffer_data['recommendations'] and not buffer_data['games']:
        logger.info("缓冲区为空，跳过模型更新")
        return {}

    logger.info(
        f"开始更新模型，新数据: {len(buffer_data['recommendations'])} 条评价, {len(buffer_data['games'])} 个游戏, "
        f"{len(buffer_data['users'])} 个用户, {len(buffer_data['metadata'])} 条元数据")
    update_count += 1

    all_metrics = {'update_id': update_count}

    # 获取全局模型和数据
    (als_model, tfidf_vectorizer, cosine_sim, indices,
     games_df, users_df, recommendations_df, metadata_df, processed_games) = get_global_models_and_data()

    # 更新全局数据
    update_model_data(buffer_data)

    # 记录数据更新指标
    if buffer_data['games']:
        all_metrics['num_new_games'] = len(buffer_data['games'])
        all_metrics['total_games'] = len(games_df) if games_df is not None else 0

    if buffer_data['users']:
        all_metrics['num_new_users'] = len(buffer_data['users'])
        all_metrics['total_users'] = len(users_df) if users_df is not None else 0

    if buffer_data['recommendations']:
        all_metrics['num_samples'] = len(buffer_data['recommendations'])
        all_metrics['total_recommendations'] = len(recommendations_df) if recommendations_df is not None else 0

    # 更新协同过滤模型
    if buffer_data['recommendations']:
        updated_als, als_metrics = update_als_model(buffer_data['recommendations'])
        if als_metrics:
            all_metrics.update(als_metrics)
            save_model_checkpoint("als", updated_als, als_metrics)

    # 更新内容过滤模型
    if buffer_data['games'] or buffer_data['metadata']:
        updated_tfidf, updated_cosine_sim, updated_indices, tfidf_metrics = update_tfidf_model(buffer_data['games'])
        if tfidf_metrics:
            all_metrics.update(tfidf_metrics)
            save_model_checkpoint("tfidf", updated_tfidf, tfidf_metrics)

    # 记录所有指标
    all_metrics['timestamp'] = datetime.now().isoformat()
    save_metrics(all_metrics)
    metrics_history.append(all_metrics)

    last_update_time = datetime.now()
    logger.info(f"模型更新完成，更新ID: {update_count}")

    return all_metrics


def update_thread_function():
    """
    模型更新线程函数

    定期检查是否需要更新模型:
    - 当缓冲区数据量达到阈值时
    - 当距离上次更新时间超过设定间隔时
    """
    logger.info("模型更新线程已启动")

    while not stop_event.is_set():
        try:
            # 检查是否有足够的数据或达到时间间隔
            buffer_size = get_buffer_size()

            current_time = time.time()
            time_since_update = float('inf')
            if last_update_time:
                time_since_update = current_time - time.mktime(last_update_time.timetuple())

            if buffer_size['total'] >= BUFFER_SIZE_THRESHOLD or time_since_update >= UPDATE_INTERVAL:
                if buffer_size['total'] > 0:  # 只有当缓冲区有数据时才更新
                    logger.info(
                        f"触发模型更新: 缓冲区大小 = {buffer_size['total']}, 距上次更新 = {time_since_update:.2f}秒")
                    update_models()
                else:
                    logger.info(f"跳过更新: 缓冲区为空，距上次更新 = {time_since_update:.2f}秒")

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


def is_update_thread_running():
    """
    检查更新线程是否正在运行

    Returns:
        bool: 如果线程正在运行则为True
    """
    return update_thread is not None and update_thread.is_alive()


def toggle_learning(enabled):
    """
    切换增量学习状态

    Args:
        enabled (bool): 是否启用增量学习

    Returns:
        bool: 操作后的增量学习状态
    """
    global INCREMENTAL_LEARNING_ENABLED

    INCREMENTAL_LEARNING_ENABLED = enabled

    if INCREMENTAL_LEARNING_ENABLED:
        start_update_thread()
    else:
        stop_update_thread()

    logger.info(f"增量学习已{'启用' if INCREMENTAL_LEARNING_ENABLED else '禁用'}")
    return INCREMENTAL_LEARNING_ENABLED


def is_learning_enabled():
    """
    获取增量学习是否启用

    Returns:
        bool: 如果增量学习已启用则为True
    """
    return INCREMENTAL_LEARNING_ENABLED


def get_last_update_time():
    """
    获取上次更新时间

    Returns:
        datetime: 上次更新时间，如果从未更新过则为None
    """
    return last_update_time


def get_update_count():
    """
    获取更新次数

    Returns:
        int: 更新计数器值
    """
    return update_count


def get_metrics_history():
    """
    获取指标历史记录

    Returns:
        list: 指标历史记录列表
    """
    return list(metrics_history)


def trigger_update():
    """
    手动触发模型更新

    Returns:
        dict: 更新指标和结果，如果未启用增量学习则为空
    """
    if not INCREMENTAL_LEARNING_ENABLED:
        logger.warning("增量学习已禁用，无法触发更新")
        return {}

    buffer_size = get_buffer_size()
    if buffer_size['total'] == 0:
        logger.warning("缓冲区为空，无需更新")
        return {}

    logger.info("手动触发模型更新")
    return update_models()