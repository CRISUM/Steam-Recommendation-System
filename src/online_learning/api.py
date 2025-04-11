# src/online_learning/api.py
from flask import Flask, request, jsonify
import os
import logging
import threading
from datetime import datetime

from .buffer import add_to_buffer, get_buffer_size, clear_buffer
from .updater import (
    is_learning_enabled, toggle_learning,
    get_last_update_time, get_update_count,
    is_update_thread_running, trigger_update,
    get_metrics_history
)

# 设置日志
logger = logging.getLogger("online_learning.api")

# 初始化Flask应用
app = Flask(__name__)


@app.route('/api/data', methods=['POST'])
def receive_data():
    """接收新数据的API端点"""
    if not is_learning_enabled():
        return jsonify({"status": "error", "message": "增量学习功能已禁用"}), 403

    try:
        data = request.get_json()

        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "无效的数据格式"}), 400

        # 添加数据到缓冲区
        if 'games' in data and isinstance(data['games'], list):
            add_to_buffer('games', data['games'])

        if 'users' in data and isinstance(data['users'], list):
            add_to_buffer('users', data['users'])

        if 'recommendations' in data and isinstance(data['recommendations'], list):
            add_to_buffer('recommendations', data['recommendations'])

        if 'metadata' in data and isinstance(data['metadata'], list):
            add_to_buffer('metadata', data['metadata'])

        # 获取缓冲区大小
        buffer_size = get_buffer_size()

        logger.info(f"接收到新数据: {len(data.get('recommendations', []))} 条评价, "
                    f"{len(data.get('games', []))} 个游戏, 当前缓冲区大小: {buffer_size['total']}")

        # 如果缓冲区太大，立即触发更新
        from . import BUFFER_SIZE_THRESHOLD
        if buffer_size['total'] >= BUFFER_SIZE_THRESHOLD:
            # 创建线程触发更新，不阻塞API响应
            threading.Thread(target=trigger_update, daemon=True).start()

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
    buffer_size = get_buffer_size()

    last_update = get_last_update_time()

    status = {
        'incremental_learning_enabled': is_learning_enabled(),
        'buffer_size': buffer_size,
        'last_update_time': last_update.isoformat() if last_update else None,
        'update_count': get_update_count(),
        'update_thread_running': is_update_thread_running()
    }

    return jsonify(status)


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """获取性能指标的API端点"""
    from . import METRICS_FILE

    # 优先从内存中获取最新指标
    metrics_history = get_metrics_history()
    if metrics_history:
        return jsonify(list(metrics_history))

    # 如果内存中没有，尝试从文件读取
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                import json
                metrics = json.load(f)
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"读取指标文件时出错: {e}")

    # 都没有则返回空列表
    return jsonify([])


@app.route('/api/toggle_learning', methods=['POST'])
def api_toggle_learning():
    """切换增量学习状态"""
    try:
        data = request.get_json()
        if 'enabled' in data:
            enabled = bool(data['enabled'])
            toggle_learning(enabled)

            return jsonify({
                "status": "success",
                "enabled": is_learning_enabled()
            })
        else:
            return jsonify({"status": "error", "message": "参数错误"}), 400

    except Exception as e:
        logger.error(f"切换学习状态时出错: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/clear_buffer', methods=['POST'])
def api_clear_buffer():
    """清空数据缓冲区"""
    clear_buffer()

    return jsonify({
        "status": "success",
        "message": "缓冲区已清空",
        "buffer_size": get_buffer_size()
    })


@app.route('/api/trigger_update', methods=['POST'])
def api_trigger_update():
    """手动触发模型更新"""
    if not is_learning_enabled():
        return jsonify({"status": "error", "message": "增量学习功能已禁用"}), 403

    buffer_size = get_buffer_size()
    if buffer_size['total'] == 0:
        return jsonify({"status": "warning", "message": "缓冲区为空，无需更新"}), 200

    # 在单独线程中更新模型
    threading.Thread(target=trigger_update, daemon=True).start()

    return jsonify({
        "status": "success",
        "message": "模型更新已触发"
    })