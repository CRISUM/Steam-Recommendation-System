# src/online_learning/buffer.py
import threading
import logging

# 设置日志
logger = logging.getLogger("online_learning.buffer")

# 全局变量: 数据缓冲区
data_buffer = {
    'games': [],
    'users': [],
    'recommendations': [],
    'metadata': []
}

# 用于保护缓冲区的线程锁
buffer_lock = threading.Lock()


def add_to_buffer(data_type, items):
    """
    添加数据到指定类型的缓冲区

    Args:
        data_type (str): 数据类型，可以是'games', 'users', 'recommendations', 'metadata'
        items (list): 要添加的数据项列表

    Returns:
        int: 添加后该类型缓冲区的大小
    """
    if data_type not in data_buffer:
        logger.warning(f"尝试添加到未知缓冲区类型: {data_type}")
        return 0

    if not items:
        return len(data_buffer[data_type])

    with buffer_lock:
        data_buffer[data_type].extend(items)
        buffer_size = len(data_buffer[data_type])

    logger.debug(f"已添加 {len(items)} 条数据到 {data_type} 缓冲区，当前大小: {buffer_size}")
    return buffer_size


def get_buffer_size():
    """
    获取所有缓冲区的大小

    Returns:
        dict: 包含各缓冲区大小的字典，以及总大小
    """
    with buffer_lock:
        buffer_size = {
            'recommendations': len(data_buffer['recommendations']),
            'games': len(data_buffer['games']),
            'users': len(data_buffer['users']),
            'metadata': len(data_buffer['metadata']),
            'total': len(data_buffer['recommendations']) + len(data_buffer['games'])
        }

    return buffer_size


def clear_buffer():
    """
    清空所有缓冲区

    Returns:
        dict: 清空后的缓冲区大小
    """
    with buffer_lock:
        for key in data_buffer:
            data_buffer[key] = []

    logger.info("数据缓冲区已清空")
    return get_buffer_size()


def extract_buffer_data():
    """
    提取并清空缓冲区中的所有数据

    Returns:
        dict: 包含所有缓冲区数据的字典
    """
    with buffer_lock:
        # 创建缓冲区数据的副本
        extracted_data = {
            'games': data_buffer['games'].copy(),
            'users': data_buffer['users'].copy(),
            'recommendations': data_buffer['recommendations'].copy(),
            'metadata': data_buffer['metadata'].copy()
        }

        # 清空缓冲区
        for key in data_buffer:
            data_buffer[key] = []

    logger.info(f"已提取缓冲区数据: {len(extracted_data['recommendations'])} 条评价, "
                f"{len(extracted_data['games'])} 个游戏, {len(extracted_data['users'])} 个用户, "
                f"{len(extracted_data['metadata'])} 条元数据")

    return extracted_data


def peek_buffer_data():
    """
    查看缓冲区数据但不清空

    Returns:
        dict: 包含所有缓冲区数据副本的字典
    """
    with buffer_lock:
        # 创建缓冲区数据的副本但不清空
        buffer_copy = {
            'games': data_buffer['games'].copy(),
            'users': data_buffer['users'].copy(),
            'recommendations': data_buffer['recommendations'].copy(),
            'metadata': data_buffer['metadata'].copy()
        }

    return buffer_copy