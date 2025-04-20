# src/utils/aws_utils.py
import boto3
import os
import logging

logger = logging.getLogger("aws_utils")


def get_s3_client():
    """获取配置好的S3客户端"""
    return boto3.client('s3')


def is_emr_cluster_mode():
    """检测是否在EMR集群上运行"""
    return "AWS_EMR_CLUSTER_ID" in os.environ

def get_storage_path(local_path, s3_bucket="steam-project-data-976193243904"):
    """
    根据运行环境返回适当的存储路径

    Args:
        local_path: 本地路径
        s3_bucket: S3存储桶名称

    Returns:
        适用于当前环境的路径
    """
    # 检查是否在EMR集群上运行
    if is_emr_cluster_mode():
        # 替换路径分隔符并移除前导斜杠
        s3_path = local_path.replace("\\", "/")
        if s3_path.startswith("/"):
            s3_path = s3_path[1:]
        # 移除可能的驱动器前缀 (如 "C:")
        if ":" in s3_path:
            s3_path = s3_path.split(":", 1)[1].lstrip("/")

        return f"s3://{s3_bucket}/{s3_path}"
    else:
        return local_path


def ensure_dir(path):
    """确保目录存在，只在本地模式下操作"""
    if not path.startswith("s3://"):
        os.makedirs(os.path.dirname(path) if "." in os.path.basename(path) else path, exist_ok=True)
    return path


def upload_file_to_s3(local_path, bucket_name, s3_key):
    """上传文件到S3"""
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(local_path, bucket_name, s3_key)
        logger.info(f"文件已上传到S3: {bucket_name}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"上传文件到S3时出错: {e}")
        return False


def download_from_s3(bucket_name, s3_key, local_path):
    """从S3下载文件"""
    try:
        s3_client = get_s3_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"文件已从S3下载: {bucket_name}/{s3_key} -> {local_path}")
        return True
    except Exception as e:
        logger.error(f"从S3下载文件时出错: {e}")
        return False


def save_to_storage(content, path):
    """保存内容到合适的存储位置"""
    if path.startswith("s3://"):
        # 解析S3路径
        s3_path = path.replace("s3://", "")
        bucket_name = s3_path.split("/")[0]
        key = "/".join(s3_path.split("/")[1:])

        # 使用boto3保存
        s3_client = get_s3_client()
        s3_client.put_object(Body=content, Bucket=bucket_name, Key=key)
        logger.info(f"内容已保存到S3: {bucket_name}/{key}")
    else:
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        logger.info(f"内容已保存到本地: {path}")