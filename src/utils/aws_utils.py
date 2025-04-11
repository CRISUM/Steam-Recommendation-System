# src/utils/aws_utils.py
import boto3
import os
import logging

logger = logging.getLogger("aws_utils")

def get_s3_client():
    """获取配置好的S3客户端"""
    return boto3.client('s3')

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