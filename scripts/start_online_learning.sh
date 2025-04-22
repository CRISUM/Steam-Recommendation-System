#!/bin/bash
# -*- coding: utf-8 -*-
cd /home/hadoop/Steam-Recommendation-System

# 创建zip包含所有依赖
zip -r src.zip src

# 启动在线学习服务 - 使用client模式以便API可以对外可见
spark-submit \
  --master yarn \
  --deploy-mode client \
  --executor-memory 8g \
  --driver-memory 8g \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memoryOverhead=2g \
  --conf spark.driver.memoryOverhead=2g \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.shuffle.service.enabled=true \
  --py-files src.zip \
  src/online_learning/main.py \
  --data-path s3://steam-project-data-976193243904 \
  --model-path s3://steam-project-data-976193243904/models/als_model \
  --port 5000 \
  --host 0.0.0.0 \
  --use-s3

echo "在线学习服务已启动"