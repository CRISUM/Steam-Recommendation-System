#!/bin/bash
cd /home/hadoop/Steam-Recommendation-System

# 创建zip包含所有依赖
zip -r src.zip src

# 运行主训练脚本 - 优化 m5.xlarge 的配置
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 8g \
  --driver-memory 8g \
  --conf spark.executor.cores=4 \
  --conf spark.executor.memoryOverhead=2g \
  --conf spark.driver.memoryOverhead=2g \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.memory.fraction=0.8 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.default.parallelism=20 \
  --conf spark.sql.shuffle.partitions=20 \
  --conf spark.driver.extraJavaOptions="-Divy.cache.dir=/tmp -Divy.home=/tmp" \
  --py-files src.zip \
  main.py

echo "主训练脚本已提交"