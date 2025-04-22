#!/bin/bash
# run_main_s.sh - 小型数据集版本
cd /home/hadoop/Steam-Recommendation-System

# 创建zip包含所有依赖
zip -r src.zip src

# 运行小型数据集版本的主脚本 - 针对小数据集优化的配置
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4g \
  --driver-memory 4g \
  --conf spark.executor.cores=2 \
  --conf spark.executor.memoryOverhead=1g \
  --conf spark.driver.memoryOverhead=1g \
  --conf spark.driver.maxResultSize=2g \
  --conf spark.memory.fraction=0.8 \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.endpoint=s3.amazonaws.com \
  --conf spark.hadoop.fs.s3a.path.style.access=false \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=true \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider,com.amazonaws.auth.InstanceProfileCredentialsProvider \
  --conf spark.jars.packages=org.apache.hadoop:hadoop-aws:3.2.2,com.amazonaws:aws-java-sdk-bundle:1.11.901 \
  --conf spark.pyspark.python=/usr/bin/python3 \
  --conf spark.pyspark.driver.python=/usr/bin/python3 \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.default.parallelism=10 \
  --conf spark.sql.shuffle.partitions=10 \
  --conf spark.hadoop.mapred.output.committer.class=org.apache.hadoop.mapred.FileOutputCommitter \
  --py-files src.zip,create_small_dataset_s.py \
  main_s.py

echo "小型数据集推荐系统训练已提交"