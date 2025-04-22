#!/bin/bash
cd /home/hadoop/Steam-Recommendation-System

# 创建zip包含所有依赖
zip -r src.zip src

# 运行主训练脚本 - 优化 m5.xlarge 的配置并添加S3A支持
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
  --conf spark.default.parallelism=20 \
  --conf spark.sql.shuffle.partitions=20 \
  --conf spark.hadoop.mapred.output.committer.class=org.apache.hadoop.mapred.FileOutputCommitter
  --py-files src.zip \
  main.py

echo "主训练脚本已提交"