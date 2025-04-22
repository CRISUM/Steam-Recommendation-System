# main_s.py - 小型数据集版本
import os
import json
import time
import sys
import boto3

import pandas as pd

import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 添加初始环境诊断
print("\n" + "=" * 50)
print("STEAM RECOMMENDATION SYSTEM (SMALL) - ENVIRONMENT DIAGNOSTIC")
print("=" * 50)
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Script path: {os.path.abspath(__file__)}")

# 检查主要环境变量
env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_EMR_CLUSTER_ID",
            "HADOOP_CONF_DIR", "YARN_CONF_DIR", "SPARK_HOME"]
print("\nEnvironment variables:")
for var in env_vars:
    status = "SET" if os.environ.get(var) else "NOT SET"
    print(f"  {var}: {status}")

# 标记诊断部分结束
print("=" * 50 + "\n")

from src import initialize_spark, load_data, preprocess_data, split_data
from src import build_als_model, evaluate_als_model
from src import build_tfidf_model
from src import build_hybrid_recommender
from src import compare_recommenders, visualize_comparison, save_evaluation_results
from src import build_popularity_model, build_content_based_cold_start
from src.utils.aws_utils import get_storage_path, is_emr_cluster_mode, ensure_dir

# 导入小型数据集创建脚本
from create_small_dataset_s import create_small_dataset_s


def main():
    # 记录开始时间
    start_time = time.time()

    # 数据路径 - 指向小型数据集
    data_path = "s3://steam-project-data-976193243904/data_small_s"

    # 创建小型数据集（如果需要）
    if len(sys.argv) > 1 and sys.argv[1] == "--create-dataset":
        print("创建小型数据集...")
        create_small_dataset_s(
            original_data_path="s3://steam-project-data-976193243904",
            small_data_path=data_path,
            n_games=200,  # 只使用200个游戏
            n_users=100,  # 只使用100个用户
            n_ratings=1000  # 只使用1000条评价
        )

    # Spark 格式数据路径
    train_data_path = "s3://steam-project-data-976193243904/data_small_s/spark_train_data"
    test_data_path = "s3://steam-project-data-976193243904/data_small_s/spark_test_data"

    # 结果保存路径
    results_path = "s3://steam-project-data-976193243904/data_small_s/results"
    figures_path = "s3://steam-project-data-976193243904/data_small_s/results/figures"
    models_path = "s3://steam-project-data-976193243904/data_small_s/models"

    # 初始化Spark
    print("初始化Spark...")
    spark = initialize_spark()

    # 加载小型数据集
    print(f"从 {data_path} 加载数据...")
    games_df, users_df, recommendations_df, metadata_df = load_data(data_path)

    print(f"已加载 {len(games_df)} 个游戏, {len(users_df)} 个用户, "
          f"{len(recommendations_df)} 条评价, {len(metadata_df)} 条游戏元数据")

    # 数据预处理
    print("预处理数据...")
    games_with_metadata, spark_ratings, processed_recommendations = preprocess_data(
        games_df, users_df, recommendations_df, metadata_df, spark
    )

    # 分割数据
    print("分割数据为训练集和测试集...")
    train_data, test_data = split_data(processed_recommendations, test_ratio=0.2, random_state=42)

    # 如果Spark格式数据不存在，创建并保存
    spark_data_exists = False
    try:
        # 尝试列出S3存储桶中的文件
        s3_client = boto3.client('s3')
        bucket_name = train_data_path.replace("s3://", "").split("/")[0]
        prefix = "/".join(train_data_path.replace("s3://", "").split("/")[1:])

        # 检查训练数据
        train_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=1
        )

        # 检查测试数据
        test_prefix = "/".join(test_data_path.replace("s3://", "").split("/")[1:])
        test_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=test_prefix,
            MaxKeys=1
        )

        # 判断两个数据集是否都存在
        if 'Contents' in train_response and 'Contents' in test_response:
            spark_data_exists = True
            print("S3上发现已有Spark格式数据")
    except Exception as e:
        print(f"检查S3 Spark数据时出错: {e}")
        spark_data_exists = False

    # 根据检查结果加载或处理数据
    if spark_data_exists:
        # 直接从S3读取已处理的Parquet文件
        print("从S3加载Spark格式数据...")
        try:
            spark_train = spark.read.parquet(train_data_path)
            spark_test = spark.read.parquet(test_data_path)
            print(f"成功加载Spark训练数据: {spark_train.count()} 条记录")
            print(f"成功加载Spark测试数据: {spark_test.count()} 条记录")
        except Exception as e:
            print(f"加载S3 Spark数据失败: {e}")
            spark_data_exists = False  # 标记为不存在，需要重新处理

    # 如果S3上没有有效的Spark数据，则需要处理并存储
    if not spark_data_exists:
        print("将训练和测试数据转换为Spark格式并存储...")

        # 处理并存储训练数据
        train_spark = spark.createDataFrame(train_data[['user_id', 'app_id', 'rating']])
        train_spark.write.parquet(train_data_path, mode="overwrite")

        # 处理并存储测试数据
        test_spark = spark.createDataFrame(test_data[['user_id', 'app_id', 'rating']])
        test_spark.write.parquet(test_data_path, mode="overwrite")

        print(f"Spark训练数据已保存: {train_spark.count()} 条记录")
        print(f"Spark测试数据已保存: {test_spark.count()} 条记录")

        # 读取刚存储的数据
        spark_train = train_spark
        spark_test = test_spark

    # 使用预设参数 - 为小数据集优化
    rank = 10  # 降低特征维度
    reg_param = 0.1  # 正则化参数
    alpha = 1.0  # 隐式反馈参数
    max_iter = 5  # 减少迭代次数，加快收敛

    # 训练ALS协同过滤模型
    print(f"训练ALS模型 (rank={rank}, regParam={reg_param}, alpha={alpha}, maxIter={max_iter})...")
    als_model = build_als_model(spark_train, rank, reg_param, alpha, max_iter)

    # 评估ALS模型
    try:
        print("评估ALS模型...")
        als_metrics = evaluate_als_model(als_model, spark_test)
        print(f"ALS模型评估结果: {als_metrics}")
    except Exception as e:
        print(f"评估ALS模型时出错: {e}")
        # 使用默认值继续执行
        als_metrics = {"RMSE": float('nan'), "MAE": float('nan')}

    # 构建TF-IDF模型 - 降低特征数量
    print("构建TF-IDF模型...")
    tfidf, cosine_sim, indices, content_df = build_tfidf_model(games_with_metadata, max_features=1000)

    # 构建混合推荐模型
    print("构建混合推荐模型...")
    hybrid_recommender = build_hybrid_recommender(
        als_model, cosine_sim, indices, games_with_metadata,
        train_data, 0.7, spark
    )

    # 创建纯协同过滤推荐模型
    pure_cf_model = build_hybrid_recommender(
        als_model, cosine_sim, indices, games_with_metadata, train_data, 1.0, spark
    )

    # 创建纯内容推荐模型
    pure_content_model = build_hybrid_recommender(
        als_model, cosine_sim, indices, games_with_metadata, train_data, 0.0, spark
    )

    # 创建流行度推荐模型（用于冷启动）
    popularity_recommender = build_popularity_model(train_data, games_with_metadata)

    # 创建基于内容的冷启动推荐
    content_cold_start = build_content_based_cold_start(cosine_sim, indices, games_with_metadata)

    # 保存ALS模型
    als_model_path = models_path + "/als_model"
    print(f"保存ALS模型到 {als_model_path}...")

    try:
        als_model.save(als_model_path)
        print("ALS模型已保存")
    except Exception as e:
        print(f"保存ALS模型时出错: {e}")

    # 评估所有模型
    print("评估所有推荐模型...")

    # 使用所有测试用户，因为数据集很小
    test_users = test_data['user_id'].unique()
    print(f"使用 {len(test_users)} 个测试用户进行评估")

    # 创建所有推荐器字典
    all_recommenders = {
        "Pure_CF": pure_cf_model,
        "Pure_Content": pure_content_model,
        "Hybrid_0.7": hybrid_recommender,
        "Popularity": popularity_recommender,
    }

    # 比较所有推荐器
    evaluation_results = compare_recommenders(
        all_recommenders, test_users, games_with_metadata, test_data
    )

    # 保存评估结果
    print("保存评估结果...")
    eval_results_path = f"{results_path}/evaluation_results.json"
    save_evaluation_results(
        evaluation_results,
        eval_results_path
    )

    # 示例：为一些用户生成推荐
    print("\n推荐示例:")

    # 选择一个测试用户
    if len(test_users) > 0:
        example_user = test_users[0]
        print(f"\n为用户 {example_user} 生成推荐:")

        # 使用混合模型生成推荐
        hybrid_recommendations = hybrid_recommender(example_user, 5)
        if not hybrid_recommendations.empty:
            print("混合推荐结果:")
            print(hybrid_recommendations[['app_id', 'title']].to_string(index=False))
        else:
            print("混合模型无法生成推荐")

        # 使用协同过滤生成推荐
        cf_recommendations = pure_cf_model(example_user, 5)
        if not cf_recommendations.empty:
            print("\n纯协同过滤推荐结果:")
            print(cf_recommendations[['app_id', 'title']].to_string(index=False))
        else:
            print("\n协同过滤模型无法生成推荐")

    # 流行度推荐（冷启动）
    popular_games = popularity_recommender(5)
    print("\n热门游戏推荐（适用于新用户）:")
    print(popular_games[['app_id', 'title']].to_string(index=False))

    # 为一个游戏推荐相似游戏
    if len(games_df) > 0:
        example_game_id = games_df['app_id'].iloc[0]
        example_game_title = games_df.loc[games_df['app_id'] == example_game_id, 'title'].values[0]

        print(f"\n为游戏 '{example_game_title}' (ID: {example_game_id}) 推荐相似游戏:")
        similar_games = content_cold_start(example_game_id, 5)
        if not similar_games.empty:
            print(similar_games[['app_id', 'title', 'similarity_score']].to_string(index=False))
        else:
            print("没有找到相似游戏。")

    # 关闭Spark会话
    print("关闭Spark会话...")
    spark.stop()

    # 打印总运行时间
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\n总运行时间: {run_time:.2f} 秒 ({run_time / 60:.2f} 分钟)")

    print("\n完成！小型数据集推荐系统模型已训练并评估。")
    print(f"模型保存在: {models_path}")
    print(f"评估结果保存在: {eval_results_path}")


if __name__ == "__main__":
    main()