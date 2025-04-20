# main.py
import os
import json
import time
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 添加初始环境诊断
print("\n" + "="*50)
print("STEAM RECOMMENDATION SYSTEM - ENVIRONMENT DIAGNOSTIC")
print("="*50)
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
print("="*50 + "\n")

from src import initialize_spark, load_data, preprocess_data, split_data
from src import build_als_model, evaluate_als_model, tune_als_parameters
from src import build_tfidf_model
from src import build_hybrid_recommender
from src import compare_recommenders, visualize_comparison, save_evaluation_results
from src import build_popularity_model, build_content_based_cold_start
import boto3
from src.utils.aws_utils import get_storage_path, is_emr_cluster_mode, ensure_dir, save_to_storage


def save_checkpoint(model, step, metrics, bucket_name="steam-project-data-976193243904"):
    """保存训练检查点到本地和S3"""
    # 使用通用路径函数获取适当的路径
    checkpoint_path = get_storage_path(f"checkpoints/checkpoint_{step}")

    # 如果是本地模式，确保目录存在
    if not checkpoint_path.startswith("s3://"):
        os.makedirs(checkpoint_path, exist_ok=True)

        # 保存模型
        model.save(os.path.join(checkpoint_path, "model"))

        # 保存训练进度和指标
        with open(os.path.join(checkpoint_path, "metrics.json"), 'w') as f:
            json.dump(metrics, f)

        # 上传到S3
        try:
            s3_client = boto3.client('s3')
            for root, dirs, files in os.walk(checkpoint_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, "checkpoints")
                    s3_key = f"checkpoints/{relative_path}"
                    s3_client.upload_file(local_path, bucket_name, s3_key)
            print(f"检查点 {step} 已保存到S3")
        except Exception as e:
            print(f"保存检查点到S3时出错: {e}")
    else:
        # 直接保存到S3
        s3_bucket = checkpoint_path.split("/")[2]
        s3_prefix = "/".join(checkpoint_path.split("/")[3:])

        # 保存模型到S3
        # 注意：需要先本地保存然后上传，因为Spark的save方法不直接支持S3路径
        temp_path = f"temp_checkpoint_{step}"
        os.makedirs(temp_path, exist_ok=True)

        try:
            # 本地临时保存
            model.save(os.path.join(temp_path, "model"))

            # 写入metrics.json
            with open(os.path.join(temp_path, "metrics.json"), 'w') as f:
                json.dump(metrics, f)

            # 上传到S3
            s3_client = boto3.client('s3')
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_path)
                    s3_key = f"{s3_prefix}/{relative_path}"
                    s3_client.upload_file(local_path, s3_bucket, s3_key)

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_path)

            print(f"检查点 {step} 已保存到S3: {checkpoint_path}")
        except Exception as e:
            print(f"保存检查点到S3时出错: {e}")

            # 清理临时文件
            import shutil
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)


def main():
    # 记录开始时间
    start_time = time.time()

    # 数据路径 - 始终指向S3
    data_path = "s3://steam-project-data-976193243904"

    # 结果保存路径 - 根据环境使用本地或S3
    results_path = get_storage_path("results")
    figures_path = get_storage_path("results/figures")
    models_path = get_storage_path("models")

    # 确保本地目录存在 (如果在本地运行)
    if not is_emr_cluster_mode():
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(figures_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

    # 初始化Spark
    print("初始化Spark...")
    spark = initialize_spark()

    # 加载数据
    print("加载数据...")
    games_df, users_df, recommendations_df, metadata_df = load_data(data_path)

    # 数据预处理
    print("预处理数据...")
    games_with_metadata, spark_ratings, processed_recommendations = preprocess_data(
        games_df, users_df, recommendations_df, metadata_df, spark
    )

    # 分割数据
    print("分割数据为训练集和测试集...")
    train_data, test_data = split_data(processed_recommendations)

    # 保存训练和测试数据
    data_dir = get_storage_path("data/processed")
    if not data_dir.startswith("s3://"):
        os.makedirs(data_dir, exist_ok=True)

        # 保存为CSV
        train_data.to_csv(f"{data_dir}/train_data.csv", index=False)
        test_data.to_csv(f"{data_dir}/test_data.csv", index=False)
        print(f"训练和测试数据已保存到 {data_dir}")
    else:
        # S3保存逻辑
        s3_bucket = data_dir.replace("s3://", "").split("/")[0]
        s3_prefix = "/".join(data_dir.replace("s3://", "").split("/")[1:])

        # 创建临时文件
        train_data.to_csv("temp_train_data.csv", index=False)
        test_data.to_csv("temp_test_data.csv", index=False)

        # 上传到S3
        s3_client = boto3.client('s3')
        s3_client.upload_file("temp_train_data.csv", s3_bucket, f"{s3_prefix}/train_data.csv")
        s3_client.upload_file("temp_test_data.csv", s3_bucket, f"{s3_prefix}/test_data.csv")

        # 删除临时文件
        os.remove("temp_train_data.csv")
        os.remove("temp_test_data.csv")
        print(f"训练和测试数据已上传到 S3: {data_dir}")

    # 创建Spark格式的训练和测试数据
    spark_train = spark.createDataFrame(train_data[['user_id', 'app_id', 'rating']])
    spark_test = spark.createDataFrame(test_data[['user_id', 'app_id', 'rating']])

    # 是否进行超参数调优（可选，耗时较长）
    tune_parameters = False

    if tune_parameters:
        print("进行ALS模型参数调优...")
        best_params, tuning_results = tune_als_parameters(spark_train)
        rank = best_params['rank']
        reg_param = best_params['regParam']
        alpha = best_params['alpha']
    else:
        # 使用预设参数
        rank = 20
        reg_param = 0.1
        alpha = 1.0

    # 训练ALS协同过滤模型
    print(f"训练ALS模型 (rank={rank}, regParam={reg_param}, alpha={alpha})...")
    als_model = build_als_model(spark_train, rank, reg_param, alpha)

    # 评估ALS模型
    print("评估ALS模型...")
    als_metrics = evaluate_als_model(als_model, spark_test)
    print(f"ALS模型评估结果: {als_metrics}")

    # 在这里添加保存检查点
    save_checkpoint(als_model, "als_complete", als_metrics)

    # 构建TF-IDF模型
    print("构建TF-IDF模型...")
    tfidf, cosine_sim, indices, content_df = build_tfidf_model(games_with_metadata)

    # 在这里添加保存检查点（仅保存指标，不保存模型）
    save_checkpoint(None, "tfidf_complete", {"status": "complete"})

    # 构建混合推荐模型
    print("构建混合推荐模型...")
    # 测试不同的混合权重
    hybrid_models = {}
    for weight in [0.3, 0.5, 0.7, 0.9]:
        print(f"创建混合模型 (CF权重={weight})...")
        hybrid_models[f"Hybrid_{weight}"] = build_hybrid_recommender(
            als_model, cosine_sim, indices, games_with_metadata, train_data, weight, spark
        )

    # 在这里添加保存检查点
    save_checkpoint(None, "hybrid_complete", {"hybrid_weights": [0.3, 0.5, 0.7, 0.9]})

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

    # 评估所有模型
    print("评估所有推荐模型...")

    # 选择一部分测试用户
    test_users = test_data['user_id'].unique()[:100]  # 限制为100个用户加速评估

    # 创建所有推荐器字典
    all_recommenders = {
        "Pure_CF": pure_cf_model,
        "Pure_Content": pure_content_model,
        "Popularity": popularity_recommender,
    }

    # 添加混合模型
    all_recommenders.update(hybrid_models)

    # 比较所有推荐器
    evaluation_results = compare_recommenders(
        all_recommenders, test_users, games_with_metadata, test_data
    )

    # 可视化比较结果
    print("可视化评估结果...")
    figure_path = os.path.join(figures_path, "model_comparison.png") if not figures_path.startswith(
        "s3://") else f"{figures_path}/model_comparison.png"
    visualize_comparison(
        evaluation_results,
        save_path=figure_path
    )

    # 保存评估结果
    eval_results_path = os.path.join(results_path, "evaluation_results.json") if not results_path.startswith(
        "s3://") else f"{results_path}/evaluation_results.json"
    save_evaluation_results(
        evaluation_results,
        eval_results_path
    )

    # 保存模型和相关数据
    print("保存模型...")

    # 保存ALS模型
    als_model_path = os.path.join(models_path, "als_model") if not models_path.startswith(
        "s3://") else f"{models_path}/als_model"

    try:
        # 本地保存或创建临时路径
        if not models_path.startswith("s3://"):
            # 本地保存
            als_model.save(als_model_path)
            print("ALS模型已保存到本地")
        else:
            # S3路径 - 需要先临时保存然后上传
            temp_model_dir = "temp_als_model"
            os.makedirs(temp_model_dir, exist_ok=True)

            # 本地临时保存
            als_model.save(temp_model_dir)
            print("ALS模型已临时保存到本地")

            # 解析S3路径
            s3_parts = models_path.replace("s3://", "").split("/")
            bucket_name = s3_parts[0]
            prefix = "/".join(s3_parts[1:]) if len(s3_parts) > 1 else ""

            # 上传到S3
            s3_client = boto3.client('s3')
            for root, dirs, files in os.walk(temp_model_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, temp_model_dir)
                    s3_key = f"{prefix}/als_model/{relative_path}" if prefix else f"als_model/{relative_path}"
                    s3_client.upload_file(local_path, bucket_name, s3_key)

            print(f"ALS模型已上传到S3: {models_path}/als_model")

            # 清理临时目录
            import shutil
            shutil.rmtree(temp_model_dir)

    except Exception as e:
        print(f"保存ALS模型时出错: {e}")

    # 示例：为一些用户生成推荐
    print("\n推荐示例:")

    # 选择一个测试用户
    if len(test_users) > 0:
        example_user = test_users[0]
        print(f"\n为用户 {example_user} 生成推荐:")

        # 使用混合模型生成推荐
        hybrid_recommendations = hybrid_models["Hybrid_0.7"](example_user, 5)
        print("混合推荐结果:")
        print(hybrid_recommendations[['app_id', 'title']].to_string(index=False))

        # 使用协同过滤生成推荐
        cf_recommendations = pure_cf_model(example_user, 5)
        print("\n纯协同过滤推荐结果:")
        print(cf_recommendations[['app_id', 'title']].to_string(index=False))

        # 使用内容推荐生成推荐
        content_recommendations = pure_content_model(example_user, 5)
        print("\n纯内容推荐结果:")
        print(content_recommendations[['app_id', 'title']].to_string(index=False))

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

    # 最终检查点
    save_checkpoint(None, "training_complete", {
        "total_runtime_seconds": run_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    print("\n完成！推荐系统模型已训练并评估。")


if __name__ == "__main__":
    main()