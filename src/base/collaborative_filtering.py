# src/collaborative_filtering.py
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd


def build_als_model(ratings_df, rank=10, reg_param=0.1, alpha=1.0, max_iter=10):
    """构建ALS协同过滤模型"""
    # 构建ALS模型
    als = ALS(
        maxIter=max_iter,
        rank=rank,  # 潜在因子数量
        regParam=reg_param,  # 正则化参数
        alpha=alpha,  # 控制隐式反馈置信度
        userCol="user_id",
        itemCol="app_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=True  # 将评分视为隐式反馈
    )

    # 训练模型
    model = als.fit(ratings_df)
    return model


def evaluate_als_model(model, test_data):
    """评估ALS模型性能"""
    # 使用模型进行预测
    predictions = model.transform(test_data)

    # RMSE评估
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)

    # MAE评估
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)

    return {"RMSE": rmse, "MAE": mae}


def tune_als_parameters(train_data, validation_data=None, test_ratio=0.2):
    """ALS模型参数调优"""
    if validation_data is None:
        # 如果没有提供验证集，从训练集分割出一部分
        (train_subset, validation_data) = train_data.randomSplit([1.0 - test_ratio, test_ratio], seed=42)
    else:
        train_subset = train_data

    # 需要调优的参数
    ranks = [10, 20, 50]
    reg_params = [0.01, 0.1, 1.0]
    alphas = [0.5, 1.0, 2.0]

    # 存储结果
    results = []

    # 评估器
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    # 网格搜索
    best_rmse = float("inf")
    best_params = None

    for rank in ranks:
        for reg_param in reg_params:
            for alpha in alphas:
                als = ALS(
                    maxIter=10,
                    rank=rank,
                    regParam=reg_param,
                    alpha=alpha,
                    userCol="user_id",
                    itemCol="app_id",
                    ratingCol="rating",
                    coldStartStrategy="drop",
                    implicitPrefs=True
                )

                model = als.fit(train_subset)
                predictions = model.transform(validation_data)

                # 有些预测可能是NaN，需要过滤掉
                predictions = predictions.filter("prediction IS NOT NULL")

                if predictions.count() > 0:
                    rmse = evaluator.evaluate(predictions)
                else:
                    rmse = float("inf")

                print(f"Rank: {rank}, RegParam: {reg_param}, Alpha: {alpha}, RMSE: {rmse}")

                results.append({
                    'rank': rank,
                    'regParam': reg_param,
                    'alpha': alpha,
                    'rmse': rmse
                })

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {
                        'rank': rank,
                        'regParam': reg_param,
                        'alpha': alpha,
                    }

    # 找出最佳参数
    print(f"最佳参数: {best_params}, RMSE: {best_rmse}")

    return best_params, results


def get_als_recommendations(model, user_id, num_items=10, spark=None):
    """为指定用户获取ALS模型推荐"""
    if spark is None:
        raise ValueError("必须提供Spark会话才能生成推荐")

    # 创建用户DataFrame
    user_df = spark.createDataFrame([(user_id,)], ["user_id"])

    # 获取推荐
    recommendations = model.recommendForUserSubset(user_df, num_items)

    if recommendations.count() > 0:
        # 提取推荐结果
        rec_df = recommendations.collect()[0]
        items = [{"app_id": rec.app_id, "rating": rec.rating} for rec in rec_df.recommendations]
        return pd.DataFrame(items)
    else:
        return pd.DataFrame(columns=["app_id", "rating"])