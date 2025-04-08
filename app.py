# app.py
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from pyspark.sql import SparkSession

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_processing import initialize_spark, load_data, preprocess_data
from src.content_based import build_tfidf_model, get_content_recommendations
from src.hybrid_model import build_hybrid_recommender
from src.cold_start import build_popularity_model

# 页面配置
st.set_page_config(
    page_title="Steam游戏推荐系统",
    page_icon="🎮",
    layout="wide"
)

# 初始化会话状态
if 'spark' not in st.session_state:
    st.session_state.spark = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_built' not in st.session_state:
    st.session_state.models_built = False

# 标题和介绍
st.title("Steam游戏推荐系统")
st.markdown("""
这是一个基于协同过滤和内容过滤的混合推荐系统，可以为用户推荐Steam游戏。
""")

# 侧边栏
st.sidebar.title("控制面板")

# 初始化Spark（如果尚未初始化）
if st.sidebar.button("初始化Spark") or st.session_state.spark is None:
    with st.spinner("正在初始化Spark..."):
        st.session_state.spark = initialize_spark()
    st.success("Spark已初始化！")

# 加载数据
if st.sidebar.button("加载数据") or (st.session_state.spark is not None and not st.session_state.data_loaded):
    with st.spinner("正在加载数据..."):
        data_path = "data"
        games_df, users_df, recommendations_df, metadata_df = load_data(data_path)
        games_with_metadata, spark_ratings, processed_recommendations = preprocess_data(
            games_df, users_df, recommendations_df, metadata_df, st.session_state.spark
        )

        # 保存到会话状态
        st.session_state.games_df = games_df
        st.session_state.users_df = users_df
        st.session_state.recommendations_df = recommendations_df
        st.session_state.metadata_df = metadata_df
        st.session_state.games_with_metadata = games_with_metadata
        st.session_state.spark_ratings = spark_ratings
        st.session_state.processed_recommendations = processed_recommendations
        st.session_state.data_loaded = True

    st.success("数据加载完成！")

# 构建模型
if st.sidebar.button("构建模型") and st.session_state.data_loaded:
    with st.spinner("正在构建推荐模型..."):
        # 构建TF-IDF模型
        tfidf, cosine_sim, indices, content_df = build_tfidf_model(st.session_state.games_with_metadata)

        # 从训练好的模型中加载ALS模型
        try:
            from pyspark.ml.recommendation import ALSModel

            als_model = ALSModel.load("models/als_model")
            st.success("成功加载已训练的ALS模型！")
        except Exception as e:
            st.error(f"加载ALS模型失败: {e}")
            st.info("您需要先运行main.py训练模型")
            als_model = None

        # 构建混合推荐模型
        if als_model is not None:
            hybrid_recommender = build_hybrid_recommender(
                als_model, cosine_sim, indices, st.session_state.games_with_metadata,
                st.session_state.processed_recommendations, 0.7, st.session_state.spark
            )

            # 构建流行度推荐模型
            popularity_recommender = build_popularity_model(
                st.session_state.processed_recommendations, st.session_state.games_with_metadata
            )

            # 保存到会话状态
            st.session_state.tfidf = tfidf
            st.session_state.cosine_sim = cosine_sim
            st.session_state.indices = indices
            st.session_state.als_model = als_model
            st.session_state.hybrid_recommender = hybrid_recommender
            st.session_state.popularity_recommender = popularity_recommender
            st.session_state.models_built = True

    if st.session_state.models_built:
        st.success("模型构建完成！")

# 创建选项卡
tab1, tab2, tab3 = st.tabs(["用户推荐", "游戏相似度", "推荐系统评估"])

# 用户推荐选项卡
with tab1:
    st.header("游戏推荐")

    if not st.session_state.models_built:
        st.warning("请先构建模型")
    else:
        # 用户选择
        if 'test_users' not in st.session_state:
            st.session_state.test_users = sorted(st.session_state.processed_recommendations['user_id'].unique()[:100])

        user_id = st.selectbox(
            "选择用户ID",
            options=st.session_state.test_users,
            format_func=lambda x: f"用户 {x}"
        )

        # 推荐类型选择
        rec_type = st.radio(
            "推荐类型",
            options=["混合推荐", "协同过滤推荐", "流行度推荐"],
            horizontal=True
        )

        if st.button("生成推荐"):
            with st.spinner("正在生成推荐..."):
                if rec_type == "混合推荐":
                    recommendations = st.session_state.hybrid_recommender(user_id, 10)
                    title = f"为用户 {user_id} 的混合推荐"
                elif rec_type == "协同过滤推荐":
                    # 使用ALS模型直接推荐
                    from src.collaborative_filtering import get_als_recommendations

                    als_recs = get_als_recommendations(
                        st.session_state.als_model, user_id, 10, st.session_state.spark
                    )
                    recommendations = st.session_state.games_with_metadata[
                        st.session_state.games_with_metadata['app_id'].isin(als_recs['app_id'])
                    ].copy()
                    recommendations['recommendation_score'] = recommendations['app_id'].map(
                        dict(zip(als_recs['app_id'], als_recs['rating']))
                    )
                    title = f"为用户 {user_id} 的协同过滤推荐"
                else:
                    recommendations = st.session_state.popularity_recommender(10)
                    title = "热门游戏推荐"

                # 显示推荐结果
                st.subheader(title)

                # 获取用户历史
                user_history = st.session_state.processed_recommendations[
                    st.session_state.processed_recommendations['user_id'] == user_id
                    ]

                if len(user_history) > 0:
                    st.write(f"用户历史游戏: {len(user_history)} 款")
                    history_games = st.session_state.games_df[
                        st.session_state.games_df['app_id'].isin(user_history['app_id'])
                    ]
                    st.dataframe(
                        history_games[['app_id', 'title']].head(5),
                        use_container_width=True
                    )

                # 显示推荐结果
                if len(recommendations) > 0:
                    display_cols = ['app_id', 'title']

                    if 'tags' in recommendations.columns:
                        # 将标签列表转换为字符串
                        recommendations['tags_str'] = recommendations['tags'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                        )
                        display_cols.append('tags_str')

                    if 'recommendation_score' in recommendations.columns:
                        display_cols.append('recommendation_score')

                    st.dataframe(
                        recommendations[display_cols].reset_index(drop=True),
                        use_container_width=True
                    )
                else:
                    st.info("没有推荐结果")

# 游戏相似度选项卡
with tab2:
    st.header("游戏相似度查询")

    if not st.session_state.models_built:
        st.warning("请先构建模型")
    else:
        # 游戏选择
        game_options = list(zip(
            st.session_state.games_df['app_id'],
            st.session_state.games_df['title']
        ))
        selected_game = st.selectbox(
            "选择游戏",
            options=game_options,
            format_func=lambda x: f"{x[1]} (ID: {x[0]})"
        )

        if st.button("查找相似游戏"):
            with st.spinner("正在查找相似游戏..."):
                app_id = selected_game[0]
                game_title = selected_game[1]

                # 获取内容相似的游戏
                similar_games = get_content_recommendations(
                    app_id, st.session_state.cosine_sim, st.session_state.indices,
                    st.session_state.games_with_metadata, 10
                )

                # 显示结果
                st.subheader(f"与《{game_title}》相似的游戏")

                if len(similar_games) > 0:
                    # 添加标签列
                    if 'tags' in st.session_state.games_with_metadata.columns:
                        similar_games = pd.merge(
                            similar_games,
                            st.session_state.games_with_metadata[['app_id', 'tags']],
                            on='app_id'
                        )

                        # 将标签列表转换为字符串
                        similar_games['tags_str'] = similar_games['tags'].apply(
                            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                        )

                    st.dataframe(
                        similar_games[['app_id', 'title', 'tags_str', 'similarity_score']].reset_index(drop=True),
                        use_container_width=True
                    )
                else:
                    st.info("没有找到相似游戏")

# 推荐系统评估选项卡
with tab3:
    st.header("推荐系统评估")

    # 加载评估结果
    eval_path = "results/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)

        # 显示评估结果
        st.subheader("各模型性能比较")

        # 转换为DataFrame以便于显示
        metrics = ['precision', 'recall', 'f1', 'ndcg', 'diversity']
        eval_df = pd.DataFrame({
            'Model': list(eval_results.keys()),
            **{metric.upper(): [results[metric] for results in eval_results.values()] for metric in metrics}
        })

        st.dataframe(eval_df, use_container_width=True)

        # 可视化
        st.subheader("性能指标可视化")
        fig, ax = plt.subplots(figsize=(10, 6))

        # 准备数据
        models = list(eval_results.keys())
        metrics_values = {
            metric.upper(): [results[metric] for results in eval_results.values()]
            for metric in metrics
        }

        # 创建条形图
        x = np.arange(len(models))
        width = 0.15
        multiplier = 0

        for metric, values in metrics_values.items():
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=metric)
            multiplier += 1

        # 添加标签和图例
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('分数')
        ax.set_title('推荐模型性能比较')
        ax.legend(loc='upper left', ncols=5)

        plt.tight_layout()
        st.pyplot(fig)

        # 显示评估图像
        img_path = "results/figures/model_comparison.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="模型性能比较详细图表")
    else:
        st.info("未找到评估结果。请先运行main.py生成评估结果。")

# 关闭Spark会话
if st.sidebar.button("关闭Spark"):
    if st.session_state.spark is not None:
        st.session_state.spark.stop()
        st.session_state.spark = None
        st.success("Spark会话已关闭！")