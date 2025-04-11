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
import threading
import requests
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime, timedelta
from src.online_learning import METRICS_FILE

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

if 'online_learning_enabled' not in st.session_state:
    st.session_state.online_learning_enabled = False
if 'online_learning_status' not in st.session_state:
    st.session_state.online_learning_status = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = []

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

# 在线学习控制
st.sidebar.header("在线学习")
online_learning_enabled = st.sidebar.checkbox("启用在线学习", value=st.session_state.online_learning_enabled)

if online_learning_enabled != st.session_state.online_learning_enabled:
    st.session_state.online_learning_enabled = online_learning_enabled

    try:
        # 通知在线学习服务切换状态
        response = requests.post(
            "http://localhost:5000/api/toggle_learning",
            json={"enabled": online_learning_enabled},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            st.sidebar.success(f"在线学习已{'启用' if result['enabled'] else '禁用'}")
            st.session_state.online_learning_status = "running" if result['enabled'] else "stopped"
        else:
            st.sidebar.error(f"切换在线学习失败: {response.text}")
    except Exception as e:
        st.sidebar.error(f"连接在线学习服务失败: {e}")
        st.session_state.online_learning_status = "error"

# 在线学习状态
if st.session_state.online_learning_enabled:
    try:
        # 获取在线学习服务状态
        response = requests.get("http://localhost:5000/api/status", timeout=5)

        if response.status_code == 200:
            status = response.json()

            # 显示状态信息
            buffer_size = status.get('buffer_size', {})
            total_buffer = buffer_size.get('total', 0)

            st.sidebar.metric("缓冲区大小", total_buffer)
            st.sidebar.text(f"上次更新: {status.get('last_update_time', '无')}")
            st.sidebar.text(f"更新次数: {status.get('update_count', 0)}")

            if status.get('update_thread_running', False):
                st.sidebar.success("在线学习正在运行")
            else:
                st.sidebar.warning("在线学习已启用但未运行")
        else:
            st.sidebar.error("获取在线学习状态失败")

    except Exception as e:
        st.sidebar.error(f"连接在线学习服务失败: {e}")

# 添加新的选项卡
tab1, tab2, tab3, tab4 = st.tabs(["用户推荐", "游戏相似度", "推荐系统评估", "在线学习分析"])

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

# 在线学习分析选项卡
with tab4:
    st.header("在线学习性能分析")

    if not st.session_state.online_learning_enabled:
        st.warning("请先启用在线学习功能")
    else:
        # 添加刷新按钮
        if st.button("刷新在线学习指标"):
            try:
                # 获取指标数据
                response = requests.get("http://localhost:5000/api/metrics", timeout=5)

                if response.status_code == 200:
                    metrics_data = response.json()
                    st.session_state.metrics_data = metrics_data
                    st.success(f"已加载 {len(metrics_data)} 条指标数据")
                else:
                    st.error(f"获取指标失败: {response.text}")
            except Exception as e:
                st.error(f"连接在线学习服务失败: {e}")

        # 如果有指标数据，则显示
        if st.session_state.metrics_data:
            metrics = st.session_state.metrics_data

            # 创建性能指标表格
            st.subheader("最近更新指标")

            # 提取最近5次更新的指标
            recent_metrics = metrics[-5:] if len(metrics) > 5 else metrics
            recent_metrics.reverse()  # 最新的在前面

            # 转换为DataFrame以便显示
            metrics_df = pd.DataFrame(recent_metrics)

            if 'timestamp' in metrics_df.columns:
                metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
                metrics_df['timestamp'] = metrics_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # 选择要显示的列
            display_cols = ['update_id', 'timestamp', 'rmse', 'processing_time', 'training_time',
                            'num_samples', 'num_new_games', 'update_type']
            display_cols = [col for col in display_cols if col in metrics_df.columns]

            st.dataframe(metrics_df[display_cols], use_container_width=True)

            # 绘制RMSE随时间变化图
            st.subheader("模型性能随时间变化")

            # 创建绘图数据
            plot_metrics = pd.DataFrame(metrics)
            if 'timestamp' in plot_metrics.columns:
                plot_metrics['timestamp'] = pd.to_datetime(plot_metrics['timestamp'])
                plot_metrics = plot_metrics.sort_values('timestamp')

            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制RMSE图
            if 'rmse' in plot_metrics.columns:
                ax.plot(range(len(plot_metrics)), plot_metrics['rmse'], 'b-', label='RMSE')
                ax.set_ylabel('RMSE', color='b')
                ax.tick_params(axis='y', labelcolor='b')

                # 设置x轴标签
                if len(plot_metrics) > 10:
                    step = len(plot_metrics) // 10
                    ax.set_xticks(range(0, len(plot_metrics), step))
                    if 'timestamp' in plot_metrics.columns:
                        ax.set_xticklabels(plot_metrics['timestamp'].iloc[::step].dt.strftime('%H:%M'), rotation=45)
                    else:
                        ax.set_xticklabels(
                            plot_metrics['update_id'].iloc[::step] if 'update_id' in plot_metrics.columns else range(1,
                                                                                                                     len(plot_metrics) + 1,
                                                                                                                     step))

                # 添加训练时间图
                if 'training_time' in plot_metrics.columns or 'processing_time' in plot_metrics.columns:
                    ax2 = ax.twinx()

                    if 'training_time' in plot_metrics.columns:
                        ax2.plot(range(len(plot_metrics)), plot_metrics['training_time'], 'r--', label='训练时间')

                    if 'processing_time' in plot_metrics.columns:
                        ax2.plot(range(len(plot_metrics)), plot_metrics['processing_time'], 'g-.', label='处理时间')

                    ax2.set_ylabel('时间 (秒)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')

                # 添加图例
                lines1, labels1 = ax.get_legend_handles_labels()
                if 'training_time' in plot_metrics.columns or 'processing_time' in plot_metrics.columns:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                else:
                    ax.legend(loc='upper right')

                ax.set_title('模型性能随时间变化')
                ax.grid(True, linestyle='--', alpha=0.7)
                fig.tight_layout()

                st.pyplot(fig)

            # 显示样本数和新游戏数
            st.subheader("数据增长情况")

            fig2, ax = plt.subplots(figsize=(10, 6))

            if 'num_samples' in plot_metrics.columns:
                ax.bar(range(len(plot_metrics)), plot_metrics['num_samples'], alpha=0.7, label='新样本数')

            if 'num_new_games' in plot_metrics.columns:
                ax.bar(range(len(plot_metrics)), plot_metrics['num_new_games'], alpha=0.5, label='新游戏数')

            ax.set_ylabel('数量')

            # 设置x轴标签
            if len(plot_metrics) > 10:
                step = len(plot_metrics) // 10
                ax.set_xticks(range(0, len(plot_metrics), step))
                if 'timestamp' in plot_metrics.columns:
                    ax.set_xticklabels(plot_metrics['timestamp'].iloc[::step].dt.strftime('%H:%M'), rotation=45)
                else:
                    ax.set_xticklabels(
                        plot_metrics['update_id'].iloc[::step] if 'update_id' in plot_metrics.columns else range(1,
                                                                                                                 len(plot_metrics) + 1,
                                                                                                                 step))

            ax.legend()
            ax.set_title('数据增长情况')
            ax.grid(True, linestyle='--', alpha=0.7)
            fig2.tight_layout()

            st.pyplot(fig2)

        else:
            st.info("暂无在线学习指标数据，请点击刷新按钮获取")


# 启动在线学习服务的函数
def start_online_learning_service():
    """启动在线学习服务"""
    import subprocess
    import sys

    try:
        # 启动在线学习服务
        cmd = [sys.executable, "-m", "src.online_learning"]
        process = subprocess.Popen(cmd)

        # 等待服务启动
        time.sleep(5)

        return process
    except Exception as e:
        st.error(f"启动在线学习服务失败: {e}")
        return None


# 关闭Spark会话
if st.sidebar.button("关闭Spark"):
    if st.session_state.spark is not None:
        st.session_state.spark.stop()
        st.session_state.spark = None
        st.success("Spark会话已关闭！")