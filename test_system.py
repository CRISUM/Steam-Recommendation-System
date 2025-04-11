# test_system.py
import time
from src import load_data, preprocess_data
from src import build_tfidf_model, get_content_recommendations


def test_system():
    """测试推荐系统的基本功能（不包括协同过滤和混合推荐）"""
    start_time = time.time()

    # 使用小型数据集
    data_path = "data_small"

    print(f"从 {data_path} 加载数据...")
    try:
        # 加载数据
        games_df, users_df, recommendations_df, metadata_df = load_data(data_path)

        print("\n基本数据统计:")
        print(f"游戏数量: {len(games_df)}")
        print(f"用户数量: {len(users_df)}")
        print(f"评价数量: {len(recommendations_df)}")
        print(f"元数据数量: {len(metadata_df)}")

        # 数据预处理 - 不使用Spark
        print("\n进行数据预处理...")
        games_with_metadata, _, processed_recommendations = preprocess_data(
            games_df, users_df, recommendations_df, metadata_df, spark=None
        )

        # 构建内容推荐模型
        print("\n构建TF-IDF模型...")
        tfidf, cosine_sim, indices, content_df = build_tfidf_model(games_with_metadata)

        # 测试游戏相似度推荐
        if len(games_df) > 0:
            # 选择一个游戏示例
            example_game_id = games_df['app_id'].iloc[0]
            game_title = games_df.loc[games_df['app_id'] == example_game_id, 'title'].values[0]

            print(f"\n为游戏 '{game_title}' (ID: {example_game_id}) 推荐相似游戏:")
            similar_games = get_content_recommendations(
                example_game_id, cosine_sim, indices, games_with_metadata, 5
            )

            if not similar_games.empty:
                print(similar_games[['app_id', 'title', 'similarity_score']].to_string(index=False))
            else:
                print("没有找到相似游戏")

        end_time = time.time()
        print(f"\n测试完成! 运行时间: {end_time - start_time:.2f} 秒")

    except Exception as e:
        print(f"测试时出错: {e}")


if __name__ == "__main__":
    test_system()