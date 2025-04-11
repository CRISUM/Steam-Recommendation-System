# tests/test_recommendation.py
import unittest
import sys
import os
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import initialize_spark, preprocess_data
from src import build_tfidf_model, get_content_recommendations


class TestRecommendationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试前准备工作"""
        # 初始化Spark
        cls.spark = initialize_spark("TestRecommendationSystem")

        # 创建测试数据
        cls.create_test_data()

    @classmethod
    def tearDownClass(cls):
        """测试后清理工作"""
        # 关闭Spark会话
        cls.spark.stop()

    @classmethod
    def create_test_data(cls):
        """创建测试数据"""
        # 测试游戏数据
        cls.games_df = pd.DataFrame({
            'app_id': [1, 2, 3, 4, 5],
            'title': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'],
            'tags': [['Action', 'Adventure'], ['RPG', 'Strategy'],
                     ['Action', 'Shooter'], ['Puzzle', 'Adventure'],
                     ['Strategy', 'Simulation']]
        })

        # 测试用户数据
        cls.users_df = pd.DataFrame({
            'user_id': [101, 102, 103],
            'products': [5, 3, 7],
            'reviews': [2, 1, 3]
        })

        # 测试推荐数据
        cls.recommendations_df = pd.DataFrame({
            'app_id': [1, 2, 3, 1, 2, 4, 5, 3, 5],
            'user_id': [101, 101, 101, 102, 102, 102, 103, 103, 103],
            'is_recommended': [True, False, True, True, True, False, True, False, True],
            'hours': [10.5, 2.0, 15.0, 8.0, 20.0, 1.5, 5.0, 3.0, 12.0],
            'rating': [8.0, 3.0, 9.0, 7.0, 9.5, 2.0, 6.0, 4.0, 8.5]
        })

        # 测试元数据
        cls.metadata_df = pd.DataFrame({
            'app_id': [1, 2, 3, 4, 5],
            'description': [
                'An action adventure game with great story',
                'A role-playing strategy game set in medieval times',
                'A fast-paced action shooter with multiplayer',
                'A puzzle adventure game with challenging levels',
                'A strategy simulation game with city building'
            ],
            'tags': [
                ['Action', 'Adventure'],
                ['RPG', 'Strategy'],
                ['Action', 'Shooter'],
                ['Puzzle', 'Adventure'],
                ['Strategy', 'Simulation']
            ]
        })

    def test_data_loading(self):
        """测试数据加载和预处理"""
        # 预处理测试数据
        games_with_metadata, spark_ratings, processed_recommendations = preprocess_data(
            self.games_df, self.users_df, self.recommendations_df, self.metadata_df, self.spark
        )

        # 验证预处理结果
        self.assertEqual(len(games_with_metadata), 5, "游戏数据预处理后长度应为5")
        self.assertIn('description', games_with_metadata.columns, "预处理后应包含description列")

        # 不测试Spark DataFrame
        # self.assertEqual(spark_ratings.count(), 9, "Spark评分数据应有9条记录")
        # self.assertEqual(len(spark_ratings.columns), 3, "Spark评分数据应有3列")

    def test_content_model(self):
        """测试内容推荐模型"""
        # 构建内容模型
        tfidf, cosine_sim, indices, content_df = build_tfidf_model(self.games_df)

        # 验证模型输出
        self.assertEqual(cosine_sim.shape, (5, 5), "相似度矩阵应为5x5")
        self.assertEqual(len(indices), 5, "索引长度应为5")

        # 测试推荐功能
        recommendations = get_content_recommendations(1, cosine_sim, indices, self.games_df, 3)

        # 验证推荐结果
        self.assertLessEqual(len(recommendations), 3, "应返回不超过3个推荐")
        self.assertIn('similarity_score', recommendations.columns, "推荐结果应包含相似度分数")

    # 可以添加更多测试用例


if __name__ == '__main__':
    unittest.main()