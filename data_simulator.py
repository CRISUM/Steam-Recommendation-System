# data_simulator.py
import os
import pandas as pd
import numpy as np
import json
import time
import random
import argparse
import requests
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("data_simulator")


class SteamDataSimulator:
    """模拟Steam游戏推荐系统的数据流"""

    def __init__(self, data_path="data", api_url="http://localhost:5000/api/data"):
        """初始化模拟器"""
        self.data_path = data_path
        self.api_url = api_url
        self.games_df = None
        self.users_df = None
        self.tags_by_game = {}
        self.game_titles = {}
        self.user_profiles = {}
        self.sent_data_count = 0
        self.start_time = datetime.now()

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载游戏和用户数据"""
        logger.info(f"从 {self.data_path} 加载数据...")
        self.sent_data_count = 0
        self.start_time = datetime.now()

        try:
            # 加载游戏数据
            games_file = os.path.join(self.data_path, "games.csv")
            if os.path.exists(games_file):
                self.games_df = pd.read_csv(games_file)
                logger.info(f"已加载 {len(self.games_df)} 个游戏")

                # 创建游戏标题查找字典
                self.game_titles = dict(zip(self.games_df['app_id'], self.games_df['title']))
            else:
                logger.error(f"找不到游戏数据: {games_file}")
                raise FileNotFoundError(f"找不到游戏数据: {games_file}")

            # 加载用户数据
            users_file = os.path.join(self.data_path, "users.csv")
            if os.path.exists(users_file):
                self.users_df = pd.read_csv(users_file)
                logger.info(f"已加载 {len(self.users_df)} 个用户")
            else:
                logger.error(f"找不到用户数据: {users_file}")
                raise FileNotFoundError(f"找不到用户数据: {users_file}")

            # 加载游戏元数据以获取标签
            metadata_file = os.path.join(self.data_path, "games_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    # 从JSON文件读取元数据
                    game_tags = {}
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if 'app_id' in data and 'tags' in data:
                                    game_tags[data['app_id']] = data['tags']
                            except json.JSONDecodeError:
                                continue

                    self.tags_by_game = game_tags
                    logger.info(f"已加载 {len(game_tags)} 个游戏的标签信息")
                except Exception as e:
                    logger.error(f"加载游戏元数据出错: {e}")
            else:
                logger.warning(f"找不到游戏元数据: {metadata_file}")
                # 创建一些模拟的标签
                all_tags = ["Action", "Adventure", "RPG", "Strategy", "Simulation",
                            "Casual", "Indie", "Sports", "Racing", "MMO", "FPS",
                            "Puzzle", "Horror", "Open World", "Story Rich"]

                for game_id in self.games_df['app_id']:
                    num_tags = random.randint(2, 5)
                    self.tags_by_game[game_id] = random.sample(all_tags, num_tags)

                logger.info(f"已生成 {len(self.tags_by_game)} 个游戏的模拟标签")

            # 初始化用户游戏偏好模拟
            self._initialize_user_profiles()

        except Exception as e:
            logger.error(f"加载游戏元数据出错: {e}")

    def _initialize_user_profiles(self):
        """为用户创建模拟偏好档案"""
        for user_id in self.users_df['user_id']:
            # 随机选择用户喜欢的标签
            all_tags = list(set([tag for tags in self.tags_by_game.values() for tag in tags]))
            num_preferred_tags = random.randint(2, 5)
            preferred_tags = random.sample(all_tags, min(num_preferred_tags, len(all_tags)))

            # 用户偏好强度 (0.5-1.0)
            tag_weights = {tag: random.uniform(0.5, 1.0) for tag in preferred_tags}

            # 游戏偏好（偏好概率高的游戏）
            preferred_games = []
            for game_id, tags in self.tags_by_game.items():
                overlap = set(tags) & set(preferred_tags)
                if overlap:
                    weight = sum([tag_weights.get(tag, 0) for tag in overlap]) / len(overlap)
                    preferred_games.append((game_id, weight))

            preferred_games.sort(key=lambda x: x[1], reverse=True)
            preferred_games = preferred_games[:min(20, len(preferred_games))]

            self.user_profiles[user_id] = {
                'preferred_tags': preferred_tags,
                'tag_weights': tag_weights,
                'preferred_games': dict(preferred_games),
                'played_games': set()  # 记录已经模拟过的游戏
            }

        logger.info(f"已为 {len(self.user_profiles)} 个用户创建模拟偏好档案")

    def generate_recommendation_batch(self, batch_size=10):
        """生成一批模拟的用户游戏评价数据"""
        recommendations = []

        # 决定是否添加新用户和新游戏
        add_new_user = random.random() < 0.05  # 5%概率添加新用户
        add_new_game = random.random() < 0.02  # 2%概率添加新游戏

        new_users = []
        new_games = []

        # 可能添加一个新用户
        if add_new_user:
            new_user_id = max(self.users_df['user_id']) + random.randint(1, 100)
            new_user = {
                'user_id': int(new_user_id),
                'products': random.randint(1, 20),
                'reviews': random.randint(0, 5)
            }
            new_users.append(new_user)

            # 为新用户创建偏好档案
            all_tags = list(set([tag for tags in self.tags_by_game.values() for tag in tags]))
            preferred_tags = random.sample(all_tags, min(3, len(all_tags)))
            tag_weights = {tag: random.uniform(0.5, 1.0) for tag in preferred_tags}

            self.user_profiles[new_user_id] = {
                'preferred_tags': preferred_tags,
                'tag_weights': tag_weights,
                'preferred_games': {},
                'played_games': set()
            }

            logger.info(f"创建新用户 ID: {new_user_id}")

        # 可能添加一个新游戏
        if add_new_game:
            new_game_id = max(self.games_df['app_id']) + random.randint(1, 100)

            # 随机选择标签
            all_tags = list(set([tag for tags in self.tags_by_game.values() for tag in tags]))
            game_tags = random.sample(all_tags, random.randint(2, 5))

            # 创建新游戏
            new_game = {
                'app_id': int(new_game_id),
                'title': f"New Game {new_game_id}",
                'date_release': datetime.now().strftime('%Y-%m-%d'),
                'win': random.choice([0, 1]),
                'mac': random.choice([0, 1]),
                'linux': random.choice([0, 1]),
                'rating': random.randint(1, 10),
                'positive_ratio': random.randint(0, 100),
                'user_reviews': random.randint(10, 1000),
                'price_final': random.uniform(0, 60),
                'price_original': random.uniform(0, 60),
                'discount': random.randint(0, 75),
                'steam_deck': random.choice([0, 1])
            }

            # 创建新游戏的元数据
            new_metadata = {
                'app_id': int(new_game_id),
                'title': f"New Game {new_game_id}",
                'description': f"This is a new game with tags: {', '.join(game_tags)}",
                'tags': game_tags
            }

            new_games.append(new_game)
            self.tags_by_game[new_game_id] = game_tags
            self.game_titles[new_game_id] = f"New Game {new_game_id}"

            logger.info(f"创建新游戏 ID: {new_game_id}, 标签: {game_tags}")

        # 生成用户评价数据
        for _ in range(batch_size):
            # 随机选择一个用户
            user_id = random.choice(list(self.user_profiles.keys()))
            user_profile = self.user_profiles[user_id]

            # 基于用户偏好选择游戏
            if random.random() < 0.7 and user_profile['preferred_games']:  # 70%选择偏好游戏
                available_games = [(game_id, weight) for game_id, weight in user_profile['preferred_games'].items()
                                   if game_id not in user_profile['played_games']]

                if not available_games:  # 如果没有未玩过的偏好游戏，选择任意游戏
                    all_game_ids = [gid for gid in self.tags_by_game.keys()
                                    if gid not in user_profile['played_games']]
                    if not all_game_ids:  # 如果所有游戏都玩过，允许重复
                        all_game_ids = list(self.tags_by_game.keys())

                    game_id = random.choice(all_game_ids)
                    weight = 0.5  # 默认权重
                else:
                    # 按权重抽样选择游戏
                    weights = [w for _, w in available_games]
                    weights_sum = sum(weights)
                    normalized_weights = [w / weights_sum for w in weights] if weights_sum > 0 else None
                    game_index = np.random.choice(len(available_games), p=normalized_weights)
                    game_id, weight = available_games[game_index]
            else:
                # 随机选择任意游戏
                all_game_ids = list(self.tags_by_game.keys())
                game_id = random.choice(all_game_ids)

                # 计算基于标签的偏好度
                game_tags = self.tags_by_game.get(game_id, [])
                overlap = set(game_tags) & set(user_profile['preferred_tags'])

                if overlap:
                    weight = sum([user_profile['tag_weights'].get(tag, 0) for tag in overlap]) / len(overlap)
                else:
                    weight = random.uniform(0.1, 0.5)  # 非偏好游戏的低权重

            # 记录已玩游戏
            user_profile['played_games'].add(game_id)

            # 根据偏好程度确定评价倾向
            if random.random() < weight:  # 偏好度越高，越可能推荐
                is_recommended = True
                hours_played = random.uniform(5, 100)  # 喜欢的游戏玩的时间更长
            else:
                is_recommended = False
                hours_played = random.uniform(0.1, 10)  # 不喜欢的游戏玩的时间短

            # 创建推荐记录
            recommendation = {
                'app_id': int(game_id),
                'user_id': int(user_id),
                'is_recommended': is_recommended,
                'hours': float(hours_played),
                'timestamp': datetime.now().isoformat()
            }

            recommendations.append(recommendation)

        return {
            'recommendations': recommendations,
            'users': new_users,
            'games': new_games,
            'metadata': [new_metadata] if new_games else []
        }

    def send_batch(self, batch_data):
        """发送数据批次到API"""
        try:
            response = requests.post(self.api_url, json=batch_data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"数据发送成功: {result.get('message')}, 缓冲区大小: {result.get('buffer_size')}")
                return True
            else:
                logger.error(f"发送数据失败: HTTP {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"发送数据时发生错误: {e}")
            return False

    def run_simulation(self, num_batches=10, batch_size=10, interval_seconds=5):
        """运行模拟数据流发送"""
        logger.info(f"开始模拟数据流，批次数: {num_batches}, 批次大小: {batch_size}, 间隔: {interval_seconds}秒")

        for i in range(num_batches):
            try:
                logger.info(f"生成批次 {i + 1}/{num_batches}")
                batch_data = self.generate_recommendation_batch(batch_size)

                recommendations_count = len(batch_data.get('recommendations', []))
                games_count = len(batch_data.get('games', []))
                users_count = len(batch_data.get('users', []))

                logger.info(f"批次 {i + 1} 包含: {recommendations_count} 条评价, "
                            f"{games_count} 个新游戏, {users_count} 个新用户")

                if self.send_batch(batch_data):
                    self.sent_data_count += recommendations_count

                # 等待指定的间隔时间
                if i < num_batches - 1:  # 最后一批不需要等待
                    time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("模拟被用户中断")
                break

            except Exception as e:
                logger.error(f"批次 {i + 1} 处理时出错: {e}")
                time.sleep(interval_seconds)  # 出错后仍等待间隔时间

        # 打印总结信息
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"模拟完成! 总计发送 {self.sent_data_count} 条数据, 用时 {duration:.1f} 秒")


def main():
    """主函数，处理命令行参数并运行模拟器"""
    parser = argparse.ArgumentParser(description="Steam推荐系统数据流模拟器")
    parser.add_argument('--data-path', type=str, default='data', help='数据目录路径')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000/api/data', help='API端点URL')
    parser.add_argument('--batches', type=int, default=10, help='要发送的批次数')
    parser.add_argument('--batch-size', type=int, default=10, help='每批次的记录数')
    parser.add_argument('--interval', type=int, default=5, help='批次之间的间隔秒数')

    args = parser.parse_args()

    try:
        simulator = SteamDataSimulator(args.data_path, args.api_url)
        simulator.run_simulation(args.batches, args.batch_size, args.interval)

    except Exception as e:
        logger.error(f"模拟器运行失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())