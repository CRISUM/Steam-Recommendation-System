#!/bin/bash
# clone_repo_s.sh - 小型数据集版本
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

echo "开始克隆代码并设置小型数据集环境..."

# 安装git
sudo yum install -y git

# 克隆代码仓库
cd /home/hadoop
git clone https://github.com/CRISUM/Steam-Recommendation-System.git
cd Steam-Recommendation-System

# 安装依赖
echo "安装项目依赖..."
pip3 install --user -r requirements.txt

# 从S3下载小型数据集创建脚本和主脚本
echo "下载小型数据集脚本..."
aws s3 cp s3://steam-project-data-976193243904/scripts/create_small_dataset_s.py ./create_small_dataset_s.py
aws s3 cp s3://steam-project-data-976193243904/scripts/main_s.py ./main_s.py

# 确保脚本有执行权限
chmod +x create_small_dataset_s.py
chmod +x main_s.py

# 创建小型数据集
echo "创建小型数据集..."
python3 main_s.py --create-dataset

echo "代码克隆和小型数据集环境设置完成"
echo "验证pandas是否安装:"
pip3 show pandas || echo "pandas未安装！"