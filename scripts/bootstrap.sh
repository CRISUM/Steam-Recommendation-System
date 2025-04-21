#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "开始执行引导脚本..."

# 使用--ignore-installed选项来避免尝试卸载系统包
echo "安装Python依赖包..."
sudo pip3 install --ignore-installed pandas numpy scikit-learn matplotlib seaborn pyspark joblib jupyterlab boto3 fsspec s3fs
# 如果上面的命令失败，尝试逐个安装重要的包
if [ $? -ne 0 ]; then
    echo "整体安装失败，尝试逐个安装关键包..."
    for package in numpy pandas scikit-learn pyspark boto3; do
        echo "安装 $package..."
        sudo pip3 install --ignore-installed $package || echo "警告: $package 安装失败，但继续执行"
    done
fi

echo "引导脚本执行完成"
exit 0  # 确保脚本返回成功状态