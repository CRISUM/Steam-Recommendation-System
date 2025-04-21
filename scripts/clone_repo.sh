#!/bin/bash
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
sudo yum install -y git
cd /home/hadoop
git clone https://github.com/CRISUM/Steam-Recommendation-System.git
cd Steam-Recommendation-System
pip3 install --user -r requirements.txt

echo "代码克隆和依赖安装完成"
echo "验证pandas是否安装:"
pip3 show pandas || echo "pandas未安装！"