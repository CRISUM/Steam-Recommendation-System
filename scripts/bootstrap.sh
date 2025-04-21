#!/bin/bash
# 不使用set -e，而是手动检查错误

echo "开始执行引导脚本..."

# 安装S3A依赖
echo "检查S3A依赖..."
if sudo yum list installed | grep -q "hadoop-aws"; then
    echo "hadoop-aws已安装"
else
    echo "尝试安装hadoop-aws..."
    sudo yum install -y hadoop-aws || echo "hadoop-aws安装失败，但继续执行"
fi

if sudo yum list installed | grep -q "aws-java-sdk-bundle"; then
    echo "aws-java-sdk-bundle已安装"
else
    echo "尝试安装aws-java-sdk-bundle..."
    sudo yum install -y aws-java-sdk-bundle || echo "aws-java-sdk-bundle安装失败，但继续执行"
fi

# 安装Python包，逐个安装以隔离失败
echo "安装Python包..."
python_packages=("pandas" "numpy" "scikit-learn" "matplotlib" "seaborn" "pyspark" "joblib" "jupyterlab" "boto3" "fsspec" "s3fs")

for package in "${python_packages[@]}"; do
    echo "安装 $package..."
    sudo pip3 install $package || echo "$package安装失败，但继续执行"
done

# 检查并复制AWS SDK JAR文件
echo "配置Hadoop S3A文件系统..."

# 查找hadoop-aws.jar文件
hadoop_aws_jar=$(find /usr/lib -name "hadoop-aws*.jar" | head -1)
if [ -n "$hadoop_aws_jar" ]; then
    echo "找到hadoop-aws JAR: $hadoop_aws_jar"
    sudo mkdir -p /usr/lib/spark/jars/
    sudo cp "$hadoop_aws_jar" /usr/lib/spark/jars/ || echo "复制hadoop-aws JAR失败，但继续执行"
else
    echo "未找到hadoop-aws JAR文件"
fi

# 查找aws-java-sdk JAR文件
aws_sdk_jars=$(find /usr/lib -name "aws-java-sdk*.jar")
if [ -n "$aws_sdk_jars" ]; then
    echo "找到aws-java-sdk JAR文件"
    sudo mkdir -p /usr/lib/spark/jars/
    for jar in $aws_sdk_jars; do
        sudo cp "$jar" /usr/lib/spark/jars/ || echo "复制 $jar 失败，但继续执行"
    done
else
    echo "未找到aws-java-sdk JAR文件"
fi

# 尝试使用预装版本
if [ -d "/usr/lib/hadoop-lzo/lib" ]; then
    echo "使用Hadoop LZO lib目录下的JAR文件"
    sudo cp -r /usr/lib/hadoop-lzo/lib/* /usr/lib/spark/jars/ || echo "从hadoop-lzo复制JAR失败"
fi

# 列出spark jars目录内容
echo "Spark JAR目录内容:"
ls -la /usr/lib/spark/jars/

echo "引导脚本执行完成"
exit 0