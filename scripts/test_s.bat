@echo off
REM test_s.bat - 针对小型数据集的EMR集群创建批处理文件

echo 创建小型数据集EMR集群...

aws emr create-cluster ^
 --name "SRS_SMALL_TEST" ^
 --log-uri "s3://aws-logs-976193243904-ap-southeast-1/elasticmapreduce" ^
 --release-label "emr-7.8.0" ^
 --service-role "arn:aws:iam::976193243904:role/service-role/AmazonEMR-ServiceRole-20250414T041618" ^
 --ec2-attributes "{\"InstanceProfile\":\"EC2SteamRecommenderS3Access\",\"EmrManagedMasterSecurityGroup\":\"sg-03f6c1b42590c9390\",\"EmrManagedSlaveSecurityGroup\":\"sg-03f6c1b42590c9390\",\"KeyName\":\"SRS 1\",\"SubnetIds\":[\"subnet-0286f0b19639c4657\"]}" ^
 --tags "for-use-with-amazon-emr-managed-policies=true" ^
 --applications Name=Hadoop Name=Hive Name=JupyterEnterpriseGateway Name=Livy Name=Spark ^
 --configurations "[{\"Classification\":\"spark-hive-site\",\"Properties\":{\"hive.metastore.client.factory.class\":\"com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory\"}},{\"Classification\":\"hadoop-env\",\"Properties\":{},\"Configurations\":[{\"Classification\":\"export\",\"Properties\":{\"HADOOP_CLASSPATH\":\"$HADOOP_CLASSPATH:/usr/lib/hadoop/hadoop-aws.jar:/usr/lib/hadoop/aws-java-sdk-*.jar\"}}]},{\"Classification\":\"spark-defaults\",\"Properties\":{\"spark.hadoop.fs.s3a.impl\":\"org.apache.hadoop.fs.s3a.S3AFileSystem\",\"spark.driver.extraClassPath\":\"/usr/lib/hadoop/hadoop-aws.jar:/usr/lib/hadoop/aws-java-sdk-bundle-*.jar\",\"spark.executor.extraClassPath\":\"/usr/lib/hadoop/hadoop-aws.jar:/usr/lib/hadoop/aws-java-sdk-bundle-*.jar\",\"spark.jars.packages\":\"org.apache.hadoop:hadoop-aws:3.2.2,com.amazonaws:aws-java-sdk-bundle:1.11.901\"}}]" ^
 --bootstrap-actions "[{\"Path\":\"s3://steam-project-data-976193243904/scripts/bootstrap.sh\",\"Name\":\"Install Python Dependencies\"}]" ^
 --instance-groups "[{\"InstanceCount\":1,\"InstanceGroupType\":\"MASTER\",\"Name\":\"Primary\",\"InstanceType\":\"m5.xlarge\",\"EbsConfiguration\":{\"EbsBlockDeviceConfigs\":[{\"VolumeSpecification\":{\"VolumeType\":\"gp2\",\"SizeInGB\":32},\"VolumesPerInstance\":2}]}},{\"InstanceCount\":1,\"InstanceGroupType\":\"CORE\",\"Name\":\"Core\",\"InstanceType\":\"m5.xlarge\",\"EbsConfiguration\":{\"EbsBlockDeviceConfigs\":[{\"VolumeSpecification\":{\"VolumeType\":\"gp2\",\"SizeInGB\":32},\"VolumesPerInstance\":2}]}}]" ^
 --steps "[{\"Name\":\"upload_scripts\",\"ActionOnFailure\":\"CONTINUE\",\"Jar\":\"command-runner.jar\",\"Properties\":\"\",\"Args\":[\"aws\",\"s3\",\"cp\",\"s3://steam-project-data-976193243904/scripts/create_small_dataset_s.py\",\"s3://steam-project-data-976193243904/scripts/create_small_dataset_s.py\"],\"Type\":\"CUSTOM_JAR\"},{\"Name\":\"upload_scripts2\",\"ActionOnFailure\":\"CONTINUE\",\"Jar\":\"command-runner.jar\",\"Properties\":\"\",\"Args\":[\"aws\",\"s3\",\"cp\",\"s3://steam-project-data-976193243904/scripts/main_s.py\",\"s3://steam-project-data-976193243904/scripts/main_s.py\"],\"Type\":\"CUSTOM_JAR\"},{\"Name\":\"clone_git_s\",\"ActionOnFailure\":\"CONTINUE\",\"Jar\":\"s3://ap-southeast-1.elasticmapreduce/libs/script-runner/script-runner.jar\",\"Properties\":\"\",\"Args\":[\"s3://steam-project-data-976193243904/scripts/clone_git_s.sh\"],\"Type\":\"CUSTOM_JAR\"},{\"Name\":\"run_main_s\",\"ActionOnFailure\":\"CONTINUE\",\"Jar\":\"s3://ap-southeast-1.elasticmapreduce/libs/script-runner/script-runner.jar\",\"Properties\":\"\",\"Args\":[\"s3://steam-project-data-976193243904/scripts/run_main_s.sh\"],\"Type\":\"CUSTOM_JAR\"}]" ^
 --scale-down-behavior "TERMINATE_AT_TASK_COMPLETION" ^
 --ebs-root-volume-size "32" ^
 --auto-termination-policy "{\"IdleTimeout\":3600}" ^
 --region "ap-southeast-1"

echo EMR集群创建命令已执行