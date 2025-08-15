#!/bin/bash

# 设置变量
REGION="cn-northwest-1"
ACCOUNT_ID="104946057020"
REPOSITORY_NAME="sagemaker_endpoint/whisper-turbo"
IMAGE_TAG="latest"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com.cn/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "开始构建Whisper SageMaker镜像..."

# 登录到ECR
echo "登录到ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com.cn

# 同时登录到AWS中国区ECR以拉取基础镜像
echo "登录到AWS中国区ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 727897471807.dkr.ecr.${REGION}.amazonaws.com.cn

# 构建镜像
echo "构建Docker镜像..."
docker build -f Dockerfile.whisper -t ${IMAGE_URI} .

if [ $? -ne 0 ]; then
    echo "镜像构建失败!"
    exit 1
fi

# 推送镜像到ECR
echo "推送镜像到ECR..."
docker push ${IMAGE_URI}

if [ $? -ne 0 ]; then
    echo "镜像推送失败!"
    exit 1
fi

echo "镜像构建和推送完成!"
echo "镜像URI: ${IMAGE_URI}"

# 输出用于SageMaker的镜像URI
echo ""
echo "在SageMaker中使用以下镜像URI:"
echo "${IMAGE_URI}"
