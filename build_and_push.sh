#!/bin/bash

# Get the account ID
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region
region=$(aws configure get region)

# Set the repository name
repository_name=whisper-webui

# Get the login command from ECR and execute it
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com.cn

# Create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${repository_name} --region $region || aws ecr create-repository --repository-name ${repository_name} --region $region

# Build the docker image
echo "Building the Docker image..."
docker build -t ${repository_name} .

# Tag the image
fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${repository_name}:latest"
docker tag ${repository_name} ${fullname}

# Push the image
echo "Pushing the Docker image..."
docker push ${fullname}

echo "Image pushed to: $fullname"
