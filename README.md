# Whisper Web UI

一个基于Flask的Web应用程序，使用OpenAI的Whisper模型通过AWS SageMaker进行音频转录。

## 项目概述

Whisper Web UI为用户提供了一个简单的界面，允许他们上传MP3音频文件并使用AWS SageMaker上部署的Whisper模型将其转录成文本。该应用支持处理任意长度的音频文件，将其分成30秒的片段进行处理，并实时显示转录结果。

### 主要功能

- 用户认证系统
- MP3文件上传和处理
- 通过SageMaker端点进行实时音频转录
- 支持长音频文件，自动分段处理
- 实时显示转录进度和结果
- 程序化API访问支持

## 部署架构

该应用程序设计为在AWS EKS（Elastic Kubernetes Service）上运行，使用以下AWS服务：

- **EKS**：托管Kubernetes集群
- **ECR**：存储Docker镜像
- **Secrets Manager**：存储应用程序认证凭据
- **SageMaker**：托管Whisper模型并提供推理端点
- **IAM**：管理服务访问权限

## 前置要求

- AWS账户
- 已安装并配置AWS CLI
- Docker
- kubectl
- eksctl
- AWS中国区域需要AWS账户具有ICP备案

## 安装步骤

### 1. 环境配置

1. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/whisper-webui.git
   cd whisper-webui
   ```

2. 创建AWS资源：
   ```bash
   # 编辑CloudFormation参数
   cp create_stack.sh.sample create_stack.sh
   # 修改create_stack.sh中的参数
   chmod +x create_stack.sh
   ./create_stack.sh
   ```

### 2. 创建Secrets

1. 在AWS Secrets Manager中创建包含用户凭证的密钥：
   ```bash
   aws secretsmanager create-secret --name whisper-app-credentials --secret-string '{"admin":"password","user1":"password1"}'
   ```

### 3. SageMaker设置

确保您已经在SageMaker中部署了Whisper模型并获取了端点名称。

### 4. 构建和推送Docker镜像

```bash
# 编辑Dockerfile如有必要
cp .env.sample .env
# 修改.env中的环境变量
chmod +x build_and_push.sh
./build_and_push.sh
```

### 5. 部署到EKS

```bash
# 创建EKS集群（如果尚未创建）
eksctl create cluster --name whisper-cluster --region cn-northwest-1

# 编辑和部署Kubernetes资源
cp webui_whisper_deployment.yaml.sample webui_whisper_deployment.yaml
# 编辑webui_whisper_deployment.yaml中的参数
kubectl apply -f webui_whisper_deployment.yaml
```

## 使用方法

### Web界面

1. 访问应用程序URL（可通过以下命令获取）：
   ```bash
   kubectl get ingress -n whisper-app
   ```

2. 使用Secrets Manager中配置的用户名和密码登录。

3. 上传MP3文件并等待转录结果。

### 编程访问

可以使用提供的Python客户端脚本以编程方式访问API：

```bash
# 设置环境变量
source .env
# 运行客户端
python demo_client.py
```

客户端脚本将：
1. 登录到应用程序
2. 上传指定的音频文件
3. 接收转录结果流
4. 将最终转录保存到文本文件

## 配置选项

### 环境变量

应用程序支持以下环境变量：

- `SECRET_NAME`: AWS Secrets Manager中的密钥名称
- `SAGEMAKER_ENDPOINT`: SageMaker端点名称
- `AWS_REGION`: AWS区域

客户端脚本支持以下环境变量：

- `WHISPER_API_URL`: Whisper Web UI的URL
- `WHISPER_USERNAME`: 登录用户名
- `WHISPER_PASSWORD`: 登录密码
- `WHISPER_AUDIO_FILE`: 要转录的音频文件路径

## 开发

### 本地开发

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行应用程序：
   ```bash
   python app.py
   ```

3. 访问 http://localhost:8080

### 自定义API

项目提供了以下几个主要API端点：

- `/login`: 用于用户认证，接受POST请求，包含用户名和密码
- `/transcribe`: 接收音频文件上传的端点，处理文件并启动转录过程
- `/stream`: 提供实时转录结果的Server-Sent Events (SSE)流
- `/api/transcribe`: 一站式转录API，适用于程序化访问，直接返回JSON格式的完整转录结果

典型的API调用流程：
1. 向`/login`发送POST请求进行认证
2. 向`/transcribe`上传音频文件
3. 连接到`/stream`获取实时转录结果

或者:
- 直接向`/api/transcribe`发送带有音频文件的POST请求，获取完整转录结果（仍需先登录）

详细的API使用方法可以参考`demo_client.py`中的示例代码。

## 故障排除

### 常见问题

1. **SageMaker连接错误**：
   - 检查IAM角色权限
   - 验证SageMaker端点是否活跃
   - 检查区域设置

2. **会话管理问题**：
   - 确保客户端正确处理cookies
   - 验证Flask会话配置

3. **转录流中断**：
   - 增加客户端超时设置
   - 检查网络连接

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT

## 致谢

- 基于OpenAI的Whisper模型
- 使用Flask框架构建
- 使用AWS SageMaker进行模型托管
