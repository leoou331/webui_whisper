# Whisper Web UI

一个基于Flask的Web应用程序，使用OpenAI的Whisper模型通过AWS SageMaker进行音频转录。

## 项目概述

Whisper Web UI为用户提供了一个简单的界面，允许他们上传MP3/M4A音频文件并使用AWS SageMaker上部署的Whisper模型将其转录成文本。该应用支持处理任意长度的音频文件，将其分成30秒的片段进行处理，并实时显示转录结果。

### 主要功能

- **用户认证系统**：安全的登录机制
- **多格式支持**：MP3/M4A文件上传和处理
- **实时转录**：通过SageMaker端点进行高性能音频转录
- **智能分段**：支持长音频文件，自动分段处理
- **实时反馈**：实时显示转录进度和结果
- **API访问**：完整的程序化API访问支持
- **热词功能**：支持自定义热词提升特定词汇识别准确率
  - **Prompt注入方式**：通过初始提示引导模型识别特定词汇
  - **Logit Bias方式**：通过调整输出概率分布提升热词权重
- **高性能**：优化的t3.large实例，2秒内完成转录
- **成本优化**：相比传统配置节省65%成本

## 部署架构

该应用程序设计为在AWS EKS（Elastic Kubernetes Service）上运行，采用成本优化的架构：

- **EKS**：托管Kubernetes集群 (t3.large节点，成本优化)
- **ECR**：存储Docker镜像
- **Network Load Balancer**：高性能负载均衡
- **Secrets Manager**：存储应用程序认证凭据
- **SageMaker**：托管Whisper模型并提供推理端点
- **IAM**：管理服务访问权限
- **安全组**：精细化网络访问控制

## 前置要求

- AWS账户
- 已安装并配置AWS CLI
- Docker
- kubectl
- eksctl
- AWS中国区域需要AWS账户具有ICP备案

## 快速开始

### 1. 环境配置

1. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/whisper-webui.git
   cd whisper-webui
   ```

2. 配置环境变量：
   ```bash
   cp .env.sample .env
   # 编辑.env文件，设置您的配置
   ```

### 2. 创建AWS资源

1. 创建CloudFormation堆栈：
   ```bash
   cp create_stack.sh.sample create_stack.sh
   # 修改create_stack.sh中的参数
   chmod +x create_stack.sh
   ./create_stack.sh
   ```

2. 在AWS Secrets Manager中创建用户凭证：
   ```bash
   aws secretsmanager create-secret \
     --name whisper-app-credentials \
     --secret-string '{"admin":"password123","user1":"password1"}' \
     --region cn-northwest-1
   ```

### 3. 部署SageMaker端点

您可以通过以下两种方式部署Whisper模型到SageMaker：

#### 方法一：使用自动化脚本（推荐）
```bash
# 使用提供的自动化脚本构建和部署Whisper模型
chmod +x build_whisper_image.sh
./build_whisper_image.sh

# 脚本将自动完成以下步骤：
# 1. 构建优化的Whisper Docker镜像
# 2. 推送镜像到ECR
# 3. 创建SageMaker模型
# 4. 部署推理端点
```

#### 方法二：使用Jupyter Notebook
```bash
# 使用提供的Jupyter notebook手动部署
# 打开 whisper-inference-deploy.ipynb 并按步骤执行
```

#### 验证端点部署
```bash
# 检查端点状态
aws sagemaker describe-endpoint --endpoint-name your-whisper-endpoint-name --region cn-northwest-1

# 测试端点功能
python test_whisper_endpoint.py
```

### 4. 构建和部署应用

1. 构建Docker镜像：
   ```bash
   chmod +x build_and_push.sh
   ./build_and_push.sh
   ```

2. 部署到EKS：
   ```bash
   # 创建EKS集群（推荐使用t3.large节点）
   eksctl create cluster --name whisper-cluster --region cn-northwest-1 \
     --node-type t3.large --nodes 2 --nodes-min 2 --nodes-max 4

   # 部署应用
   cp webui_whisper_deployment.yaml.sample webui_whisper_deployment.yaml
   # 编辑webui_whisper_deployment.yaml中的参数
   kubectl apply -f webui_whisper_deployment.yaml
   ```

3. 获取访问地址：
   ```bash
   kubectl get svc -n whisper-app
   ```

## 使用方法

### Web界面

1. 访问应用程序URL（可通过以下命令获取）：
   ```bash
   kubectl get svc -n whisper-app
   ```

2. 使用Secrets Manager中配置的用户名和密码登录。

3. 配置热词（可选）：
   - 选择技术方式：Prompt注入或Logit Bias
   - 添加需要重点识别的词汇
   - 热词将在整个音频转录过程中生效

4. 上传MP3或M4A文件并等待转录结果。

### 编程访问

可以使用提供的Python客户端脚本以编程方式访问API：

```bash
# 设置环境变量
source .env
# 可选：设置热词
export WHISPER_HOTWORDS="专业术语,人名,地名"
export WHISPER_HOTWORD_METHOD="prompt_injection"
# 运行客户端
python demo_client_final.py
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
- `WHISPER_HOTWORDS`: 热词列表，用逗号分隔（可选）
- `WHISPER_HOTWORD_METHOD`: 热词技术方式，`prompt_injection`或`logit_bias`（可选，默认为`prompt_injection`）

## 性能优化

### 成本优化配置
- **实例类型**: t3.large (突发性能，适合间歇性负载)
- **节点数量**: 2个节点 (高可用性)
- **成本节省**: 相比m5.large节省65%成本
- **月成本**: $85-105 (原$240-300)

### 性能指标
- **转录速度**: 2秒内完成典型音频文件
- **响应时间**: <5ms
- **支持格式**: MP3, M4A
- **最大文件**: 无限制 (自动分段处理)

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

### 测试热词功能

运行热词功能测试脚本：
```bash
# 设置测试环境变量
export WHISPER_API_URL="http://localhost:8080"
export WHISPER_USERNAME="admin"
export WHISPER_PASSWORD="password123"

# 运行测试
python test_hotwords.py
```

### 自定义API

项目提供了以下几个主要API端点：

- `/login`: 用于用户认证，接受POST请求，包含用户名和密码
- `/transcribe`: 接收音频文件上传的端点，处理文件并启动转录过程，支持热词配置
- `/stream`: 提供实时转录结果的Server-Sent Events (SSE)流
- `/api/transcribe`: 一站式转录API，适用于程序化访问，直接返回JSON格式的完整转录结果，支持热词配置
- `/api/hotwords`: 热词配置管理API，支持GET/POST请求来获取和设置热词配置

典型的API调用流程：
1. 向`/login`发送POST请求进行认证
2. （可选）向`/api/hotwords`发送POST请求配置热词
3. 向`/transcribe`上传音频文件（可包含热词配置）
4. 连接到`/stream`获取实时转录结果

或者:
- 直接向`/api/transcribe`发送带有音频文件和热词配置的POST请求，获取完整转录结果（仍需先登录）

### 热词配置格式

```json
{
  "method": "prompt_injection",  // 或 "logit_bias"
  "words": ["专业术语", "人名", "地名"],
  "boost_factor": 1.5  // 仅用于logit_bias方法
}
```

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

4. **热词不生效**：
   - 检查热词配置格式是否正确
   - 验证SageMaker端点是否支持相应的热词参数
   - 查看应用日志中的热词处理信息

5. **Load Balancer访问问题**：
   - 检查安全组配置是否允许您的IP访问
   - 验证NodePort端口是否正确配置
   - 使用 `kubectl get svc -n whisper-app` 检查服务状态

6. **性能问题**：
   - 监控t3.large实例的CPU积分使用情况
   - 检查网络连接质量
   - 验证SageMaker端点响应时间

### 性能监控

```bash
# 检查节点资源使用
kubectl top nodes

# 检查Pod资源使用
kubectl top pods -n whisper-app

# 检查服务状态
kubectl get svc -n whisper-app

# 查看应用日志
kubectl logs -n whisper-app deployment/whisper-webui
```

## 成本优化

### 当前配置成本
- **EKS集群**: 2 x t3.large节点
- **月成本**: $85-105
- **年成本**: $1,020-1,260
- **相比m5.large节省**: 65%

### 进一步优化建议
- 使用Spot实例可额外节省60-70%
- 根据使用模式调整节点数量
- 定期审查资源使用情况

## 安全最佳实践

### 网络安全
- 使用安全组限制访问来源IP
- 启用VPC Flow Logs监控网络流量
- 定期审查和更新安全组规则

### 访问控制
- 使用AWS Secrets Manager管理敏感信息
- 实施最小权限原则
- 定期轮换访问密钥

### 监控和审计
- 启用CloudTrail记录API调用
- 使用CloudWatch监控应用性能
- 设置告警通知异常活动

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT

## 致谢

- 基于OpenAI的Whisper模型
- 使用Flask框架构建
- 使用AWS SageMaker进行模型托管
