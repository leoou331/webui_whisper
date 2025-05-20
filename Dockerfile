FROM python:3.9-slim

WORKDIR /app

# 检查 Debian 版本并设置相应的源
RUN cat /etc/os-release | grep VERSION_CODENAME | cut -d= -f2 > /tmp/debian_version.txt && \
    DEBIAN_VERSION=$(cat /tmp/debian_version.txt) && \
    rm -rf /etc/apt/sources.list.d/* && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEBIAN_VERSION main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEBIAN_VERSION-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ $DEBIAN_VERSION-backports main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security $DEBIAN_VERSION-security main contrib non-free" >> /etc/apt/sources.list

# Install system dependencies including ffmpeg
RUN apt-get clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

# 检查 sagemaker 是否安装成功
RUN python -c "import sagemaker; print(f'SageMaker Version: {sagemaker.__version__}')"

# Copy application
COPY app.py .

# Create templates directory
RUN mkdir -p templates

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Run the application
CMD ["python", "app.py"]
