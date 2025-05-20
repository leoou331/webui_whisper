import requests
import json
import time
import os
import sys
import logging
import sseclient  # 需要安装 sseclient-py 库来处理 Server-Sent Events

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('whisper_client')

# 从环境变量获取配置
BASE_URL = os.environ.get("WHISPER_API_URL", "http://localhost:8080")
USERNAME = os.environ.get("WHISPER_USERNAME", "")
PASSWORD = os.environ.get("WHISPER_PASSWORD", "")
AUDIO_FILE = os.environ.get("WHISPER_AUDIO_FILE", "")

logger.info(f"配置信息: API URL = {BASE_URL}")
logger.info(f"音频文件: {AUDIO_FILE}")

def login(username, password):
    """登录并获取会话cookie"""
    login_url = f"{BASE_URL}/login"
    logger.info(f"尝试登录: {login_url}")
    
    # 创建一个会话对象，可以自动处理cookies
    session = requests.Session()
    
    try:
        # 发送登录请求
        logger.debug(f"发送POST请求到 {login_url}")
        response = session.post(
            login_url,
            data={
                "username": username,
                "password": password
            },
            allow_redirects=False,  # 不自动跟随重定向，以便检查状态码
            timeout=10  # 添加超时
        )
        
        logger.debug(f"登录响应状态码: {response.status_code}")
        logger.debug(f"响应头: {response.headers}")
        
        # 检查登录是否成功 (通常成功登录后会返回302重定向状态码)
        if response.status_code == 302:
            logger.info("登录成功！")
            return session
        else:
            logger.error(f"登录失败！状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text[:200]}...")  # 只打印前200个字符
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"登录请求异常: {str(e)}")
        return None

def transcribe_audio(session, audio_file_path):
    """上传并转录音频文件"""
    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        logger.error(f"错误: 文件 {audio_file_path} 不存在")
        return
        
    # 检查文件是否为MP3格式
    if not audio_file_path.lower().endswith('.mp3'):
        logger.error("错误: 只支持MP3文件")
        return
    
    # 第1步: 上传文件到 /transcribe 端点
    transcribe_url = f"{BASE_URL}/transcribe"
    logger.info(f"上传文件到: {transcribe_url}")
    
    try:
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"文件大小: {file_size / 1024:.2f} KB")
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': (os.path.basename(audio_file_path), f, 'audio/mpeg')}
            
            logger.info("正在上传音频文件...")
            response = session.post(
                transcribe_url,
                files=files,
                timeout=60  # 较长的超时时间用于上传大文件
            )
        
        # 检查上传是否成功
        logger.debug(f"上传响应状态码: {response.status_code}")
        logger.debug(f"响应头: {response.headers}")
        
        if response.status_code != 200:
            logger.error(f"上传失败！状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text[:200]}...")
            return
        
        logger.info("文件上传成功，准备接收转录结果...")
        logger.debug(f"响应内容: {response.text[:200]}...")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"上传文件时发生错误: {str(e)}")
        return
    
    # 第2步: 连接到 SSE 流以获取实时结果
    stream_url = f"{BASE_URL}/stream"
    logger.info(f"连接到事件流: {stream_url}")
    
    try:
        response = session.get(stream_url, stream=True, timeout=5)
        logger.debug(f"SSE连接状态码: {response.status_code}")
        
        # 使用 sseclient 处理 Server-Sent Events
        logger.info("开始处理事件流...")
        client = sseclient.SSEClient(response)
        
        # 保存完整转录结果
        full_transcript = ""
        
        # 处理事件流
        print("\n开始接收转录结果流:")
        print("-" * 50)
        
        event_count = 0
        timeout_start = time.time()
        timeout_limit = 300  # 5分钟超时限制
        
        for event in client.events():
            # 重置超时计时器
            timeout_start = time.time()
            event_count += 1
            
            logger.debug(f"收到事件 #{event_count}: {event.data[:50]}...")
            
            try:
                data = json.loads(event.data)
                event_type = data.get("type")
                
                if event_type == "init":
                    logger.info(f"初始化转录 - 总计 {data['total_segments']} 个音频段")
                    print(f"初始化转录 - 总计 {data['total_segments']} 个音频段")
                    
                elif event_type == "progress":
                    progress = data["progress"]
                    current_segment = data["current_segment"]
                    transcript = data["transcript"]
                    
                    # 记录进度日志（每10%记录一次）
                    if progress % 10 == 0:
                        logger.info(f"转录进度: {progress}% (段 {current_segment})")
                    
                    # 打印进度条
                    progress_bar = f"[{'#' * int(progress/2)}{' ' * (50-int(progress/2))}] {progress}%"
                    print(f"\r处理中: {progress_bar} (段 {current_segment})", end="")
                    
                    # 保存最新的转录结果
                    full_transcript = transcript
                    
                elif event_type == "complete":
                    logger.info("转录完成!")
                    print("\n\n转录完成!")
                    print("-" * 50)
                    full_transcript = data["transcript"]
                    break
                    
                elif event_type == "error":
                    logger.error(f"转录错误: {data['message']}")
                    print(f"\n转录错误: {data['message']}")
                    break
                else:
                    logger.warning(f"收到未知类型的事件: {event_type}")
                
            except json.JSONDecodeError as e:
                logger.error(f"解析事件数据失败: {str(e)}")
                logger.error(f"原始数据: {event.data}")
                continue
            
            # 检查超时
            if time.time() - timeout_start > timeout_limit:
                logger.error(f"等待事件超时! 已经等待了 {timeout_limit} 秒")
                print("\n处理超时! 请检查服务器状态。")
                break
        
        logger.info(f"事件流处理完成，共接收 {event_count} 个事件")
        return full_transcript
        
    except requests.exceptions.RequestException as e:
        logger.error(f"连接事件流时发生错误: {str(e)}")
        print(f"\n连接错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"处理事件流时发生未知错误: {str(e)}")
        print(f"\n处理错误: {str(e)}")
        return None

def main():
    # 检查必要的环境变量
    if not USERNAME or not PASSWORD:
        logger.error("错误: 请设置 WHISPER_USERNAME 和 WHISPER_PASSWORD 环境变量")
        print("错误: 请设置 WHISPER_USERNAME 和 WHISPER_PASSWORD 环境变量")
        return
    
    if not AUDIO_FILE:
        logger.error("错误: 请设置 WHISPER_AUDIO_FILE 环境变量指定音频文件路径")
        print("错误: 请设置 WHISPER_AUDIO_FILE 环境变量指定音频文件路径")
        return
    
    try:
        # 步骤1: 登录
        logger.info("=== 步骤1: 登录 ===")
        session = login(USERNAME, PASSWORD)
        if not session:
            return
        
        # 步骤2: 转录音频
        logger.info("=== 步骤2: 转录音频 ===")
        transcript = transcribe_audio(session, AUDIO_FILE)
        
        # 显示最终结果
        if transcript:
            logger.info("=== 转录完成, 保存结果 ===")
            print("\n最终转录结果:")
            print("-" * 50)
            print(transcript)
            
            # 将结果保存到文件
            output_file = f"{os.path.splitext(AUDIO_FILE)[0]}_transcript.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript)
            logger.info(f"转录结果已保存到: {output_file}")
            print(f"\n转录结果已保存到: {output_file}")
    
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
        print("\n程序已被用户中断")
    except Exception as e:
        logger.error(f"程序运行时发生未捕获的异常: {str(e)}", exc_info=True)
        print(f"\n程序错误: {str(e)}")

if __name__ == "__main__":
    main()
