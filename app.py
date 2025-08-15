import os
import boto3
import json
import tempfile
import time
import numpy as np
import logging
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, stream_with_context, jsonify
from werkzeug.utils import secure_filename
from functools import wraps
from pydub import AudioSegment
import sagemaker
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import StringDeserializer

# 配置日志
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.logger.setLevel(logging.INFO)

# AWS region
region_name = os.environ.get('AWS_REGION', 'cn-northwest-1')
app.logger.info(f"Using AWS Region: {region_name}")

# 记录AWS环境变量，帮助调试
if os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'):
    app.logger.info("Running with AWS container credentials")
elif os.environ.get('AWS_WEB_IDENTITY_TOKEN_FILE'):
    app.logger.info("Running with AWS web identity token")
else:
    app.logger.info("No specific AWS credential method detected")

# Initialize AWS clients
secretsmanager = boto3.client('secretsmanager', region_name=region_name)

# Get secret name from environment variable
SECRET_NAME = os.environ.get('SECRET_NAME', 'whisper-app-credentials')
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT', 'whisper-endpoint')
app.logger.info(f"SageMaker Endpoint Name: {ENDPOINT_NAME}")

# 初始化 SageMaker Predictor
def get_predictor():
    try:
        # 确保使用正确的区域名称
        boto_session = boto3.Session(region_name=region_name)
        sagemaker_session = sagemaker.session.Session(boto_session=boto_session)
        
        app.logger.info(f"Initializing SageMaker predictor with endpoint: {ENDPOINT_NAME} in region: {region_name}")
        
        predictor = sagemaker.Predictor(
            endpoint_name=ENDPOINT_NAME,
            serializer=NumpySerializer(),
            deserializer=StringDeserializer("utf-8"),
            sagemaker_session=sagemaker_session
        )
        return predictor
    except Exception as e:
        app.logger.error(f"Error creating predictor: {str(e)}")
        # 记录更多调试信息
        app.logger.error(f"AWS Region setting: {region_name}")
        app.logger.error(f"Endpoint name: {ENDPOINT_NAME}")
        return None

def get_credentials():
    """Retrieve credentials from AWS Secrets Manager"""
    try:
        response = secretsmanager.get_secret_value(SecretId=SECRET_NAME)
        secret = json.loads(response['SecretString'])
        return secret
    except Exception as e:
        app.logger.error(f"Error fetching credentials: {str(e)}")
        return {}

def process_hotwords_config(request):
    """处理热词配置"""
    try:
        hotwords_json = request.form.get('hotwords', '[]')
        hotwords = json.loads(hotwords_json) if hotwords_json else []
        method = request.form.get('hotword_method', 'prompt_injection')
        
        config = {
            'method': method,
            'words': hotwords,
            'boost_factor': 1.5  # 默认增强因子，用于logit_bias方法
        }
        
        app.logger.info(f"处理热词配置: {config}")
        return config
    except Exception as e:
        app.logger.error(f"处理热词配置时出错: {str(e)}")
        return {'method': 'prompt_injection', 'words': [], 'boost_factor': 1.5}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        credentials = get_credentials()
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not credentials:
            flash('Error retrieving credentials. Please try again later.', 'danger')
            return render_template('login.html')
            
        if username in credentials and credentials[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/transcribe', methods=['POST', 'GET'])
@login_required
def transcribe():
    if request.method == 'GET':
        # 如果是GET请求，重定向回首页
        return redirect(url_for('index'))
        
    if 'audio_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
        
    file = request.files['audio_file']
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
        
    # 修改此处以支持M4A格式
    supported_formats = ['.mp3', '.m4a']
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file and file_ext in supported_formats:
        # 确定文件格式
        format_name = file_ext[1:]  # 去掉点号
        
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            file.save(temp.name)
            temp_filename = temp.name
            
        # 处理热词配置
        hotwords_config = process_hotwords_config(request)
        
        # 将临时文件名、格式和热词配置保存到会话中
        session['temp_filename'] = temp_filename
        session['file_format'] = format_name
        session['hotwords_config'] = hotwords_config
        
        # 明确保存会话 - 确保会话状态被持久化
        session.modified = True
        
        # 添加调试日志
        app.logger.info(f"保存临时文件: {temp_filename}, 格式: {format_name}")
        app.logger.info(f"热词配置: {hotwords_config}")
        
        # 返回包含进度页面的HTML
        return render_template('transcribe.html')
        
    else:
        flash('Invalid file format. Please upload MP3 or M4A files.', 'danger')
        return redirect(url_for('index'))

@app.route('/stream', methods=['GET'])
@login_required
def stream():
    """Stream the transcription results using SSE"""
    # 添加日志以便调试
    app.logger.info(f"Stream请求接收，会话内容: {dict(session)}")
    
    # 从会话中获取临时文件名
    temp_filename = session.get('temp_filename')
    if not temp_filename:
        app.logger.error("会话中没有找到临时文件名")
        return jsonify({"error": "No file to process"}), 400
        
    # 检查文件是否存在
    if not os.path.exists(temp_filename):
        app.logger.error(f"临时文件不存在: {temp_filename}")
        return jsonify({"error": "Temporary file not found"}), 400
        
    app.logger.info(f"开始处理临时文件: {temp_filename}")
    
    return Response(
        stream_with_context(process_audio(temp_filename)),
        mimetype='text/event-stream'
    )

def process_audio(file_path):
    """Process audio file in chunks and stream results"""
    try:
        # 获取 predictor 实例
        predictor = get_predictor()
        if not predictor:
            raise Exception("Failed to create SageMaker predictor")
            
        # 获取热词配置
        hotwords_config = session.get('hotwords_config', {'method': 'prompt_injection', 'words': [], 'boost_factor': 1.5})
        app.logger.info(f"使用热词配置: {hotwords_config}")
        
        # 确定文件格式 - 从会话中获取或从文件路径推断
        file_format = session.get('file_format', 'mp3')
        app.logger.info(f"加载音频文件，格式: {file_format}")
        
        # Load audio file with appropriate format
        audio_full = AudioSegment.from_file(file_path, format=file_format)
        
        # Whisper expects 16 kHz, mono channel, ≤30s
        audio_full = audio_full.set_channels(1).set_frame_rate(16000)
        
        # Set chunk size to 30 seconds (in milliseconds)
        chunk_ms = 30000
        segments = list(range(0, len(audio_full), chunk_ms))
        total_segments = len(segments)
        
        # 初始页面设置 - 使用SSE (Server-Sent Events)格式
        yield "data: " + json.dumps({
            "type": "init",
            "total_segments": total_segments,
            "hotwords_config": hotwords_config
        }) + "\n\n"
        
        # Initialize empty transcript
        transcripts = []
        
        # Process each chunk
        for i, start_ms in enumerate(segments):
            chunk = audio_full[start_ms:start_ms + chunk_ms]
            
            # Convert to numpy array and normalize to [-1, 1]
            samples = np.array(chunk.get_array_of_samples(), dtype=np.float16)
            samples = samples / 32768.0
            
            app.logger.info(f"Processing chunk {i+1}/{total_segments}, length: {len(samples)} samples")
            
            try:
                # 应用热词处理
                text = predict_with_hotwords(predictor, samples, hotwords_config)
                app.logger.info(f"Transcription result: {text[:100]}...")
                transcripts.append(text)
            except Exception as e:
                app.logger.error(f"Error calling SageMaker endpoint: {str(e)}")
                error_message = f"[Error in segment {i+1}: {str(e)}]"
                transcripts.append(error_message)
            
            # 发送更新
            current_transcript = " ".join(transcripts).strip()
            progress = min(100, int(100 * (i + 1) / total_segments))
            
            yield "data: " + json.dumps({
                "type": "progress",
                "progress": progress,
                "current_segment": i + 1,
                "transcript": current_transcript
            }) + "\n\n"
            
            # 添加小延迟让前端有时间处理
            time.sleep(0.1)
        
        # Clean up the temp file
        os.unlink(file_path)
        
        # 发送完成信号
        final_transcript = " ".join(transcripts).strip()
        yield "data: " + json.dumps({
            "type": "complete",
            "transcript": final_transcript
        }) + "\n\n"
        
    except Exception as e:
        app.logger.error(f"Error in transcription: {str(e)}")
        # 发送错误信息
        yield "data: " + json.dumps({
            "type": "error",
            "message": str(e)
        }) + "\n\n"
        
        # Delete temp file
        try:
            os.unlink(file_path)
        except:
            pass

def get_header_html():
    """Return the HTML header part"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Whisper Language Assistant</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            .form-group {{
                margin-bottom: 15px;
            }}
            .btn {{
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                margin-right: 10px;
                text-decoration: none;
                display: inline-block;
            }}
            .result {{
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .alert {{
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid transparent;
            }}
            .alert-danger {{
                color: #721c24;
                background-color: #f8d7da;
                border-color: #f5c6cb;
            }}
            .alert-success {{
                color: #155724;
                background-color: #d4edda;
                border-color: #c3e6cb;
            }}
            .progress {{
                height: 20px;
                background-color: #f5f5f5;
                border-radius: 4px;
                overflow: hidden;
            }}
            .progress-bar {{
                height: 100%;
                background-color: #4CAF50;
                color: white;
                text-align: center;
                line-height: 20px;
            }}
        </style>
        <script>
            // 添加JavaScript以处理完成后的行为
            window.onload = function() {{
                // 检查是否是最终结果页面
                if (document.querySelector('.alert-success')) {{
                    // 已完成，添加返回按钮的事件处理
                    const backBtn = document.querySelector('a[href="{url_for('index')}"]');
                    if (backBtn) {{
                        backBtn.addEventListener('click', function(e) {{
                            // 使用正常的导航，而不是表单提交
                            // 不需要阻止默认行为
                        }});
                    }}
                }}
            }};
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Whisper Language Assistant</h1>
            <div>
                <a href="{url_for('index')}" class="btn">Back</a>
                <a href="{url_for('logout')}" class="btn">Logout</a>
            </div>
        </div>
    """

def get_footer_html():
    """Return the HTML footer part"""
    return """
    </body>
    </html>
    """

# 添加一个新的API端点，直接处理文件并返回结果 (不使用SSE)
@app.route('/api/transcribe', methods=['POST'])
@login_required
def api_transcribe():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and file.filename.endswith('.mp3'):
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp:
            file.save(temp.name)
            temp_filename = temp.name
            
        try:
            # 直接处理音频并收集结果
            predictor = get_predictor()
            if not predictor:
                return jsonify({'error': 'Failed to create SageMaker predictor'}), 500
                
            # Load audio file
            audio_full = AudioSegment.from_file(temp_filename, format="mp3")
            
            # Whisper expects 16 kHz, mono channel, ≤30s
            audio_full = audio_full.set_channels(1).set_frame_rate(16000)
            
            # Set chunk size to 30 seconds (in milliseconds)
            chunk_ms = 30000
            segments = list(range(0, len(audio_full), chunk_ms))
            total_segments = len(segments)
            
            # Initialize empty transcript
            transcripts = []
            
            # Process each chunk
            for i, start_ms in enumerate(segments):
                chunk = audio_full[start_ms:start_ms + chunk_ms]
                
                # Convert to numpy array and normalize to [-1, 1]
                samples = np.array(chunk.get_array_of_samples(), dtype=np.float16)
                samples = samples / 32768.0
                
                app.logger.info(f"Processing chunk {i+1}/{total_segments}, length: {len(samples)} samples")
                
                try:
                    # 使用 predictor 而不是直接调用 boto3
                    text = predictor.predict(samples)
                    app.logger.info(f"Transcription result: {text[:100]}...")
                    transcripts.append(text)
                except Exception as e:
                    app.logger.error(f"Error calling SageMaker endpoint: {str(e)}")
                    error_message = f"[Error in segment {i+1}: {str(e)}]"
                    transcripts.append(error_message)
            
            # Clean up the temp file
            os.unlink(temp_filename)
            
            # 返回完整转录结果
            final_transcript = " ".join(transcripts).strip()
            return jsonify({
                'success': True,
                'transcript': final_transcript
            })
            
        except Exception as e:
            app.logger.error(f"Error in transcription: {str(e)}")
            # 清理临时文件
            try:
                os.unlink(temp_filename)
            except:
                pass
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Only MP3 files are allowed'}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create login.html template
    with open('templates/login.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Language Assistant - Login</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
        }
        .form-control {
            width: 100%;
            padding: 8px;
        }
        .btn {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
        }
        .alert-danger {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
    </style>
</head>
<body>
    <h1>Whisper Language Assistant</h1>
    <h2>Login</h2>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="alert alert-{{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    <form method="post">
        <div class="form-group">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" class="form-control" required>
        </div>
        <div class="form-group">
            <label for="password">Password</label>
            <input type="password" id="password" name="password" class="form-control" required>
        </div>
        <button type="submit" class="btn">Login</button>
    </form>
</body>
</html>
        ''')
    
    # Create index.html template - update it to mention support for longer files
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Language Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .btn {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-right: 10px;
        }
        .btn-secondary {
            background-color: #6c757d;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
        }
        .alert-danger {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .info {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 10px;
            margin-bottom: 15px;
        }
        .hotwords-section {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .hotwords-section h3 {
            margin-top: 0;
            color: #495057;
        }
        .method-selection {
            margin-bottom: 15px;
        }
        .method-selection label {
            margin-right: 15px;
            font-weight: normal;
        }
        .method-selection input[type="radio"] {
            margin-right: 5px;
        }
        #hotword-input {
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            margin-right: 10px;
            width: 200px;
        }
        #hotwords-display {
            margin-top: 10px;
        }
        #hotwords-list {
            margin-top: 5px;
        }
        .hotword-item {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 12px;
        }
        .hotword-remove {
            margin-left: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        .hotword-remove:hover {
            color: #ff6b6b;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Whisper Language Assistant</h1>
        <a href="{{ url_for('logout') }}" class="btn">Logout</a>
    </div>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="alert alert-{{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    <div class="info">
        <p>Upload MP3 or M4A files of any length. Long recordings will be processed in 30-second chunks with results displayed in real-time.</p>
    </div>
    
    <form method="post" action="{{ url_for('transcribe') }}" enctype="multipart/form-data">
        <div class="form-group">
            <label for="audio_file">Upload Audio File:</label>
            <input type="file" id="audio_file" name="audio_file" accept=".mp3,.m4a" required>
        </div>
        
        <div class="hotwords-section">
            <h3>热词配置 (Hotwords Configuration)</h3>
            
            <div class="method-selection">
                <label>技术方式 (Method):</label>
                <label><input type="radio" name="hotword_method" value="prompt_injection" checked> Prompt注入</label>
                <label><input type="radio" name="hotword_method" value="logit_bias"> Logit Bias</label>
            </div>
            
            <div class="form-group">
                <input type="text" id="hotword-input" placeholder="输入热词..." maxlength="50">
                <button type="button" class="btn btn-secondary" onclick="addHotword()">添加</button>
            </div>
            
            <div id="hotwords-display">
                <label>当前热词:</label>
                <div id="hotwords-list"></div>
            </div>
            
            <input type="hidden" id="hotwords-data" name="hotwords" value="">
        </div>
        
        <button type="submit" class="btn">转文字 (Transcribe)</button>
    </form>
    
    {% if transcription %}
    <div class="result">
        <h2>Transcription Result:</h2>
        <p>{{ transcription }}</p>
    </div>
    {% endif %}
    
    <script>
        let hotwords = [];
        
        function addHotword() {
            const input = document.getElementById('hotword-input');
            const word = input.value.trim();
            
            if (word && !hotwords.includes(word)) {
                hotwords.push(word);
                updateHotwordsDisplay();
                input.value = '';
            }
        }
        
        function removeHotword(word) {
            hotwords = hotwords.filter(w => w !== word);
            updateHotwordsDisplay();
        }
        
        function updateHotwordsDisplay() {
            const list = document.getElementById('hotwords-list');
            const data = document.getElementById('hotwords-data');
            
            list.innerHTML = hotwords.map(word => 
                `<span class="hotword-item">${word}<span class="hotword-remove" onclick="removeHotword('${word}')">&times;</span></span>`
            ).join('');
            
            data.value = JSON.stringify(hotwords);
        }
        
        // 支持回车键添加热词
        document.getElementById('hotword-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                addHotword();
            }
        });
    </script>
</body>
</html>
        ''')
        
    # Install necessary dependencies
    try:
        import pydub
    except ImportError:
        os.system('pip install pydub')
        
    # Create transcribe.html template
    with open('templates/transcribe.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Whisper Language Assistant - Transcribing</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .btn {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-right: 10px;
            text-decoration: none;
            display: inline-block;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
        }
        .alert-danger {
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .alert-success {
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .progress {
            height: 20px;
            background-color: #f5f5f5;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-bar {
            height: 100%;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            line-height: 20px;
            transition: width 0.5s ease;
        }
        #status {
            font-weight: bold;
            margin-bottom: 10px;
        }
        #controls {
            margin-top: 20px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Whisper Language Assistant</h1>
        <div>
            <a href="{{ url_for('index') }}" class="btn">Back</a>
            <a href="{{ url_for('logout') }}" class="btn">Logout</a>
        </div>
    </div>
    
    <h2>Transcribing Audio</h2>
    <div id="status">Initializing transcription...</div>
    
    <div class="progress">
        <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%;" 
             aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
    </div>
    
    <div class="result">
        <h3>Transcription Result:</h3>
        <div id="transcript"></div>
    </div>
    
    <div id="controls">
        <a href="{{ url_for('index') }}" class="btn">Upload New File</a>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const progressBar = document.getElementById('progress-bar');
            const transcript = document.getElementById('transcript');
            const status = document.getElementById('status');
            const controls = document.getElementById('controls');
            
            // 连接到事件流
            const eventSource = new EventSource("{{ url_for('stream') }}");
            
            // 初始化
            eventSource.onopen = function() {
                status.textContent = "Connected, waiting for transcription to begin...";
            };
            
            // 处理消息
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                switch(data.type) {
                    case "init":
                        status.textContent = `Processing audio file (0/${data.total_segments} segments)`;
                        break;
                        
                    case "progress":
                        progressBar.style.width = data.progress + "%";
                        progressBar.textContent = data.progress + "%";
                        progressBar.setAttribute('aria-valuenow', data.progress);
                        status.textContent = `Processing audio file (${data.current_segment}/${data.total_segments} segments)`;
                        transcript.textContent = data.transcript;
                        break;
                        
                    case "complete":
                        progressBar.style.width = "100%";
                        progressBar.textContent = "100%";
                        progressBar.setAttribute('aria-valuenow', 100);
                        status.textContent = "Transcription complete!";
                        transcript.textContent = data.transcript;
                        controls.style.display = "block";
                        eventSource.close();
                        break;
                        
                    case "error":
                        status.textContent = "Error: " + data.message;
                        status.style.color = "#721c24";
                        eventSource.close();
                        controls.style.display = "block";
                        break;
                }
            };
            
            // 错误处理
            eventSource.onerror = function() {
                status.textContent = "Connection error. Please try again.";
                status.style.color = "#721c24";
                eventSource.close();
                controls.style.display = "block";
            };
        });
    </script>
</body>
</html>
    ''')

def predict_with_hotwords(predictor, samples, hotwords_config):
    """使用热词配置进行预测"""
    method = hotwords_config.get('method', 'prompt_injection')
    words = hotwords_config.get('words', [])
    
    if not words:
        # 没有热词，使用标准预测
        return predictor.predict(samples)
    
    if method == 'prompt_injection':
        return predict_with_prompt_injection(predictor, samples, words)
    elif method == 'logit_bias':
        return predict_with_logit_bias(predictor, samples, words, hotwords_config.get('boost_factor', 1.5))
    else:
        # 默认使用标准预测
        return predictor.predict(samples)

def predict_with_prompt_injection(predictor, samples, hotwords):
    """使用Prompt注入方法"""
    try:
        # 构建包含热词的提示
        prompt = f"以下音频可能包含这些词汇: {', '.join(hotwords)}。请准确转录音频内容。"
        
        # 创建包含prompt的请求数据
        request_data = {
            'audio': samples.tolist(),
            'initial_prompt': prompt
        }
        
        # 使用自定义序列化器发送请求
        response = predictor.predict(request_data)
        return response
    except Exception as e:
        app.logger.warning(f"Prompt注入失败，回退到标准预测: {str(e)}")
        return predictor.predict(samples)

def predict_with_logit_bias(predictor, samples, hotwords, boost_factor):
    """使用Logit Bias方法"""
    try:
        # 构建logit bias配置
        # 注意：这需要根据实际的SageMaker端点实现来调整
        logit_bias = {word: boost_factor for word in hotwords}
        
        request_data = {
            'audio': samples.tolist(),
            'logit_bias': logit_bias
        }
        
        response = predictor.predict(request_data)
        return response
    except Exception as e:
        app.logger.warning(f"Logit Bias失败，回退到标准预测: {str(e)}")
        return predictor.predict(samples)

# 添加热词配置API端点
@app.route('/api/hotwords', methods=['GET', 'POST'])
@login_required
def api_hotwords():
    """热词配置API"""
    if request.method == 'GET':
        # 返回当前会话的热词配置
        config = session.get('hotwords_config', {'method': 'prompt_injection', 'words': [], 'boost_factor': 1.5})
        return jsonify(config)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            method = data.get('method', 'prompt_injection')
            words = data.get('words', [])
            boost_factor = data.get('boost_factor', 1.5)
            
            # 验证方法
            if method not in ['prompt_injection', 'logit_bias']:
                return jsonify({'error': 'Invalid method. Use prompt_injection or logit_bias'}), 400
            
            # 验证热词列表
            if not isinstance(words, list):
                return jsonify({'error': 'Words must be a list'}), 400
            
            # 保存到会话
            session['hotwords_config'] = {
                'method': method,
                'words': words,
                'boost_factor': boost_factor
            }
            session.modified = True
            
            return jsonify({'success': True, 'config': session['hotwords_config']})
            
        except Exception as e:
            app.logger.error(f"处理热词配置API时出错: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
