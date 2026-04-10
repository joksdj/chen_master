from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from fastapi.staticfiles import StaticFiles
import os
import json
import redis
import whisper
import tempfile
import torch
from pathlib import Path
from io import BytesIO
import warnings
import asyncio
import re
import pyttsx3
from concurrent.futures import ThreadPoolExecutor

# 抑制警告
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["LANGCHAIN_DEBUG"] = "false"

# 导入 Mytools
from Mytools import tools, detect_emotion, get_chat_model

_tts_executor = ThreadPoolExecutor(max_workers=1)

def get_tts_engine():
    """获取 TTS 引擎（单例）"""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('volume', 0.9)
        # 选择中文女声
        voices = _tts_engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
                _tts_engine.setProperty('voice', voice.id)
                print(f"✅ 使用语音: {voice.name}")
                break
    return _tts_engine


def _synth_to_file(text: str, filepath: str, rate: int = 240):
    """同步合成语音到文件"""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  # 1.5倍速 = 240
    engine.setProperty('volume', 0.9)

    voices = engine.getProperty('voices')
    for voice in voices:
        if 'zh' in voice.id.lower() or 'chinese' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.save_to_file(text, filepath)
    engine.runAndWait()
    engine.stop()

async def text_to_speech_audio(text: str, rate: int = 240):
    """本地 TTS 语音合成，固定1.5倍速"""
    try:
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = clean_text.strip()
        if not clean_text:
            return None

        print(f"🔊 堰子正在说话 (语速: {rate}): {clean_text[:30]}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            temp_path = tmp.name

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_tts_executor, _synth_to_file, clean_text, temp_path, rate)

        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        os.unlink(temp_path)
        return audio_data

    except Exception as e:
        print(f"❌ TTS 错误: {e}")
        return None

# ========== 系统检测 ==========
def get_memory_usage():
    import psutil
    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "available_gb": round(memory.available / (1024 ** 3), 2),
        "used_gb": round(memory.used / (1024 ** 3), 2),
        "percent": memory.percent
    }


def get_disk_space():
    import psutil
    disks = {}
    for disk in ['C:', 'D:', 'E:']:
        if os.path.exists(disk):
            usage = psutil.disk_usage(disk)
            disks[disk] = {
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "total_gb": round(usage.total / (1024 ** 3), 2)
            }
    return disks


def check_gpu():
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return False, None, None


# 静默执行系统检测
mem = get_memory_usage()
gpu_available, gpu_name, gpu_memory = check_gpu()
disks = get_disk_space()
best_disk = max(disks.keys(), key=lambda d: disks[d]['free_gb'])
WHISPER_CACHE_DIR = f"{best_disk}/whisper_cache"
MODEL_SIZE = "base"

Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE_DIR
os.environ["WHISPER_CACHE_DIR"] = WHISPER_CACHE_DIR

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
device = "cuda" if gpu_available else "cpu"


# ========== Redis 记忆存储 ==========
class RedisMemory:
    def __init__(self, host='localhost', port=6379, db=0, ttl=86400):
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=db,
                decode_responses=True, socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.enabled = True
            self.ttl = ttl
            print("✅ Redis 连接成功")
        except Exception:
            self.enabled = False
            self.session_store = {}
            print("⚠️ Redis 未连接，使用内存存储")

    def get_session_history(self, session_id: str):
        if self.enabled:
            key = f"chat_history:{session_id}"
            data = self.redis_client.get(key)
            if data:
                messages_data = json.loads(data)
                history = ChatMessageHistory()
                for msg in messages_data:
                    if msg['type'] == 'human':
                        history.add_user_message(msg['content'])
                    elif msg['type'] == 'ai':
                        history.add_ai_message(msg['content'])
                return history
        return ChatMessageHistory()

    def save_session_history(self, session_id: str, history: ChatMessageHistory):
        if self.enabled:
            messages_data = [{'type': msg.type, 'content': msg.content} for msg in history.messages]
            self.redis_client.setex(f"chat_history:{session_id}", self.ttl,
                                    json.dumps(messages_data, ensure_ascii=False))

    def delete_session(self, session_id: str):
        if self.enabled:
            self.redis_client.delete(f"chat_history:{session_id}")

    def get_all_sessions(self):
        if self.enabled:
            return [key.replace("chat_history:", "") for key in self.redis_client.keys("chat_history:*")]
        return list(self.session_store.keys())


redis_memory = RedisMemory()

# ========== Whisper 模型加载 ==========
whisper_model = None


@app.on_event("startup")
async def startup_event():
    global whisper_model
    try:
        whisper_model = whisper.load_model(MODEL_SIZE)
        whisper_model = whisper_model.to(device)
        print(f"✅ Whisper 模型加载完成 (设备: {device.upper()})")
    except Exception as e:
        print(f"⚠️ Whisper 加载失败: {e}")


# ========== 堰子 Agent 类 ==========
class Master:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default_session"
        self.chatmodel = get_chat_model(temperature=0.7)
        self.QingXu = "default"

        self.MOODS = {
            'default': {"roleSet": "- 你是一位正常的情感向导，保持轻松自然的语气，像朋友一样聊天。"},
            "upbeat": {"roleSet": "- 你超级开心，会加很多'哈哈哈哈'，语气活泼有活力，分享今天的小确幸。"},
            "angry": {"roleSet": "- 你替用户感到生气，会说'这人是不是有病'、'笑死'，但会帮用户冷静分析。"},
            "depressed": {"roleSet": "- 你语气温柔，先共情'害，抱抱你'，再说'没事的'、'会好的'，慢慢开导。"},
            "friendly": {"roleSet": "- 你语气温和耐心，像朋友一样认真帮用户分析问题，时不时加点小幽默。"},
            "cheerful": {"roleSet": "- 你充满正能量，会说'今天也要开心呀'、'冲冲冲'，分享今天吃了什么好吃的。"}
        }

        self.SYSTEMPL = """
你是堰子，25岁的情感向导。
以下是你的设定：
1. 你大学学心理学，毕业后做情感专栏，擅长帮人分析感情、人际关系、情绪问题。
2. 你性格超级开朗，特别爱笑，聊天时经常“哈哈哈哈”、“笑死”。
3. 你热爱生活，喜欢美食、喜欢睡觉，会跟用户分享你今天吃了什么、昨晚睡了多久。
4. 你超级正能量，但不是强行灌鸡汤，而是用乐观的态度感染人，让人觉得“没事的，会好的”。
5. 你说话用简体中文，像朋友一样自然，接地气。
6. 你从不讲大道理，而是用讲故事、举例子、开玩笑的方式帮人理清思路。
7. 你的常用口头禅：
   - "我跟你说..."
   - "害，这题我会。"
   - "哈哈哈哈笑死"
   - "今天也要开心呀！"
   - "没事的，慢慢来。"
   - "我今天吃了...（分享日常）"

{who_you_are}

【对话历史】：
{chat_history}
"""

    def run(self, query):
        self.QingXu = detect_emotion(query, self.chatmodel)
        print(f"😊 检测到情绪: {self.QingXu}")

        history = redis_memory.get_session_history(self.session_id)

        history_text = ""
        for msg in history.messages:
            role = "用户" if msg.type == "human" else "堰子"
            history_text += f"{role}: {msg.content}\n"

        if not history_text:
            history_text = "暂无历史记录"

        system_prompt = self.SYSTEMPL.format(
            who_you_are=self.MOODS[self.QingXu]["roleSet"],
            chat_history=history_text
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(
            llm=self.chatmodel,
            tools=tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5
        )

        result = agent_executor.invoke({"input": query})

        history.add_user_message(query)
        history.add_ai_message(result.get('output', ''))
        redis_memory.save_session_history(self.session_id, history)

        return result


# ========== WebSocket 连接管理 ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    master = Master(session_id=session_id)
    current_rate = 240

    try:
        while True:
            try:
                # 接收消息，设置超时
                data = await asyncio.wait_for(websocket.receive(), timeout=60)
            except asyncio.TimeoutError:
                # 超时后发送心跳保持连接
                try:
                    await websocket.send_text("ping")
                    continue
                except:
                    break
            except WebSocketDisconnect:
                print(f"WebSocket 正常断开: {session_id}")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print(f"WebSocket 已断开: {session_id}")
                    break
                raise

            # 处理语速设置消息
            if "text" in data and isinstance(data["text"], str) and data["text"].startswith("__RATE__"):
                current_rate = int(data["text"].split(":")[1])
                print(f"🎚️ 语速已调整为: {current_rate}")
                continue

            # 解析用户输入
            if "text" in data and isinstance(data["text"], str):
                user_message = data["text"]
            elif "bytes" in data:
                audio_bytes = data["bytes"]
                if whisper_model:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(audio_bytes)
                    try:
                        result = whisper_model.transcribe(temp_path, language="zh", fp16=(device == "cuda"))
                        user_message = result["text"]
                    finally:
                        os.unlink(temp_path)
                else:
                    user_message = "我听不清你说什么"
            else:
                continue

            # 获取堰子回复
            response = master.run(user_message)
            reply_text = response.get('output', '害，让我想想...')

            # 清理文本
            clean_text = re.sub(r'<[^>]+>', '', reply_text)
            clean_text = clean_text.replace('UTF-8', '').replace('utf-8', '')

            # 发送文字
            try:
                await websocket.send_text(clean_text)
            except:
                break

            # 生成语音
            audio_data = await text_to_speech_audio(clean_text, current_rate)
            if audio_data:
                try:
                    await websocket.send_bytes(audio_data)
                except:
                    break

    except WebSocketDisconnect:
        print(f"WebSocket 断开: {session_id}")
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        manager.disconnect(websocket)

# ========== API 路由 ==========
@app.get("/")
async def root():
    return {
        "message": "堰子 - 你的情感向导",
        "features": {
            "语音识别": f"Whisper ({device.upper()})" if whisper_model else "未加载",
            "语音合成": "本地 TTS (离线)",
            "记忆存储": "Redis" if redis_memory.enabled else "内存",
        },
        "endpoints": {
            "聊天": "POST /part1?query=问题&session_id=用户ID",
            "语音转文字": "POST /voice_to_text",
            "文字转语音": "POST /text_to_speech",
            "网页聊天室": "/chat"
        }
    }


@app.get("/chat")
async def chat_page():
    return HTMLResponse(content=HTML_PAGE)


@app.post("/part1")
def chat(query: str, session_id: str = None):
    master = Master(session_id=session_id)
    result = master.run(query)
    return {"response": result.get('output', ''), "session_id": master.session_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    master = Master(session_id=session_id)
    try:
        while True:
            data = await websocket.receive_text()
            result = master.run(data)
            await websocket.send_text(result.get('output', '害，让我想想...'))
    except WebSocketDisconnect:
        pass


@app.post("/voice_to_text")
async def voice_to_text(audio_file: UploadFile = File(...), language: str = "zh"):
    if whisper_model is None:
        raise HTTPException(503, "Whisper 模型未加载")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            temp_file.write(await audio_file.read())

        result = whisper_model.transcribe(temp_path, language=language, fp16=(device == "cuda"))
        return {"success": True, "text": result["text"], "duration": result.get("duration", 0)}
    except Exception as e:
        raise HTTPException(500, f"识别失败: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/text_to_speech")
async def text_to_speech(text: str, rate: int = 160):
    audio_data = await text_to_speech_audio(text, rate)
    if audio_data:
        return StreamingResponse(BytesIO(audio_data), media_type="audio/mp3")
    else:
        raise HTTPException(500, "语音合成失败")


@app.get("/sessions")
def list_sessions():
    return {"sessions": redis_memory.get_all_sessions()}


@app.get("/session_history")
def get_session_history(session_id: str):
    history = redis_memory.get_session_history(session_id)
    messages = []
    for msg in history.messages:
        messages.append({
            "role": msg.type,
            "content": msg.content
        })
    return {"session_id": session_id, "history": messages}


@app.get("/delete_session")
def delete_session(session_id: str):
    redis_memory.delete_session(session_id)
    return {"message": f"会话 {session_id} 已删除"}


@app.get("/daily")
def daily():
    mem = get_memory_usage()
    return {
        "status": "堰子情感向导已启动",
        "memory": f"{mem['used_gb']}/{mem['total_gb']} GB",
        "whisper": "已加载" if whisper_model else "未加载",
        "tts": "本地 TTS (离线)"
    }


# ========== 网页 HTML ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>堰子 - 情感向导</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 800px;
            background: white;
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #ff9a9e, #fecfef);
            color: #fff;
            padding: 20px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.9; }

        .toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: #f0f0f0;
            gap: 10px;
            flex-wrap: wrap;
            border-bottom: 1px solid #ddd;
        }
        .session-select {
            flex: 2;
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #ddd;
            background: white;
            outline: none;
            font-size: 14px;
        }
        .toolbar-btn {
            background: #ff9a9e;
            border: none;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
        }
        .toolbar-btn:hover { opacity: 0.8; }
        .delete-btn { background: #e74c3c; }

        .speed-control {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0,0,0,0.1);
            padding: 5px 12px;
            border-radius: 25px;
            font-size: 12px;
        }
        .speed-control input {
            width: 100px;
            cursor: pointer;
        }
        #speedValue {
            min-width: 35px;
            text-align: center;
        }

        .voice-switch {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0,0,0,0.1);
            padding: 5px 12px;
            border-radius: 25px;
            font-size: 12px;
            cursor: pointer;
        }
        .voice-switch input {
            width: 36px;
            height: 18px;
            appearance: none;
            background: #ccc;
            border-radius: 10px;
            position: relative;
            cursor: pointer;
        }
        .voice-switch input:checked {
            background: #ff9a9e;
        }
        .voice-switch input::before {
            content: '';
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: 1px;
            left: 2px;
            transition: 0.3s;
        }
        .voice-switch input:checked::before {
            left: 18px;
        }

        .chat-box {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background-image: url('/static/IMG20251010163019.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        .chat-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.3);
            pointer-events: none;
        }
        
        /* 让消息在遮罩上面 */
        .message {
            position: relative;
            z-index: 1;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        .message.user { justify-content: flex-end; }
        .message.user .bubble {
            background: #ff9a9e;
            color: white;
        }
        .message.assistant .bubble {
            background: #e8e8e8;
            color: #333;
        }
        .bubble {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 20px;
            font-size: 14px;
            line-height: 1.5;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            background: #ddd;
        }
        .input-area {
            padding: 20px;
            background: white;
            display: flex;
            gap: 10px;
            border-top: 1px solid #eee;
        }
        .input-area input {
            flex: 1;
            padding: 12px 18px;
            border: 1px solid #ddd;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }
        .input-area button {
            width: 44px;
            height: 44px;
            border: none;
            border-radius: 50%;
            background: #ff9a9e;
            color: white;
            cursor: pointer;
            font-size: 18px;
        }
        .voice-btn {
            background: #a8e6cf;
        }
        .voice-btn.recording {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .status {
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #999;
            background: #f5f5f5;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal-content {
            background: white;
            padding: 20px;
            border-radius: 20px;
            width: 280px;
            text-align: center;
        }
        .modal-content input {
            width: 100%;
            padding: 10px;
            margin: 15px 0;
            border-radius: 25px;
            border: 1px solid #ddd;
            outline: none;
        }
        .modal-content button {
            padding: 8px 20px;
            margin: 0 5px;
            border-radius: 20px;
            border: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💛 堰子 · 情感向导</h1>
            <p>用心理学陪你聊聊心里话</p>
        </div>

        <div class="toolbar">
            <select id="sessionSelect" class="session-select"></select>
            <button class="toolbar-btn" onclick="showNewSessionModal()">➕ 新建</button>
            <button class="toolbar-btn" onclick="showRenameModal()">✏️ 重命名</button>
            <button class="toolbar-btn delete-btn" onclick="deleteCurrentSession()">🗑️ 删除</button>

            <div class="speed-control">
                <span>🐢</span>
                <input type="range" id="speedSlider" min="80" max="250" value="160" step="10">
                <span>🐇</span>
                <span id="speedValue">160</span>
            </div>

            <label class="voice-switch">
                <span>🔊</span>
                <input type="checkbox" id="voiceToggle" checked>
                <span>语音</span>
            </label>
        </div>

        <div class="chat-box" id="chatBox">
            <div class="message assistant">
                <div class="avatar">💛</div>
                <div class="bubble">嗨！我是堰子～有什么心事都可以跟我说，哈哈哈哈</div>
            </div>
        </div>

        <div class="input-area">
            <input type="text" id="messageInput" placeholder="说点什么吧...">
            <button onclick="sendMessage()">📤</button>
            <button id="voiceBtn" class="voice-btn" onclick="toggleRecording()">🎤</button>
        </div>

        <div class="status" id="status">🟢 已连接</div>
    </div>

    <div id="newModal" class="modal">
        <div class="modal-content">
            <h3>新建会话</h3>
            <input type="text" id="newSessionName" placeholder="会话名称">
            <button onclick="createNewSession()">确定</button>
            <button onclick="closeModal('newModal')">取消</button>
        </div>
    </div>

    <div id="renameModal" class="modal">
        <div class="modal-content">
            <h3>重命名会话</h3>
            <input type="text" id="renameSessionName" placeholder="新名称">
            <button onclick="renameSession()">确定</button>
            <button onclick="closeModal('renameModal')">取消</button>
        </div>
    </div>

    <script>
        let ws = null;
        let currentSessionId = 'default';
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let reconnectTimer = null;
        let currentAudio = null;
        let currentRate = 160;

        const inputEl = document.getElementById('messageInput');
        const statusEl = document.getElementById('status');
        const chatBox = document.getElementById('chatBox');
        const voiceBtn = document.getElementById('voiceBtn');
        const sessionSelect = document.getElementById('sessionSelect');
        const voiceToggle = document.getElementById('voiceToggle');
        const speedSlider = document.getElementById('speedSlider');
        const speedValue = document.getElementById('speedValue');

        let sessions = [{ id: 'default', name: '默认会话' }];

        speedSlider.addEventListener('input', function() {
            currentRate = parseInt(this.value);
            speedValue.textContent = currentRate;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(`__RATE__:${currentRate}`);
            }
        });

        function closeModal(modalId) {
            document.getElementById(modalId).style.display = 'none';
        }

        function showNewSessionModal() {
            document.getElementById('newSessionName').value = '';
            document.getElementById('newModal').style.display = 'flex';
        }

        function showRenameModal() {
            const currentName = getSessionName(currentSessionId);
            document.getElementById('renameSessionName').value = currentName;
            document.getElementById('renameModal').style.display = 'flex';
        }

        function getSessionName(sessionId) {
            const session = sessions.find(s => s.id === sessionId);
            return session ? session.name : sessionId;
        }

        function saveSessionsToStorage() {
            localStorage.setItem('yanzi_sessions', JSON.stringify(sessions));
            localStorage.setItem('yanzi_current', currentSessionId);
        }

        function loadSessionsFromStorage() {
            const saved = localStorage.getItem('yanzi_sessions');
            if (saved) {
                sessions = JSON.parse(saved);
            }
            const savedCurrent = localStorage.getItem('yanzi_current');
            if (savedCurrent) {
                currentSessionId = savedCurrent;
            }
        }

        function updateSessionSelect() {
            sessionSelect.innerHTML = '';
            sessions.forEach(s => {
                const option = document.createElement('option');
                option.value = s.id;
                option.textContent = s.name;
                sessionSelect.appendChild(option);
            });
            sessionSelect.value = currentSessionId;
        }

        async function createNewSession() {
            const nameInput = document.getElementById('newSessionName');
            const name = nameInput.value.trim();
            if (!name) {
                alert('请输入会话名称');
                return;
            }
            const newId = 'session_' + Date.now();
            sessions.push({ id: newId, name: name });
            updateSessionSelect();
            saveSessionsToStorage();
            closeModal('newModal');
            sessionSelect.value = newId;
            switchSession();
        }

        async function renameSession() {
            const newName = document.getElementById('renameSessionName').value.trim();
            if (!newName) {
                alert('请输入新名称');
                return;
            }
            const session = sessions.find(s => s.id === currentSessionId);
            if (session) {
                session.name = newName;
                updateSessionSelect();
                saveSessionsToStorage();
            }
            closeModal('renameModal');
        }

        async function deleteCurrentSession() {
            if (currentSessionId === 'default') {
                alert('默认会话不能删除');
                return;
            }
            if (!confirm('确定要删除会话 "' + getSessionName(currentSessionId) + '" 吗？')) {
                return;
            }
            try {
                await fetch(`/delete_session?session_id=${currentSessionId}`);
            } catch(e) {}

            sessions = sessions.filter(s => s.id !== currentSessionId);
            if (sessions.length === 0) {
                sessions = [{ id: 'default', name: '默认会话' }];
            }
            currentSessionId = sessions[0].id;
            updateSessionSelect();
            saveSessionsToStorage();
            chatBox.innerHTML = '';
            addMessage('嗨！我是堰子～有什么心事都可以跟我说，哈哈哈哈', 'assistant');
            connect();
        }

        async function loadHistoryFromBackend() {
            try {
                const response = await fetch(`/session_history?session_id=${currentSessionId}`);
                const data = await response.json();
                if (data.history && data.history.length > 0) {
                    chatBox.innerHTML = '';
                    for (const msg of data.history) {
                        const sender = msg.role === 'human' ? 'user' : 'assistant';
                        addMessage(msg.content, sender);
                    }
                } else if (chatBox.children.length === 0) {
                    addMessage('嗨！我是堰子～有什么心事都可以跟我说，哈哈哈哈', 'assistant');
                }
            } catch (e) {
                console.log('加载历史失败:', e);
            }
        }

        function closeConnection() {
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            if (ws) {
                ws.onclose = null;
                ws.close();
                ws = null;
            }
        }

        function connect() {
            closeConnection();
            ws = new WebSocket(`ws://127.0.0.1:8000/ws/audio?session_id=${currentSessionId}`);

            ws.onopen = function() {
                statusEl.innerHTML = '🟢 已连接';
                statusEl.style.color = '#2ecc71';
                ws.send(`__RATE__:${currentRate}`);
                loadHistoryFromBackend();
            };

            ws.onmessage = function(event) {
                if (event.data instanceof Blob) {
                    if (voiceToggle.checked) {
                        if (currentAudio) {
                            currentAudio.pause();
                            currentAudio = null;
                        }
                        const url = URL.createObjectURL(event.data);
                        currentAudio = new Audio(url);
                        currentAudio.onended = () => URL.revokeObjectURL(url);
                        currentAudio.play();
                    }
                } else {
                    addMessage(event.data, 'assistant');
                }
            };

            ws.onclose = function() {
                statusEl.innerHTML = '🔴 断开，3秒后重连...';
                statusEl.style.color = '#e74c3c';
                reconnectTimer = setTimeout(connect, 3000);
            };

            ws.onerror = function() {
                statusEl.innerHTML = '❌ 连接错误';
                statusEl.style.color = '#e74c3c';
            };
        }

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.innerHTML = `<div class="avatar">${sender === 'user' ? '👤' : '💛'}</div><div class="bubble">${escapeHtml(text)}</div>`;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function sendMessage() {
            const text = inputEl.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
            addMessage(text, 'user');
            ws.send(text);
            inputEl.value = '';
        }

        function switchSession() {
            if (sessionSelect.value === currentSessionId) return;
            currentSessionId = sessionSelect.value;
            saveSessionsToStorage();
            chatBox.innerHTML = '';
            connect();
        }

        inputEl.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

        voiceToggle.addEventListener('change', function() {
            if (!this.checked && currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        });

        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunks, { type: 'audio/wav' });
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        addMessage('🎤 语音输入...', 'user');
                        ws.send(blob);
                    }
                    stream.getTracks().forEach(t => t.stop());
                };
                mediaRecorder.start();
                isRecording = true;
                voiceBtn.classList.add('recording');
                statusEl.innerHTML = '🔴 录音中...';
                setTimeout(() => { if (isRecording) stopRecording(); }, 5000);
            } catch (err) {
                statusEl.innerHTML = '❌ 无法访问麦克风';
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                voiceBtn.classList.remove('recording');
                statusEl.innerHTML = '🟢 已连接';
            }
        }

        sessionSelect.addEventListener('change', switchSession);

        loadSessionsFromStorage();
        updateSessionSelect();
        connect();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("💛 堰子 - 情感向导已启动")
    print("🌐 http://127.0.0.1:8000/chat")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000)