# 堰子 - 情感向导 AI Agent

一个基于 FastAPI + Ollama + Redis + Qdrant 的智能情感向导 AI Agent。

## ✨ 特点

- 🧠 **本地大模型**：使用 Ollama 运行 qwen2:7b，完全免费，无需 API
- 🎤 **语音交互**：Whisper 语音识别 + Edge TTS 语音合成
- 💾 **持久化记忆**：Redis 存储对话历史，服务器重启不丢失
- 📚 **知识库学习**：输入 URL 自动学习并向量化存储到 Qdrant
- 🎭 **情绪识别**：自动识别用户情绪，切换回复风格
- 🔧 **工具扩展**：搜索、八字、解梦等工具集成
- 💬 **会话隔离**：session_id 隔离不同用户/会话

## 🎯 人设

**堰子** - 25岁情感向导

- 大学学心理学，擅长分析感情、人际关系
- 性格开朗、爱笑、正能量
- 说话幽默接地气，像朋友一样聊天
- 喜欢分享美食、睡觉等日常

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| Web框架 | FastAPI + Uvicorn |
| 大模型 | Ollama (qwen2:7b) |
| 向量数据库 | Qdrant |
| 缓存/记忆 | Redis |
| 语音识别 | Whisper (GPU加速) |
| 语音合成 | Edge TTS |
| 工具集成 | SerpAPI、缘分居API |

## 📦 安装

```bash
# 克隆项目
git clone https://gitee.com/y_h_r/yanzi-agent.git
cd yanzi-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 Ollama 并下载模型
ollama pull qwen2:7b

启动
python server.py

访问 http://127.0.0.1:8000/chat

yanzi-agent/
├── server.py          # FastAPI 主程序
├── Mytools.py         # Agent 工具集
├── requirements.txt   # Python 依赖
├── .env.example       # 环境变量示例
├── .gitignore         # Git 忽略配置
├── README.md          # 项目说明
└── static/            # 静态文件


📡 API 接口
接口	           说明
/	              查看可用接口
/docs	          Swagger API文档
/chat	          交互界面
/part1	          聊天接口
/voice_to_text	  语音转文字
/text_to_speech	  文字转语音
/sessions	      查看所有会话
/ws/audio	      WebSocket音频聊天

📄 许可证
MIT License

👤 作者
    作者：y_h_r
    邮箱：1544260628@qq.com















