
# chen-master - 陈大师智能算命服务

一个基于 FastAPI + Ollama + Redis + Qdrant 的智能算命 AI Agent，具备情绪感知、持久化记忆、知识库学习、多工具调用能力。

## 核心功能

| 功能 | 说明 |
|------|------|
| **情绪感知** | 自动识别用户情绪（开心/愤怒/悲伤/兴奋/友善），切换回答语气 |
| **持久化记忆** | Redis 存储对话历史，服务器重启不丢失，Session 隔离 |
| **知识库学习** | 输入 URL 自动学习网页内容，向量化存储到 Qdrant |
| **多工具调用** | 搜索、八字排盘、摇卦占卜、周公解梦 |
| **WebSocket 支持** | 实时对话，支持长连接 |
| **Session 隔离** | 不同用户独立记忆，互不干扰 |

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| Web框架 | FastAPI + Uvicorn | API服务器 |
| 大模型 | Ollama (Qwen2.5:7b) | 对话生成（完全本地运行） |
| 向量数据库 | Qdrant | 知识库存储 |
| 缓存/记忆 | Redis | 持久化对话历史 |
| 搜索 | SerpAPI | 实时信息检索 |
| 第三方API | 缘分居 | 八字、摇卦、解梦 |
| 文本处理 | LangChain | 文档分割、向量化 |

## 📦 安装教程

### 1. 克隆项目


git clone https://gitee.com/你的用户名/chen-master.git
cd chen-master


### 2. 创建虚拟环境


# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate


### 3. 安装依赖


pip install -r requirements.txt


### 4. 安装 Ollama
# 访问 https://ollama.com/download 下载安装
# 然后拉取模型
ollama pull qwen2.5:7b


### 5. 安装 Redis（可选）

`
# Windows: 下载 https://github.com/microsoftarchive/redis/releases
# 或使用 Docker
docker run -d -p 6379:6379 --name redis redis


### 6. 配置环境变量

cp .env.example .env
# 编辑 .env 文件，填入你的 API Key

### 7. 启动服务

python server.py

## 📡 API 接口列表

| 接口 | 方法 | 说明 | 示例 |
|------|------|------|------|
| `/` | GET | 查看可用接口 | `http://localhost:8000/` |
| `/docs` | GET | Swagger API 文档 | `http://localhost:8000/docs` |
| `/daily` | GET | 健康检查 | `http://localhost:8000/daily` |
| `/sessions` | GET | 查看所有会话 | `http://localhost:8000/sessions` |
| `/delete_session` | GET | 删除指定会话 | `?session_id=user_001` |
| `/part1` | POST | 算命接口 | `?query=你好&session_id=user_001` |
| `/add_urls` | POST | 学习网页内容 | `?url=https://xxx.com` |
| `/ws` | WebSocket | 实时对话 | `ws://localhost:8000/ws?session_id=user_001` |

## 🧠 情绪识别机制

| 用户输入示例 | 识别情绪 | 回应风格 |
|-------------|----------|----------|
| "太棒了！我中奖了！" | upbeat | 兴奋有活力，提醒乐极生悲 |
| "唉，最近好倒霉..." | depressed | 语气消沉，鼓励寻求支持 |
| "你算的不准！" | angry | 短促激烈，提醒冷静 |
| "你好，陈大师" | friendly | 温和友善 |
| "一起加油吧！" | cheerful | 开朗积极 |

## 🔧 可用工具

| 工具名称 | 触发条件 | 说明 |
|----------|----------|------|
| `search_tool` | 不知道的事情、实时数据 | SerpAPI 实时搜索 |
| `bazi_cesuan` | 用户提供姓名+生辰 | 八字排盘 |
| `yaoyigua` | 用户要求摇卦 | 占卜吉凶 |
| `jiemeng` | 用户描述梦境 | 周公解梦 |
| `get_info_from_local_db` | 龙年/2024年运势 | 知识库检索 |

## 📁 项目结构


chen-master/
├── server.py              # FastAPI 主程序
├── Mytools.py             # Agent 工具集
├── requirements.txt       # Python 依赖列表
├── .env.example          # 环境变量示例
├── .gitignore            # Git 忽略配置
├── README.md             # 项目说明
└── qdrant_data/          # Qdrant 本地数据（自动生成）


## ❓ 常见问题

### Q1: Redis 连接失败怎么办？
A: 如果不需要持久化记忆，代码会自动降级为内存存储。

### Q2: 如何更换大模型？
A: 修改 `server.py` 中的 `model` 参数，支持任何 Ollama 模型。

### Q3: API Key 填在哪里？
A: 复制 `.env.example` 为 `.env`，填入真实 Key。

## 📅 后续计划

- [ ] 修复 Qdrant Embeddings 问题
- [ ] 添加 Docker 一键部署
- [ ] 增加更多占卜工具
- [ ] 添加 Web 管理界面

## 🤝 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request

## 📄 许可证

MIT License

## 👤 作者

- 作者：y_h_r
- 邮箱：1544260628@qq.com


**注意**：本项目仅供学习交流使用，算命结果仅供参考。