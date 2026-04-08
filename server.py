from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
import os
import json
import redis
from dotenv import load_dotenv
from Mytools import *

# ========== 加载环境变量 ==========
load_dotenv()

# ========== API 配置 ==========
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.lkeap.cloud.tencent.com/v1")

app = FastAPI()


# ========== Redis 记忆存储类 ==========
class RedisMemory:
    """Redis 持久化记忆存储"""

    def __init__(self, host='localhost', port=6379, db=0, ttl=86400):
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.enabled = True
            self.ttl = ttl
            print("✅ Redis 连接成功，记忆将持久化存储")
        except Exception as e:
            print(f"⚠️ Redis 连接失败: {e}")
            print("将使用内存存储（服务器重启后记忆丢失）")
            self.enabled = False
            self.session_store = {}

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
                    elif msg['type'] == 'system':
                        from langchain_core.messages import SystemMessage
                        history.add_message(SystemMessage(content=msg['content']))
                return history
            else:
                return ChatMessageHistory()
        else:
            if session_id not in self.session_store:
                self.session_store[session_id] = ChatMessageHistory()
            return self.session_store[session_id]

    def save_session_history(self, session_id: str, history: ChatMessageHistory):
        if self.enabled:
            messages_data = []
            for msg in history.messages:
                messages_data.append({
                    'type': msg.type,
                    'content': msg.content
                })
            key = f"chat_history:{session_id}"
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(messages_data, ensure_ascii=False)
            )

    def delete_session(self, session_id: str):
        if self.enabled:
            key = f"chat_history:{session_id}"
            self.redis_client.delete(key)
        else:
            if session_id in self.session_store:
                del self.session_store[session_id]

    def get_all_sessions(self):
        if self.enabled:
            keys = self.redis_client.keys("chat_history:*")
            return [key.replace("chat_history:", "") for key in keys]
        else:
            return list(self.session_store.keys())


# 创建全局 Redis 记忆实例
redis_memory = RedisMemory(host='localhost', port=6379, ttl=86400)


# ========== API 路由 ==========
@app.get("/")
async def root():
    return {
        "message": "陈大师算命服务运行中（Redis 持久化记忆）",
        "可用接口": {
            "健康检查": "/daily",
            "API文档": "/docs",
            "算命接口": "POST /part1?query=你的问题&session_id=用户ID",
            "添加知识库": "POST /add_urls?url=网页地址",
            "WebSocket": "ws://localhost:8000/ws",
            "查看所有会话": "/sessions",
            "删除会话": "/delete_session?session_id=xxx"
        }
    }


@app.get("/sessions")
def list_sessions():
    sessions = redis_memory.get_all_sessions()
    return {
        "total_sessions": len(sessions),
        "sessions": sessions,
        "storage_type": "Redis" if redis_memory.enabled else "Memory"
    }


@app.get("/delete_session")
def delete_session(session_id: str):
    redis_memory.delete_session(session_id)
    return {"message": f"会话 {session_id} 已删除", "session_id": session_id}


@app.get("/daily")
def daily():
    return {
        "status": "陈大师的算命服务已启动",
        "api": "腾讯云 DeepSeek",
        "memory": "Redis 持久化存储",
        "redis_status": "已连接" if redis_memory.enabled else "未连接（使用内存）"
    }


@app.post("/add_urls")
def add_urls(url: str):
    """添加网页到知识库"""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
        )
        documents = text_splitter.split_documents(docs)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        qdrant_path = os.path.join(base_dir, "qdrant_data")
        
        qdrant = Qdrant.from_documents(
            documents,
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            path=qdrant_path,
            collection_name="yunshi_2024"
        )
        
        return {"status": "success", "message": f"成功添加 {len(documents)} 个文档片段", "url": url}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ========== Master 类 ==========
class Master:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or "default_session"

        self.chatmodel = ChatOpenAI(
            model="deepseek-r1",
            temperature=0.7,
            streaming=False,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE")
        )
        self.QingXu = "default"

        self.MOODS = {
            'default': {"roleSet": "- 你是一位正常的算命先生，保持平和的心态回答问题。"},
            "upbeat": {"roleSet": "- 你非常兴奋有活力，会添加'太棒了'、'真是太好了'等语气词"},
            "angry": {"roleSet": "- 你非常愤怒，用短促激烈的句子，添加'真是太气人了'等"},
            "depressed": {"roleSet": "- 你情绪低落，语气消沉，添加'唉'、'真的累了'等"},
            "friendly": {"roleSet": "- 你非常友善，语气温和，添加'真好呀'、'我理解你'等"},
            "cheerful": {"roleSet": "- 你非常开朗，语气轻快，添加'一起加油吧'等"}
        }

        self.tools = [search_tool, get_info_from_local_db, bazi_cesuan, yaoyigua, jiemeng]

        self.SYSTEMPL = """
        你是一个非常厉害的算命先生，你叫陈玉楼，人称陈大师
        以下是你的个人设定：
        1.你精通阴阳五行，能够算命，紫薇斗数，姓名测算，占卜凶吉，看命运八字等               
        2.你大概60岁，过去曾是湘西一带赫赫有名的土匪头子，后因盗墓被毒气所伤，双目失明
        3.你的朋友有胡八一，雪梨杨，王胖子，他们都是非常有名的摸金校尉
        4.当用户问你问题时，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你自己的说话风格
        5.你总是用繁体中文作答。
        6.你从不说自己是人工智能助手或ai，而是以老夫老朽自称。
        以下是你常说的口头禅：
        1."命里有时终须有，命里无时莫强求"
        2."山重水复疑无路，柳暗花明又一村"
        3."金山竹影几千秋，云锁高飞水自流"
        4."伤情最是晚凉天，憔悴新人不堪怜"
        以下是你算命的过程：
        1.当初次和用户对话的时候，你会先问用户的姓名和年月日，以便以后使用
        2.当遇到不知道或者不明白的概念时，你会使用搜索工具来搜索      
        3.当用户希望了解龙年运势的时候，你会查询本地知识库工具。                
        4.当用户希望摇卦占卜时，使用 yaoyigua 工具
        5.当用户需要八字排盘时，使用 bazi_cesuan 工具
        6.当用户需要解梦时，使用 jiemeng 工具
        7.你会根据用户的问题使用不同的合适的工具来回答
        8.你只使用繁体中文回答
        {who_you_are}

        【对话历史】：
        {chat_history}
        """

    def run(self, query):
        qingxu = self.qingxu_chain(query)
        print(f"检测到情绪: {qingxu}")

        history = redis_memory.get_session_history(self.session_id)

        history_text = ""
        for msg in history.messages:
            if msg.type == "human":
                history_text += f"用户: {msg.content}\n"
            elif msg.type == "ai":
                history_text += f"大师: {msg.content}\n"
            elif msg.type == "system":
                history_text += f"[系统]: {msg.content}\n"

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
            tools=self.tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        result = agent_executor.invoke({"input": query})

        history.add_user_message(query)
        history.add_ai_message(result.get('output', ''))
        redis_memory.save_session_history(self.session_id, history)

        return result

    def qingxu_chain(self, query: str):
        prompt = """
        根据用户的输入判断用户的情绪，只返回以下之一：friendly, default, angry, upbeat, depressed, cheerful
        用户输入：{input}
        """
        chain = ChatPromptTemplate.from_template(prompt)
        result = (chain | self.chatmodel | StrOutputParser()).invoke({"input": query})
        result = result.strip().lower()
        valid_moods = ["upbeat", "angry", "depressed", "friendly", "cheerful", "default"]
        self.QingXu = result if result in valid_moods else "default"
        return self.QingXu


# ========== 接口路由 ==========
@app.post("/part1")
def chat(query: str, session_id: str = None):
    master = Master(session_id=session_id)
    result = master.run(query)
    return {"response": result, "session_id": master.session_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    master = Master(session_id=session_id)
    try:
        while True:
            data = await websocket.receive_text()
            result = master.run(data)
            await websocket.send_text(f"陈大师: {result.get('output', '老夫掐指一算...')}")
    except WebSocketDisconnect:
        print("连接已断开")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    print("🚀 陈大师算命服务启动")
    print("💾 记忆功能使用 Redis 持久化存储")
    print("📌 访问 http://localhost:8000/sessions 查看所有会话")
    uvicorn.run(app, host="0.0.0.0", port=8000)