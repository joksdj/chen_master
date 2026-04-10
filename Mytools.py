from langchain_community.utilities import SerpAPIWrapper
from langchain_classic.tools import tool
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()

# ========== API 配置 ==========
YUANFENJU_API_KEY = os.getenv("YUANFENJU_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# ========== 使用本地 Ollama 模型 ==========
def get_chat_model(temperature: float = 0.7):
    """使用本地 Ollama 模型"""
    return ChatOllama(
        model="qwen2:7b",
        temperature=temperature,
        base_url="http://localhost:11434"
    )

# ========== 情绪识别 ==========
def detect_emotion(query: str, chat_model=None):
    """检测用户情绪 - 情感向导专用"""
    query_lower = query.lower()
    if any(w in query_lower for w in ['难过', '伤心', '哭了', '分手', '失恋', '难受', '委屈']):
        return "depressed"
    if any(w in query_lower for w in ['气死', '愤怒', '可恶', '讨厌', '渣男', '渣女']):
        return "angry"
    if any(w in query_lower for w in ['开心', '高兴', '哈哈', '耶', '脱单', '恋爱了']):
        return "upbeat"
    if any(w in query_lower for w in ['怎么办', '迷茫', '不知道', '纠结', '犹豫']):
        return "friendly"
    if any(w in query_lower for w in ['加油', '努力', '美好', '希望']):
        return "cheerful"
    return "default"

# ========== 1. 搜索工具 ==========
def search(query: str):
    """搜索功能 - 使用 SerpAPI"""
    try:
        serp = SerpAPIWrapper()
        result = serp.run(query)
        print(f"🔍 搜索: {query}")
        print(f"📄 结果: {result[:200]}...")
        return result
    except Exception as e:
        print(f"❌ 搜索错误: {e}")
        return f"搜索失败: {str(e)}"

@tool
def search_tool(query: str):
    """搜索实时信息，当需要了解最新资讯、新闻、天气等时使用"""
    return search(query)

# ========== 2. 本地知识库工具 ==========
@tool
def get_info_from_local_db(query: str):
    """查询本地知识库中的情感建议、心理知识相关内容"""
    try:
        qdrant_path = Path("./qdrant_data")
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client = Qdrant.from_documents(
            documents=[],
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
            path=str(qdrant_path),
            collection_name="emotion_knowledge"
        )
        retriever = client.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        result = retriever.get_relevant_documents(query)
        if result:
            return "\n".join([doc.page_content for doc in result])
        return "暂无相关知识"
    except Exception as e:
        return f"知识库查询失败: {str(e)}"

# ========== 3. 八字排盘工具（保留但提示不擅长）==========
@tool
def bazi_cesuan(query: str):
    """八字排盘（不太擅长，会告诉用户更擅长情感咨询）"""
    return "哈哈，我不太懂算命哦～要不我们聊聊你最近的心情？"

# ========== 4. 摇卦工具 ==========
@tool
def yaoyigua():
    """摇卦占卜（不太擅长，会引导用户聊情感话题）"""
    return "占卜我不太会，但我们可以一起分析分析你的处境～"

# ========== 5. 解梦工具 ==========
@tool
def jiemeng(query: str):
    """周公解梦（会从心理学角度分析梦境）"""
    return f"梦见{query}啊...比起解梦，我更好奇你最近是不是压力有点大？"

# ========== 工具列表 ==========
tools = [
    search_tool,
    get_info_from_local_db,
    bazi_cesuan,
    yaoyigua,
    jiemeng
]