from langchain_community.utilities import SerpAPIWrapper
from langchain_classic.tools import tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API 配置（从环境变量读取）
YUANFENJU_API_KEY = os.getenv("YUANFENJU_API_KEY", "")

# 腾讯云 DeepSeek 配置（从环境变量读取）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.lkeap.cloud.tencent.com/v1")


@tool
def get_info_from_local_db(query: str):
    """只有回答与2024年或者龙年运势相关的问题时，才用到的工具"""
    client = Qdrant(
        QdrantClient(path="local_qdrand"),
        "local_documents",
        OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


def search(query: str):
    """搜索工具 - 使用 SerpAPI"""
    try:
        serp = SerpAPIWrapper()
        result = serp.run(query)
        print(f"搜索: {query}")
        print(f"结果: {result[:200]}...")
        return result
    except Exception as e:
        print(f"搜索错误: {e}")
        return f"搜索失败: {str(e)}"


@tool
def search_tool(query: str):
    """需要了解实时信息或不知道的事情时使用这个工具"""
    return search(query)


@tool
def bazi_cesuan(query: str):
    """只有做八字排盘的时候使用，需要输入姓名，出生年月日，缺一不可"""
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/paipan"

    prompt = ChatPromptTemplate.from_template("""
        你是一个参数查询助手，能根据用户输入的内容占到相关的参数并按json格式返回。
        JSON字段如下：
        - "api_key": "{api_key}"
        - "name": "姓名"
        - "sex": "性别, 0表示男，1表示女，根据名字判断"
        - "type": "日期类型, 0农历, 1公历，默认1"
        - "year": "出生年份 例：1988"
        - "month": "出生月份 例 8"
        - "day": "出生日期，例 8"
        - "hours": "出生小时 例 14"
        - "minute": "0"

        如果没有找到相关参数，则需要提醒用户告诉你这些内容。
        只返回数据结构，不要有其他的评论。

        用户输入: {query}
    """)

    model = ChatOpenAI(
        model="deepseek-r1",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

    parser = JsonOutputParser()
    chain = prompt | model | parser

    try:
        data = chain.invoke({"query": query, "api_key": YUANFENJU_API_KEY})
        print("解析参数:", data)

        result = requests.post(url=url, data=data)
        if result.status_code == 200:
            print("八字结果:", result.json())
            try:
                json_data = result.json()
                return f"八字为: {json_data['data']['bazi_info']['bazi']}"
            except Exception as e:
                return "八字查询失败，可能是你忘记询问年月日了"
        else:
            return "技术错误，稍后重试"
    except Exception as e:
        return f"八字查询失败: {str(e)}"


@tool
def yaoyigua():
    """摇一卦工具 - 用于占卜吉凶"""
    api_key = YUANFENJU_API_KEY
    url = "https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
    result = requests.post(url, data={"api_key": api_key})
    if result.status_code == 200:
        print(result.json())
        return_string = json.loads(result.text)
        image = return_string["data"]["image"]
        print("卦图片:", image)
        return return_string
    else:
        return "技术错误，稍后重试"


@tool
def jiemeng(query: str):
    """只有想解梦的时候用，必须输入梦境内容来获得答案"""
    api_key = YUANFENJU_API_KEY
    url = "https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    
    llm = ChatOpenAI(
        model="deepseek-r1",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    
    prompt = PromptTemplate.from_template("根据内容提取关键词，只返回关键词内容: {topic}")
    chain = prompt | llm | StrOutputParser()
    keyword = chain.invoke({"topic": query})
    print("提取的关键字:", keyword)
    
    result = requests.post(url, data={"api_key": api_key, "title_zhougong": keyword})
    
    if result.status_code == 200:
        print(result.json())
        return_string = json.loads(result.text)
        return return_string
    else:
        return "技术错误，稍后重试"