import os
import warnings

warnings.filterwarnings("ignore")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from tavily import TavilyClient

load_dotenv()

# ================= 1. 构建本地论文知识库 (修改为容错模式) =================
print("1. 正在检查本地知识库...")
file_path = "./paper.pdf"
retriever = None  # 默认为空

# 🌟 修改点：如果有 PDF，就建库；如果没有，就跳过，不退出程序！
if os.path.exists(file_path):
    print("  -> 发现本地 PDF，正在构建向量知识库...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    print("  -> 本地知识库构建完成！")
else:
    print("  -> ⚠️ 未找到本地 PDF 文件，将启用【纯联网搜索模式】。")


# ================= 2. 定义工具集 (Tools) =================
@tool
def search_local_paper(query: str) -> str:
    """用于查询本地导入的论文PDF内容。如果用户询问关于这篇论文的具体细节、研究方法、实验数据时，请调用此工具。"""
    if retriever is None:
        return "当前没有上传任何本地 PDF 文档，无法进行本地检索，请直接使用联网搜索。"

    print(f"\n[系统提示: 🤖 AI 决定调用工具 -> 检索本地论文 PDF...]")
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search_web(query: str) -> str:
    """用于在互联网上搜索最新资讯、外部知识。如果用户询问论文之外的背景知识、最新新闻，或者本地查不到时，请调用此工具。"""
    print(f"\n[系统提示: 🤖 AI 决定调用工具 -> 联网搜索...]")
    try:
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        web_context = "\n\n".join([f"标题: {r['title']}\n内容: {r['content']}" for r in results])
        return web_context
    except Exception as e:
        return f"网页搜索失败: {str(e)}"


tools = [search_local_paper, search_web]

# ================= 3. 初始化 Agent =================
print("2. 正在初始化 Agent 大模型...")
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0
)

system_prompt = (
    "你是一个专业的学术助手。你可以调用工具来回答用户的问题。\n"
    "如果你需要了解这篇论文的具体细节，请调用 search_local_paper 工具。\n"
    "如果你需要查最新的新闻、外部资料，或者本地查不到，请调用 search_web 工具。\n"
    "请尽量基于你查到的内容回答问题，不要编造。"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ================= 4. 交互式问答 =================
print("\n✅ 学术助手已就绪！输入 '退出' 结束。")
chat_history = []

while True:
    query = input("\n🙋 你: ")
    if query == "退出":
        break

    result = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })

    print(f"\n💡 AI: {result['output']}")
    print("-" * 50)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result['output']))