# 📚 基于 RAG 与 Agent 的智能学术助手

本项目是一个基于 LangChain 和 DeepSeek 开发的大模型应用，旨在解决长篇幅 PDF 文献阅读效率低及大模型易产生“数据幻觉”的痛点。

## ✨ 核心特性
- **RAG 全链路搭建**：使用 BGE 模型与 ChromaDB 实现本地 PDF 的语义切片与向量化检索。
- **Agent 自主调度**：基于 Function Calling 机制，将“本地检索”与“Tavily 联网搜索”封装为 Tool，由大模型自主路由决策。
- **多轮对话记忆**：引入 MessagesPlaceholder 实现上下文记忆。
- **防幻觉设计**：强约束 System Prompt，并设计了无 PDF 时的纯联网降级模式。

## 🛠️ 技术栈
Python, LangChain, DeepSeek API, ChromaDB, Tavily API, HuggingFace (BGE)

## 🚀 如何运行
1. 克隆代码
2. 安装依赖：`pip install -r requirements.txt`
3. 配置 `.env` 文件（填入你的 DeepSeek 和 Tavily API Key）
4. 运行 `python rag_demo.py`
