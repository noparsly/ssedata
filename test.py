# 这是一个AI学习项目的主文件
# 创建于2024年4月2日

import os
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def basic_llm_example() -> None:
    """
    基础LLM调用示例
    """
    # 初始化OpenAI模型
    llm = OpenAI(temperature=0.7)
    
    # 直接调用模型
    response = llm("解释一下什么是人工智能")
    print("基础LLM回答:", response)

def chat_model_example() -> None:
    """
    聊天模型示例
    """
    # 初始化ChatOpenAI模型
    chat_model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
    
    # 创建提示模板
    template = """
    你是一个友好的AI助手。
    请用简单易懂的语言回答以下问题:
    {question}
    """
    
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    # 创建LLM链
    chain = LLMChain(llm=chat_model, prompt=prompt)
    
    # 运行链
    response = chain.run("什么是机器学习？请用初中生能理解的方式解释")
    print("聊天模型回答:", response)

def agent_with_tools_example() -> None:
    """
    使用工具的Agent示例
    """
    # 加载工具
    tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))
    
    # 添加搜索工具
    search = DuckDuckGoSearchRun()
    tools.append(search)
    
    # 初始化记忆
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # 初始化Agent
    agent = initialize_agent(
        tools, 
        OpenAI(temperature=0), 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )
    
    # 运行Agent
    response = agent.run("北京的天气怎么样？然后计算23乘以45等于多少？")
    print("Agent回答:", response)

def document_qa_example(file_path: str, query: str) -> str:
    """
    文档问答示例
    
    Args:
        file_path: 文档路径
        query: 查询问题
        
    Returns:
        str: 回答
    """
    # 根据文件类型选择加载器
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    # 加载文档
    documents = loader.load()
    
    # 分割文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # 创建嵌入
    embeddings = OpenAIEmbeddings()
    
    # 创建向量存储
    db = FAISS.from_documents(texts, embeddings)
    
    # 创建检索QA链
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    # 运行查询
    result = qa.run(query)
    return result

def main() -> None:
    """
    主函数，运行所有示例
    """
    print("=== 运行基础LLM示例 ===")
    basic_llm_example()
    
    print("\n=== 运行聊天模型示例 ===")
    chat_model_example()
    
    print("\n=== 运行Agent工具示例 ===")
    try:
        agent_with_tools_example()
    except Exception as e:
        print(f"Agent示例运行错误: {e}")
        print("注意: 运行Agent示例需要设置SERPAPI_API_KEY环境变量")
    
    print("\n=== 文档问答示例 ===")
    print("注意: 此示例需要提供文档路径才能运行")
    # document_qa_example("your_document.pdf", "文档中的主要观点是什么？")

if __name__ == "__main__":
    main()
