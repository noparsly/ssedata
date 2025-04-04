[1mdiff --git a/test.py b/test.py[m
[1mindex e69de29..9276302 100644[m
[1m--- a/test.py[m
[1m+++ b/test.py[m
[36m@@ -0,0 +1,150 @@[m
[32m+[m[32m# 这是一个AI学习项目的主文件[m
[32m+[m[32m# 创建于2024年4月2日[m
[32m+[m
[32m+[m[32mimport os[m
[32m+[m[32mfrom typing import List, Dict, Any[m
[32m+[m[32mfrom langchain.llms import OpenAI[m
[32m+[m[32mfrom langchain.chat_models import ChatOpenAI[m
[32m+[m[32mfrom langchain.agents import load_tools, initialize_agent, AgentType[m
[32m+[m[32mfrom langchain.memory import ConversationBufferMemory[m
[32m+[m[32mfrom langchain.chains import LLMChain[m
[32m+[m[32mfrom langchain.prompts import PromptTemplate[m
[32m+[m[32mfrom langchain.tools import DuckDuckGoSearchRun[m
[32m+[m[32mfrom langchain.document_loaders import PyPDFLoader, TextLoader[m
[32m+[m[32mfrom langchain.text_splitter import CharacterTextSplitter[m
[32m+[m[32mfrom langchain.embeddings import OpenAIEmbeddings[m
[32m+[m[32mfrom langchain.vectorstores import FAISS[m
[32m+[m[32mfrom langchain.chains import RetrievalQA[m
[32m+[m[32mfrom dotenv import load_dotenv[m
[32m+[m
[32m+[m[32m# 加载环境变量[m
[32m+[m[32mload_dotenv()[m
[32m+[m
[32m+[m[32m# 设置OpenAI API密钥[m
[32m+[m[32mos.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")[m
[32m+[m
[32m+[m[32mdef basic_llm_example() -> None:[m
[32m+[m[32m    """[m
[32m+[m[32m    基础LLM调用示例[m
[32m+[m[32m    """[m
[32m+[m[32m    # 初始化OpenAI模型[m
[32m+[m[32m    llm = OpenAI(temperature=0.7)[m
[32m+[m[41m    [m
[32m+[m[32m    # 直接调用模型[m
[32m+[m[32m    response = llm("解释一下什么是人工智能")[m
[32m+[m[32m    print("基础LLM回答:", response)[m
[32m+[m
[32m+[m[32mdef chat_model_example() -> None:[m
[32m+[m[32m    """[m
[32m+[m[32m    聊天模型示例[m
[32m+[m[32m    """[m
[32m+[m[32m    # 初始化ChatOpenAI模型[m
[32m+[m[32m    chat_model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")[m
[32m+[m[41m    [m
[32m+[m[32m    # 创建提示模板[m
[32m+[m[32m    template = """[m
[32m+[m[32m    你是一个友好的AI助手。[m
[32m+[m[32m    请用简单易懂的语言回答以下问题:[m
[32m+[m[32m    {question}[m
[32m+[m[32m    """[m
[32m+[m[41m    [m
[32m+[m[32m    prompt = PromptTemplate(template=template, input_variables=["question"])[m
[32m+[m[41m    [m
[32m+[m[32m    # 创建LLM链[m
[32m+[m[32m    chain = LLMChain(llm=chat_model, prompt=prompt)[m
[32m+[m[41m    [m
[32m+[m[32m    # 运行链[m
[32m+[m[32m    response = chain.run("什么是机器学习？请用初中生能理解的方式解释")[m
[32m+[m[32m    print("聊天模型回答:", response)[m
[32m+[m
[32m+[m[32mdef agent_with_tools_example() -> None:[m
[32m+[m[32m    """[m
[32m+[m[32m    使用工具的Agent示例[m
[32m+[m[32m    """[m
[32m+[m[32m    # 加载工具[m
[32m+[m[32m    tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))[m
[32m+[m[41m    [m
[32m+[m[32m    # 添加搜索工具[m
[32m+[m[32m    search = DuckDuckGoSearchRun()[m
[32m+[m[32m    tools.append(search)[m
[32m+[m[41m    [m
[32m+[m[32m    # 初始化记忆[m
[32m+[m[32m    memory = ConversationBufferMemory(memory_key="chat_history")[m
[32m+[m[41m    [m
[32m+[m[32m    # 初始化Agent[m
[32m+[m[32m    agent = initialize_agent([m
[32m+[m[32m        tools,[m[41m [m
[32m+[m[32m        OpenAI(temperature=0),[m[41m [m
[32m+[m[32m        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,[m
[32m+[m[32m        verbose=True,[m
[32m+[m[32m        memory=memory[m
[32m+[m[32m    )[m
[32m+[m[41m    [m
[32m+[m[32m    # 运行Agent[m
[32m+[m[32m    response = agent.run("北京的天气怎么样？然后计算23乘以45等于多少？")[m
[32m+[m[32m    print("Agent回答:", response)[m
[32m+[m
[32m+[m[32mdef document_qa_example(file_path: str, query: str) -> str:[m
[32m+[m[32m    """[m
[32m+[m[32m    文档问答示例[m
[32m+[m[41m    [m
[32m+[m[32m    Args:[m
[32m+[m[32m        file_path: 文档路径[m
[32m+[m[32m        query: 查询问题[m
[32m+[m[41m        [m
[32m+[m[32m    Returns:[m
[32m+[m[32m        str: 回答[m
[32m+[m[32m    """[m
[32m+[m[32m    # 根据文件类型选择加载器[m
[32m+[m[32m    if file_path.endswith('.pdf'):[m
[32m+[m[32m        loader = PyPDFLoader(file_path)[m
[32m+[m[32m    else:[m
[32m+[m[32m        loader = TextLoader(file_path)[m
[32m+[m[41m    [m
[32m+[m[32m    # 加载文档[m
[32m+[m[32m    documents = loader.load()[m
[32m+[m[41m    [m
[32m+[m[32m    # 分割文档[m
[32m+[m[32m    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)[m
[32m+[m[32m    texts = text_splitter.split_documents(documents)[m
[32m+[m[41m    [m
[32m+[m[32m    # 创建嵌入[m
[32m+[m[32m    embeddings = OpenAIEmbeddings()[m
[32m+[m[41m    [m
[32m+[m[32m    # 创建向量存储[m
[32m+[m[32m    db = FAISS.from_documents(texts, embeddings)[m
[32m+[m[41m    [m
[32m+[m[32m    # 创建检索QA链[m
[32m+[m[32m    qa = RetrievalQA.from_chain_type([m
[32m+[m[32m        llm=OpenAI(),[m
[32m+[m[32m        chain_type="stuff",[m
[32m+[m[32m        retriever=db.as_retriever()[m
[32m+[m[32m    )[m
[32m+[m[41m    [m
[32m+[m[32m    # 运行查询[m
[32m+[m[32m    result = qa.run(query)[m
[32m+[m[32m    return result[m
[32m+[m
[32m+[m[32mdef main() -> None:[m
[32m+[m[32m    """[m
[32m+[m[32m    主函数，运行所有示例[m
[32m+[m[32m    """[m
[32m+[m[32m    print("=== 运行基础LLM示例 ===")[m
[32m+[m[32m    basic_llm_example()[m
[32m+[m[41m    [m
[32m+[m[32m    print("\n=== 运行聊天模型示例 ===")[m
[32m+[m[32m    chat_model_example()[m
[32m+[m[41m    [m
[32m+[m[32m    print("\n=== 运行Agent工具示例 ===")[m
[32m+[m[32m    try:[m
[32m+[m[32m        agent_with_tools_example()[m
[32m+[m[32m    except Exception as e:[m
[32m+[m[32m        print(f"Agent示例运行错误: {e}")[m
[32m+[m[32m        print("注意: 运行Agent示例需要设置SERPAPI_API_KEY环境变量")[m
[32m+[m[41m    [m
[32m+[m[32m    print("\n=== 文档问答示例 ===")[m
[32m+[m[32m    print("注意: 此示例需要提供文档路径才能运行")[m
[32m+[m[32m    # document_qa_example("your_document.pdf", "文档中的主要观点是什么？")[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    main()[m
