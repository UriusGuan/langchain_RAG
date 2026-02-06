from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


class RAGSystem:
    def __init__(self, document_path="./document/轻量级.txt"):
        """
        初始化 RAG 系统

        Args:
            document_path: 知识库文档路径
        """
        self.document_path = document_path
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.conversation_history = []

        # 初始化组件
        self._init_models()
        self._init_vector_store()
        self._build_chain()

    def _init_models(self):
        """初始化模型"""
        # 设置大语言模型
        self.llm = ChatOpenAI(
            model="qwen3-0.6b",
            base_url=os.getenv('BASE_URL'),
            api_key=os.getenv('API_KEY'),
            extra_body={"enable_thinking": False},
            temperature=0.1  # 降低随机性，使回答更可靠
        )

        # 设置嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=r"E:\PycharmProjects\PythonProject\pretrained\BAAI\bge-large-zh-v1___5"
        )

    def _init_vector_store(self):
        """初始化向量数据库"""
        try:
            # 尝试加载现有的向量数据库
            if os.path.exists("./chroma_data") and os.listdir("./chroma_data"):
                print("加载现有向量数据库...")
                self.vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory="./chroma_data"
                )
            else:
                # 创建新的向量数据库
                print("创建新的向量数据库...")
                self._create_vector_store()

            # 初始化检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

        except Exception as e:
            print(f"初始化向量数据库时出错: {e}")
            # 重新创建向量数据库
            print("重新创建向量数据库...")
            self._create_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

    def _create_vector_store(self):
        """创建向量数据库"""
        # 加载文件
        loader = TextLoader(self.document_path, encoding="utf-8")
        data = loader.load()

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        docs = text_splitter.split_documents(data)

        # 创建向量数据库
        self.vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            persist_directory="./chroma_data"
        )

    def _build_chain(self):
        """构建 RAG 链"""
        # 提示词模板 (保持与原代码一致)
        conversation = [
            ("system",
             "你是一个严谨的RAG助手。请根据上下文信息回答问题(如果上下文信息不足以回答问题，请直接说'上下文信息不充分，无法回答'。): \n{context}"),
            ("user", "{query}")
        ]
        prompt = ChatPromptTemplate.from_messages(conversation)

        # 构建链 (保持与原代码一致)
        self.chain = (
                RunnableParallel({
                    "context": RunnablePassthrough() | self.retriever,
                    "query": RunnablePassthrough()
                })
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def query(self, question: str, include_context: bool = False):
        """
        查询 RAG 系统

        Args:
            question: 用户问题
            include_context: 是否返回检索到的上下文

        Returns:
            回答结果，可选包含上下文
        """
        try:
            # 执行查询
            result = self.chain.invoke(question)

            # 更新对话历史
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": result
            })

            # 限制对话历史长度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            # 如果不需要上下文，直接返回结果
            if not include_context:
                return result

            # 如果需要上下文，也返回检索到的文档
            retrieved_docs = self.retriever.invoke(question)
            return {
                "answer": result,
                "contexts": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }

        except Exception as e:
            error_msg = f"查询过程中出错: {str(e)}"
            return {"error": error_msg}

    def get_conversation_history(self):
        """获取对话历史"""
        return self.conversation_history

    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history = []
        return {"message": "对话历史已清空"}


# 创建全局 RAG 系统实例
rag_system = RAGSystem()
