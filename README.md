本应用为基于python语言，在LangChain框架下运用bge-large-zh-v1.5嵌入模型、chroma向量数据库和Qwen3大语言模型搭建的RAG知识库问答系统（知识库文档类型仅限.txt文件）

在使用它之前，请确保将Qwen3大语言模型（任意版本）部署到本地，然后设置好环境变量。 我在文件当中是创建.env文件来储存api_key，这里也建议您在根目录下新建一个.env文件，并将你的base_url和api_key保存在里面。

这个项目所需要的依赖库名称都保存在项目根目录下的requirements文件里，其中的modelscope模组仅用于下载嵌入模型，如果你已经下载过嵌入模型并且不希望安装新的嵌入模型，可将requirements文件里的'modelscope'删去。

进入终端用一句命令将它们部署到位：pip install -r requirements.txt

接下来要做的是下载bge-large-zh-v1.5嵌入模型，如果你之前已经部署了嵌入模型，可以跳过下面的内容。

在安装完依赖库之后，可以先检查一下其中的modelscope模组是否成功安装，在确认modelscope模组安装完成后，就可以通过它来下载bge-large-zh-v1.5模型。

本项目的根目录下有一个文件fetch_em.py，用编辑器打开它，将其中的参数 cache_dir 的值修改为你希望bge-large-zh-v1.5模型要保存的路径。然后运行这个文件，它会开始下载模型，大概需要花一些时间，等待下载完成。

下载完成之后，复制你需要使用的嵌入模型的存放路径（绝对路径）。用编辑器打开根目录下的rag_core.py文件，找到其中的初始化模型的部分，有一行代码：self.embedding_model = HuggingFaceEmbeddings，其中的参数model_name的值要替换为你复制的嵌入模型的存放路径。

到这里，嵌入模型的部署就算完成了。

requirements文件里的依赖库都部署完成后，我们需要设置知识库文档路径。

根目录下的document文件夹用于存放知识库文档，本项目仅限.txt文件，所以要注意放入其中的文件类型。

知识库文档路径是 "./document/<知识库文档名>.txt"，记得改动一下文档名。然后将知识库文档路径复制替换到rag_core.py文件的RAGSystem类的__init__方法的document_path的参数值。这样知识库文档路径就设置完成了。

templates文件夹下的index.html文件即为服务的前端页面。

static文件夹是静态文件目录，可用于之后项目的进一步扩展。

chroma_data文件夹用于存放已创建的向量数据库，每次创建向量数据库成功和会自动储存到该文件夹，之后对相同文档再次运行就可以直接加载已有的向量数据库，不必再次新建。

后端则分为两个部分：rag_core.py和main.py

rag_core.py创建了名为RAGSystem的类，在langchain框架下，对文件做切分，分别创建检索器和向量数据库，书写提示词，调用大语言模型，再用链（chain）将检索器（retriever）提示词（prompt）、大语言模型（llm）和解析器连通。将用户的问题向量化并输入检索器，通过链后由大语言模型生成结果并交由解析器完成解析，最终输出。又另写一个用法query，用chain.invoke实现RAG功能，并创建一个实例。 

main.py则是用fastapi做了一个封装，对rag_core.py当中创建的实例进行了调用。

使用方法： 在终端启动main.py，等到终端显示 “Application startup complete.” 时，打开浏览器，在url栏里输入localhost:8000，按下Enter，就可以使用智能 RAG 问答系统了。
