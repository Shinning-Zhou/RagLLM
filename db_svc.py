from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from UI.item import Query, UploadPacket
from langchain_community.vectorstores import Chroma
import logging
import os
from model.embedding_model import EmbeddingModel
import ujson
import sqlite3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingService:
    """
    embedding服务，封装了嵌入方法
    """
    def __init__(self, chunk_size: int = 100, keep_separator: bool = False):
        self.model = EmbeddingModel()
        self.splitter = RecursiveCharacterTextSplitter(  # 词语切分分词器
            chunk_size=chunk_size,
            chunk_overlap=0,
            keep_separator=keep_separator,
            separators=["\n\n", "。", "!", "！", "?", "？", ";", "；", "."],
        )
        os.makedirs(f'{(os.path.dirname(__file__))}/chromadb', exist_ok=True)
        os.makedirs(f'{(os.path.dirname(__file__))}/emoji_chromadb', exist_ok=True)
        self.vdb = {  # 配置向量数据库，embedding_function 会调用其的embed_query方法
            "问答": Chroma('team_d', embedding_function=self.model, persist_directory=f'{(os.path.dirname(__file__))}/chromadb'),
            "表情包" : Chroma('team_d_emoji', embedding_function=self.model, persist_directory=f'{(os.path.dirname(__file__))}/emoji_chromadb')
            }
        self.retriver = {k: v.as_retriever() for k, v in self.vdb.items()}  # 配置检索器

    def embed_query(self, query: str) -> List[float]:
        """临时开给EmbeddingSession用的"""
        return self.model.embed_query(query)
    
    @staticmethod
    def transform(docs: List[Document]) -> List[str]:
        """转换格式"""
        return [doc.page_content for doc in docs]
    
    @staticmethod
    def dump(docs: List[Document]) -> List[Dict]:
        """导出属性"""
        return [{"id":doc.id,'metadata': doc.metadata, 'page_content': doc.page_content} for doc in docs]

    def embedding(self, docs: List[Document], intent: str, ids: List[str] = None) -> List[str]:
        """用检索器对document进行嵌入"""        
        return self.retriver[intent].add_documents(docs, ids=ids)

    def from_pdf(self, file_path: str) -> List[str]:
        """从pdf进行嵌入，file_path是pdf路径"""
        # 全部读取、切分并嵌入
        pdf_loader = PyPDFLoader(file_path)
        splits = pdf_loader.load_and_split(self.splitter)
        return self.embedding(splits, intent='问答')

    def from_img(self, dir_path: str, data: List[Dict[str, str]]) -> List[str]:
        """对图片进行嵌入，dir_path是data.json所在目录，data是data.json文件的json字典"""
        docs = []
        for item in data:
            fpath = os.path.join(dir_path, 'emo', item['filename'])
            docs.append(Document(page_content=item['content'], metadata={'source': fpath}))  # 构造Document，填入描述和路径
        return self.embedding(docs=docs, intent='表情包')

    def query(self, query: str, intent: str, search_type='mmr'):
        """处理嵌入请求，默认使用mmr方法进行搜索（最小重复性最大相似度搜索）"""
        logger.info(f"query: {query}")
        return self.dump(self.vdb[intent].search(query, search_type=search_type))


app = FastAPI()  # 后端服务类
embed_svc = EmbeddingService()


@app.get("/")
async def root():
    return {"message": "Welcome to the ChromaDB API"}


@app.post('/embedding/query/')
async def search(q: Query):
    return JSONResponse(content=embed_svc.query(query=q.query, intent=q.intent, search_type=q.search_type))

@app.post("/embedding/api")
async def embedding_api(q: Query):
    """给EmbeddingSession用的接口"""
    return JSONResponse(content=embed_svc.embed_query(q.query))

@app.post("/database/pdf")
async def upload_pdf(u:UploadPacket):
    """读取pdf并嵌入"""
    docs = []
    for file in os.listdir(u.dir_path):
        if file.endswith(".pdf"):
            docs.extend(embed_svc.from_pdf(os.path.join(u.dir_path, file)))

    # return JSONResponse(content=data)
    return JSONResponse(content=docs)


@app.post("/database/img")
async def upload_img(u:UploadPacket):
    """处理图片输入"""
    print(u.dir_path)
    desc_path = os.path.join(u.dir_path, "data.json")
    img_path = os.path.join(u.dir_path, "emo")

    if any([not os.path.exists(desc_path), not os.path.exists(img_path)]):  # 路径不存在
        print("invalid path")
        return JSONResponse(content="empty")
    with open(desc_path, "r") as f:  # 读取描述
        data = ujson.load(f)
        resp = embed_svc.from_img(u.dir_path, data=data)
    return JSONResponse(content=resp)

@app.get("/image")
async def download_image(file_path: str):
    """根据给定路径返回图片"""

    def iter_file(file_path):  # 构造读取文件数据的生成器
        with open(file_path, "rb") as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                yield data

    content = iter_file(file_path)
            

    return StreamingResponse(content=content, media_type="image/jpeg")
