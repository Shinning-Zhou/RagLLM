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
    def __init__(self, chunk_size: int = 100, keep_separator: bool = False):
        self.model = EmbeddingModel()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            keep_separator=keep_separator,
            separators=["\n\n", "。", "!", "！", "?", "？", ";", "；", "."],
        )
        os.makedirs(f'{(os.path.dirname(__file__))}/chromadb', exist_ok=True)
        self.vdb = {
            "问答": Chroma('team_d',embedding_function=self.model, persist_directory=f'{(os.path.dirname(__file__))}/chromadb'),
            "表情包" : Chroma('team_d_emoji', embedding_function=self.model, persist_directory=f'{(os.path.dirname(__file__))}/chromadb')
            }
        self.retriver = {k: v.as_retriever() for k, v in self.vdb.items()}

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_query(query)
    
    @staticmethod
    def transform(docs: List[Document]) -> List[str]:
        return [doc.page_content for doc in docs]
    
    @staticmethod
    def dump(docs: List[Document]) -> List[Dict]:
        return [{"id":doc.id,'metadata': doc.metadata, 'page_content': doc.page_content} for doc in docs]

    def embedding(self, docs: List[Document], intent: str, ids: List[str] = None) -> List[str]:
        # self.loader = PDFPlumberLoader()
        
        return self.retriver[intent].add_documents(docs, ids=ids)

    def from_pdf(self, file_path: str) -> List[str]:
        pdf_loader = PyPDFLoader(file_path)
        splits = pdf_loader.load_and_split(self.splitter)
        return self.embedding(splits, intent='问答')

    def from_img(self, dir_path: str, data: List[Dict[str, str]]) -> List[str]:
        docs = []
        for item in data:
            fpath = os.path.join(dir_path, 'emo', item['filename'])
            docs.append(Document(page_content=item['content'], metadata={'source': fpath}))
        return self.embedding(docs=docs, intent='表情包')

    def query(self, query: str, intent: str, search_type='mmr'):
        logger.info(f"query: {query}")
        return self.dump(self.vdb[intent].search(query, search_type=search_type))


app = FastAPI()
embed_svc = EmbeddingService()


@app.get("/")
async def root():
    return {"message": "Welcome to the ChromaDB API"}


@app.post('/embedding/query/')
async def search(q: Query):
    return JSONResponse(content=embed_svc.query(query=q.query, intent=q.intent, search_type=q.search_type))

@app.post("/embedding/api")
async def embedding_api(q: Query):
    return JSONResponse(content=embed_svc.embed_query(q.query))

@app.post("/database/pdf")
async def upload_pdf(u:UploadPacket):
    docs = []
    for file in os.listdir(u.dir_path):
        if file.endswith(".pdf"):
            docs.extend(embed_svc.from_pdf(os.path.join(u.dir_path, file)))

    # return JSONResponse(content=data)
    return JSONResponse(content=docs)


@app.post("/database/img")
async def upload_img(u:UploadPacket):
    print(u.dir_path)
    desc_path = os.path.join(u.dir_path, "data.json")
    img_path = os.path.join(u.dir_path, "emo")
    if any([not os.path.exists(desc_path), not os.path.exists(img_path)]):
        print("invalid path")
        return JSONResponse(content="empty")
    with open(desc_path, "r") as f:
        data = ujson.load(f)
        resp = embed_svc.from_img(u.dir_path, data=data)
    return JSONResponse(content=resp)

@app.get("/image")
async def download_image(file_path: str):
    """根据给定路径返回图片"""

    def iter_file(file_path):
        with open(file_path, "rb") as f:
            while True:
                data = f.read(1024)
                if not data:
                    break
                yield data

    content = iter_file(file_path)
            

    return StreamingResponse(content=content, media_type="image/jpeg")
