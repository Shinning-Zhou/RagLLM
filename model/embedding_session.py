import aiohttp
import asyncio
from typing import List
from langchain_core.embeddings import Embeddings
from UI.api import send_request
from UI.item import Query

class EmbeddingSession(Embeddings):
    """
    EmbeddingSession 类用于处理 embedding 模型的 API 请求，并返回 embedding 结果。
    该类是对EmbeddingModel的网络引用封装，并提供异步接口。
    """
    def __init__(self):
        self.port = "8002"  # embedding 模型的端口
        self.path = "embedding/api"  # embedding 模型的接口路径

    def embed_query(self, query: str) -> List[float]:
        """
        计算 query 的 embedding 向量。
        该函数是对 langchain embeddings 模型的封装，为固定接口
        """
        url = f"{self.path}"
        resp = send_request(url, self.port,json=Query(query=query).model_dump(), method="POST").json()
        return resp

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        计算 documents 的 embedding 向量。
        该函数是对 langchain embeddings 模型的封装，为固定接口
        """
        texts = [doc for doc in documents]
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    async def aembed_query(self, query: str) -> List[float]:
        """异步版的 embed_query 函数"""
        url = f"{self.path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=Query(query=query).model_dump()) as resp:
                return (await resp.json())

    async def aembed_documents(self, documents: List[str]) -> List[List[float]]:
        """异步版的 embed_documents 函数"""
        tasks = [self.aembed_query(doc) for doc in documents]
        embeddings = await asyncio.gather(*tasks)
        return embeddings