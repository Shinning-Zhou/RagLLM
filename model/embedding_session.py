import aiohttp
import asyncio
from typing import List
from langchain_core.embeddings import Embeddings

from UI.api import send_request
from UI.item import Query

class EmbeddingSession(Embeddings):
    def __init__(self):
        self.port = "8002"
        self.path = "embedding/api"

    def embed_query(self, query: str) -> List[float]:
        url = f"{self.path}"
        resp = send_request(url, self.port,json=Query(query=query).model_dump(), method="POST").json()
        return resp

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        texts = [doc for doc in documents]
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    async def aembed_query(self, query: str) -> List[float]:
        url = f"{self.path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=Query(query=query).model_dump()) as resp:
                return (await resp.json())

    async def aembed_documents(self, documents: List[str]) -> List[List[float]]:
        tasks = [self.aembed_query(doc) for doc in documents]
        embeddings = await asyncio.gather(*tasks)
        return embeddings