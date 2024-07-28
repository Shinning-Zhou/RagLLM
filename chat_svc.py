from typing import Callable, AsyncIterator
import os
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from langchain.schema import messages_to_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from model import get_chat_model
from model.embedding_session import EmbeddingSession
from UI.item import Dialog
from common.chain_utils import rag_history_chain, intent_history_chain

app = FastAPI()

class ChatService:
    def __init__(self):
        self.llm = get_chat_model("qwen2")
        self.embed = EmbeddingSession()

        os.makedirs(f"{(os.path.dirname(__file__))}/chromadb", exist_ok=True)
        os.makedirs(f"{(os.path.dirname(__file__))}/emoji_chromadb", exist_ok=True)
        self.vdb = {  # 设置向量数据库
            "问答": Chroma(
                "team_d",
                embedding_function=self.embed,
                persist_directory=f"{(os.path.dirname(__file__))}/chromadb",
            ),
            "表情包": Chroma(
                "team_d",
                embedding_function=self.embed,
                persist_directory=f"{(os.path.dirname(__file__))}/emoji_chromadb",
            ),
        }

        self.retriver = {k: v.as_retriever(search_type = "mmr") for k, v in self.vdb.items()}  # langchain检索器

        self._store_cache_ = {}  # 历史记录缓存
        self.rag_chain = intent_history_chain(  # 获取意图链
            self.llm, self.retriver, history_load=self.warp_task()
        )

    def warp_task(self) -> Callable[[str], ChatMessageHistory]:
        """封装历史记录生成函数"""
        def task(session_id: str):  # 输入会话id，返回对话记录类ChatMessageHistory
            if session_id not in self._store_cache_:
                self._store_cache_[session_id] = ChatMessageHistory()

            chat_history = self._store_cache_[session_id]
            if len(chat_history.messages) >= 6:  # 只取最后3对问答
                chat_history.messages = chat_history.messages[-6:]
            return chat_history

        return task
    
    def show_chat_history(
        self, session_id: str
    ):
        """返回对话历史记录"""
        if session_id not in self._store_cache_:
            return {}  # 没有对话历史就返回空字典
        dicts = [
            (i, json.dumps(d))
            for i, d in enumerate(messages_to_dict(self._store_cache_[session_id].messages))
        ]
        return {session_id:dicts}
    

async def split_stream(chunk_stream: AsyncIterator):
    """
    对异步返回的数据做格式化
    """
    async for chunk in chunk_stream:
        print(chunk)
        for key in chunk:
            if key == "answer":  # 回答文本
                yield f"answer|{chunk[key]}"
            elif key == "context":  # 参考文献
                for d in chunk[key]:
                    yield f"context|{json.dumps({'metadata': d.metadata,'page_content': d.page_content}, ensure_ascii=False)}"
            else:  # 其他记录
                try:
                    yield f"{key}|{json.dumps(chunk[key], ensure_ascii=False)}"
                except:
                    continue

chat_svc = ChatService()  # 实例化

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API"}


@app.post("/rag")
async def rag(dialog: Dialog):
    """进行rag问答"""
    rag_chain = chat_svc.rag_chain  # 使用rag链
    stream = rag_chain.astream(  # 异步流式返回
        {"input": dialog.message},  # 输入
        config={
            "configurable": {
                "session_id": dialog.session_id,  # 会话id，用于获取历史记录
            }
        },
    )
    return StreamingResponse(
        split_stream(stream),
        media_type="text/event-stream",  # 流式返回文本
    )


@app.post("/query")
async def query(msg: str):
    """进行普通llm问答"""
    resp = (chat_svc.llm | StrOutputParser()).astream(msg)  # 流式异步返回问答
    return StreamingResponse(
        resp,
        media_type="text/event-stream",
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除对话历史"""
    try:
        chat_svc._store_cache_.pop(session_id)
        return JSONResponse({"message": "session deleted"})
    except KeyError:
        return JSONResponse({"message": "session not found"})
    
@app.get("/session")
async def session_list():
    """获取所有对话历史长度"""
    session_list = {}
    for k in chat_svc._store_cache_:
        session_list[k] = len(chat_svc.show_chat_history(k))
    return JSONResponse(session_list)

@app.get("/session/{session_id}")
async def session_show(session_id: str):
    """查看会话id对应的对话历史记录"""
    dicts = chat_svc.show_chat_history(session_id)
    return JSONResponse(dicts)
