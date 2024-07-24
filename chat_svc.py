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

from model.chat_model import ChatModel
from model.embedding_session import EmbeddingSession
from UI.item import Dialog
from common.chain_utils import rag_history_chain, intent_history_chain

app = FastAPI()

class ChatService:
    def __init__(self):
        self.llm = ChatModel()
        self.embed = EmbeddingSession()

        self.vdb = {
            "问答": Chroma(
                "team_d",
                embedding_function=self.embed,
                persist_directory=f"{(os.path.dirname(__file__))}/chromadb",
            ),
            "表情包": Chroma(
                "team_d_emoji",
                embedding_function=self.embed,
                persist_directory=f"{(os.path.dirname(__file__))}/chromadb",
            ),
        }

        self.retriver = {k: v.as_retriever() for k, v in self.vdb.items()}

        self._store_cache_ = {}
        self.rag_chain = intent_history_chain(
            self.llm, self.retriver, history_load=self.warp_task()
        )

    def warp_task(self) -> Callable[[str], ChatMessageHistory]:
        def task(session_id: str):
            if session_id not in self._store_cache_:
                self._store_cache_[session_id] = ChatMessageHistory()

            chat_history = self._store_cache_[session_id]
            if len(chat_history.messages) >= 6:
                chat_history.messages = chat_history.messages[-6:]
            return chat_history

        return task
    
    def show_chat_history(
        self, session_id: str
    ):
        if session_id not in self._store_cache_:
            return {}
        dicts = [
            (i, json.dumps(d))
            for i, d in enumerate(messages_to_dict(self._store_cache_[session_id].messages))
        ]
        return {session_id:dicts}
        


async def split_stream(chunk_stream: AsyncIterator):
    async for chunk in chunk_stream:
        print(chunk)
        for key in chunk:
            if key == "answer":
                yield f"answer|{chunk[key]}"
            elif key == "context":
                for d in chunk[key]:
                    yield f"context|{json.dumps({'metadata': d.metadata,'page_content': d.page_content}, ensure_ascii=False)}"
            else:
                yield f"{key}|{json.dumps(chunk[key], ensure_ascii=False)}"


chat_svc = ChatService()
llm = chat_svc.llm


@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API"}


@app.post("/rag")
async def rag(dialog: Dialog):
    rag_chain = chat_svc.rag_chain
    stream = rag_chain.astream(
        {"input": dialog.message},
        config={
            "configurable": {
                "session_id": dialog.session_id,
            }
        },
    )
    return StreamingResponse(
        split_stream(stream),
        media_type="text/event-stream",
    )
    # return JSONResponse(chat_svc.embed.embed_query(dialog.message))


@app.post("/query")
async def query(msg: str):
    resp = (llm | StrOutputParser()).astream(msg)
    return StreamingResponse(
        resp,
        media_type="text/event-stream",
    )

@app.delete("/session/{session_id}")
async def session_list(session_id: str):
    try:
        chat_svc._store_cache_.pop(session_id)
        return JSONResponse({"message": "session deleted"})
    except KeyError:
        return JSONResponse({"message": "session not found"})
    
@app.get("/session")
async def session_list():
    session_list = {}
    for k in chat_svc._store_cache_:
        session_list[k] = len(chat_svc.show_chat_history(k))
    return JSONResponse(session_list)

@app.get("/session/{session_id}")
async def session_show(session_id: str):
    dicts = chat_svc.show_chat_history(session_id)
    return JSONResponse(dicts)
