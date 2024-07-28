from typing import Callable, Dict, List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from .const import (
    Intent,
    system_prompt,
    contextualize_q_prompt,
    rag_intent,
    chat_intent,
    emoji_intent,
    router_prompt,
)


def get_normal_chain(llm, p_info):
    """获取普通对话链"""
    prompt = ChatPromptTemplate.from_messages(  # 问答提示词
        [
            ("system", p_info["prompt_template"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | {
        "answer": llm | StrOutputParser(name="answer")
    }  # 模板 -> 大模型 -> 解析成字符串


def rag_history_chain(
    label: str,
    model: Runnable,
    retriever: Runnable[str, List[Document]],
    intent: Intent,
) -> Runnable:
    """创建一个RAG模型的聊天链

    Args:
        model (BaseChatModel): 语言模型
        retriever (VectorStoreRetriever): 检索器
    """
    history_aware_retriever = create_history_aware_retriever(  # 构造对话历史总结链
        model,
        retriever,
        contextualize_q_prompt,
    )
    qa_prompt = ChatPromptTemplate.from_messages(  # 问答提示词
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", intent.prompt_template),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        model, qa_prompt
    )  # 构造参考文献链

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )  # 构造检索链

    return rag_chain


def wrap_router(
    chains: Dict[str, Runnable]
) -> Callable[[Dict], Runnable]:  # 路由函数，用来识别意图并切换链
    def router(info):
        print(info)
        for k, v in chains.items():
            if k != "聊天" and k in info["topic"]:
                return v
        return chains["聊天"]

    return router


def intent_history_chain(
    model: Runnable,
    retriever: Dict[str, Runnable[str, list[Document]]],
    history_load: Callable[..., BaseChatMessageHistory],
) -> Runnable:
    """创建一个RAG模型的聊天链

    Args:
        model (BaseChatModel): 语言模型
        retriever (VectorStoreRetriever): 检索器
        history_load (Callable[..., BaseChatMessageHistory]): 加载历史记录的方法
    """

    router_chain = (
        router_prompt | model | StrOutputParser()
    )  # 路由模板 -> 大模型 -> 解析回答

    candidate_chains = {
        "聊天": get_normal_chain(model, chat_intent.model_dump()),  # 普通聊天链
        "问答": rag_history_chain(
            "问答", model, retriever["问答"], rag_intent
        ),  # RAG问答链，根据给的检索器而不同
        "表情包": rag_history_chain(
            "表情包", model, retriever["表情包"], emoji_intent
        ),  # RAG表情包链
    }

    full_chain = {
        "topic": router_chain,
        "input": lambda x: x["input"],
    } | RunnableWithMessageHistory(
        RunnableLambda(wrap_router(candidate_chains)),
        history_load,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )  # 意图识别链 -> 对话历史链(路由链 -> 路由函数 -> 候选链)

    return full_chain
