
import asyncio
from typing import Callable, Dict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.transform import TransformChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.utils import trim_messages

from .const import system_prompt, qa_prompt, contextualize_q_prompt, rag_intent, chat_intent, emoji_intent, router_prompt

def get_normal_chain(llm, p_info):
    prompt = ChatPromptTemplate.from_messages(  # 问答提示词
        [
            ("system", p_info["prompt_template"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | {"answer": llm | StrOutputParser(name='answer')}

def rag_history_chain(
    label: str,
    model: Runnable,
    retriever: Runnable[str, list[Document]],
    history_load: Callable[[], BaseChatMessageHistory],
) -> Runnable:
    """创建一个RAG模型的聊天链

    Args:
        model (BaseChatModel): 语言模型
        retriever (VectorStoreRetriever): 检索器
    """
    history_aware_retriever = create_history_aware_retriever(
        model,
        retriever,
        contextualize_q_prompt,
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return label, rag_chain

def wrap_router(chains: Dict[str, Runnable]) -> Callable[[Dict], Runnable]:
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

    router_chain = router_prompt | model | StrOutputParser()

    candidate_chains = {
        "聊天": get_normal_chain(model, chat_intent.model_dump()),
    }

    rag_chain = [
        (rag_history_chain(label, model, ret, history_load)) for label, ret in retriever.items()
    ]

    for label, chain in rag_chain:
        candidate_chains[label] = chain

    full_chain = {
        "topic": router_chain,
        "input": lambda x: x["input"],
    } | RunnableWithMessageHistory(
        RunnableLambda(wrap_router(candidate_chains)),
        history_load,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )

    return full_chain
