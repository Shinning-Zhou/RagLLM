import asyncio
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModel,
    GenerationConfig,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from langchain.llms.base import LLM
import transformers
import torch
from typing import AsyncIterator, Dict, Optional, List, Any, Iterator
from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import (
    ChatGenerationChunk,
    ChatGeneration,
    ChatResult,
)
import time
from langchain_core.embeddings import Embeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from UI.api import send_request


class CompeletionModel(BaseChatModel):
    device_name: str = Field(
        "cuda:2", description="运行设备"
    )
    model_name: str = Field("Llama2/Llama-2-7b-hf", description="模型名称")

    # model_name: str = Field("qwen/Qwen-7B-Chat", description="Name of the model.")
    model: AutoModelForCausalLM = Field(None, description="模型")
    tokenizer: AutoTokenizer = Field(None, description="分词器")

    model_kwargs: Dict = Field(
        None,
        description="为模型添加的额外参数，例如max_new_tokens, repetition_penalty, do_sample, early_stopping",
    )

    class Config:
        """pydantic设置."""

        allow_population_by_field_name = True

    def __init__(self, **kwargs):
        super().__init__()
        # 加载模型路径和设备
        model_path = "/data1n1/"
        self.device_name = "cuda:3"
        # self.model_name = "Llama2/Llama-2-7b-hf"
        # self.model_name = "Llama2/Llama-2-7b-chat-hf"
        # self.model_name = "mistral-7B-v0.1"
        # self.model_name = "Mistral-7B-Instruct-v0.1"
        self.model_name = "Qwen2-7B-Instruct"

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path + self.model_name,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map=self.device_name,
            trust_remote_code=True
        )
        self.model_kwargs = {
            "max_new_tokens": 128,  # 最大生成长度
            "repetition_penalty": 1.5,  # 重复惩罚
            "do_sample": True,  # 使用采样
            "early_stopping": True,  # 早停，由于beams参数默认为1，当生成重复片段超过1时，会停止生成。
        }
        
    @property
    def _llm_type(self) -> str:
        """返回语言模型类型"""
        return self.model_name

    @torch.inference_mode()
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """该函数是对 langchain chat 模型的封装，为固定接口"""
        query = messages[-1].content
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        token_n = self.tokenizer.encode(query, return_tensors="pt").shape[1]
        generated_ids = self.model.generate(**inputs, **self.model_kwargs)
        output = self.tokenizer.batch_decode(generated_ids[:, token_n:])[0].strip()

        return output

    @torch.inference_mode()
    async def _acall(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步版本的_call，用于支持ainvoke，为固定接口"""
        query = messages[-1].content
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        token_n = self.tokenizer.encode(query, return_tensors="pt").shape[1]
        generated_ids = await self.model.agenerate(**inputs, **self.model_kwargs)
        output = self.tokenizer.batch_decode(generated_ids[:, token_n:])[0].strip()

        return output

    # 流式传输，使用分线程实现
    @torch.inference_mode()
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式推理接口，返回生成的结果，为固定接口"""
        
        query = messages[-1].content
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        token_n = self.tokenizer.encode(query, return_tensors="pt").shape[1]

        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(inputs, streamer=streamer, **self.model_kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        token_cnt = 0
        for chunk in streamer:
            print(f"content: {chunk}")
            token_cnt += self.tokenizer.encode(chunk, return_tensors="pt").shape[1]
            if token_cnt >= 1024:
                print('reach max token cnt')
                break
            if chunk and token_cnt >= token_n:
                # print(f"content: {chunk}")
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )


    # 由于分线程写异步流比较麻烦，因此暂时不支持异步流
    # @torch.inference_mode()
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:
    #     def task(**kwargs):
    #         asyncio.run(self.model.agenerate(**kwargs))
    #     query = messages[-1].content
    #     inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)

    #     streamer = TextIteratorStreamer(self.tokenizer)
    #     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
    #     thread = Thread(target=task, kwargs=generation_kwargs)
    #     thread.start()

    #     for chunk in streamer:
    #         if chunk:
    #             print(f'content: {chunk}')
    #             yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
    #             if run_manager:
    #                 run_manager.on_llm_new_token(query, chunk=chunk)

    #     chunk = ChatGenerationChunk(
    #         message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
    #     )

    #     if run_manager:
    #         run_manager.on_llm_new_token(query, chunk=chunk)

if __name__ == "__main__":
    """
    简单测试，需要实现的效果如下：
    1. 测试接口正常
    2. 输出结果开头不能包含输入（不然这里做rag会出问题）
    3. 要有对话的效果
    """
    model = CompeletionModel()

    # 测试同步接口
    print(model.invoke("你好"))

    # 测试异步接口
    asyncio.run(model.ainvoke("你好"))

    # 测试流式接口
    for chunk in model.stream("你好"):
        print(chunk)

    # 测试异步流式接口
    # async for chunk in model.astream("你好"):
    #     print(chunk)