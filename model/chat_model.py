from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
from langchain_core.outputs import (
    ChatGenerationChunk,
)
from langchain.llms.base import LLM
from typing import AsyncGenerator, Optional, List, Any, Generator
from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
import torch
from langchain_core.messages import AIMessageChunk


class ChatModel(LLM):
    """
    ChatModel 继承自 LLM 基类，实现了封装 transformers 的 langchain chat 模型。
    """

    device_name: str = Field(
        "cuda:2", description="运算设备名称，例如cuda:0, cuda:1, cpu等"
    )
    model_name: str = Field("qwen/Qwen-7B-Chat", description="模型名称")

    model: AutoModelForCausalLM = Field(None, description="模型对象")
    tokenizer: AutoTokenizer = Field(None, description="分词器")

    class Config:
        """pydantic 设置."""

        allow_population_by_field_name = True

    def __init__(
        self, model_name: str = "qwen/Qwen-7B-Chat", device_name: str = "cuda:2"
    ):
        super().__init__()
        # 配置路径和使用设备
        model_path = "/data1n1/"
        self.device_name = device_name
        self.model_name = model_name

        # 加载模型和分词器，并设置模型参数
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path + model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path + model_name,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map=device_name,
            trust_remote_code=True,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path + model_name, trust_remote_code=True
        )

    @property
    def _llm_type(self) -> str:
        """返回模型名称"""
        return self.model_name

    @torch.inference_mode()  # 推理模式，不启用梯度计算
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """该函数是对 langchain chat 模型的封装，为固定接口"""
        response, _ = self.model.chat(
            self.tokenizer, prompt, []
        )  # 调用qwen模型的chat接口，该接口是对正常生成结果的封装
        return response

    @torch.inference_mode()
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """异步版本的_call，用于支持ainvoke，为固定接口"""
        response, _ = self.model.chat(self.tokenizer, prompt, [])
        return response

    @torch.inference_mode()
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Generator[ChatGenerationChunk, None, None]:
        """流式推理接口，返回生成的结果，为固定接口"""
        response = self.model.chat_stream(
            self.tokenizer, prompt, [], stream=True
        )  # 调用qwen模型的chat_stream接口，该接口是对流式生成结果的封装

        last_chunk = ""
        for chunk in response:
            chunk = chunk.replace(last_chunk, "")  # 去掉重复的部分
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk)
            )  # yield 协程返回，将函数变成生成器
            last_chunk = chunk

    @torch.inference_mode()
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """流式异步推理接口，返回生成的结果，为固定接口"""
        response = self.model.chat_stream(self.tokenizer, prompt, [], stream=True)

        async def async_generator(gen):  # 将同步生成器封装成异步生成器
            last_chunk = ""
            for chunk in gen:
                delta = chunk.replace(last_chunk, "")
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                last_chunk = chunk

        async for resp in async_generator(response):
            yield resp
