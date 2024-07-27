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


class Qwen(LLM):
    """Qwen-7B-Chat 语言模型。"""
    
    device: torch.device = torch.device('cuda:4')
    model_name: str = "Qwen-7B-Chat"
    model_path: str = "/data1n1/qwen/Qwen-7B-Chat/"
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    
    def __init__(self):
        super().__init__()
        # 加载模型和分词器，并设置模型参数
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    @property
    def _llm_type(self) -> str:
        """返回模型名称"""
        return self.model_name

    @torch.inference_mode()
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """同步调用模型生成响应。"""
        response, _ = self.model.chat(self.tokenizer, prompt, [])
        return response

    @torch.inference_mode()
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """异步调用模型生成响应。"""
        response, _ = await asyncio.to_thread(self.model.chat, self.tokenizer, prompt, [])
        return response
    
    @torch.inference_mode()
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Generator[ChatGenerationChunk, None, None]:
        """同步流式推理接口。"""
        response = self.model.chat_stream(self.tokenizer, prompt, [], stream=True)
        return self._generate_chunks(response)
    
    @torch.inference_mode()
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """异步流式推理接口。"""
        response = self.model.chat_stream(self.tokenizer, prompt, [], stream=True)
        async for chunk in self._generate_async_chunks(response):
            yield chunk

    def _generate_chunks(self, response):
        """从同步流中生成块。"""
        last_chunk = ""
        for chunk in response:
            delta = chunk.replace(last_chunk, "")
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            last_chunk = chunk

    async def _generate_async_chunks(self, response):
        """将同步生成器封装成异步生成器。"""
        last_chunk = ""
        for chunk in response:
            delta = chunk.replace(last_chunk, "")
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            last_chunk = chunk

# 测试 Qwen2Model 类
if __name__ == "__main__":
    import asyncio
    # 创建模型实例
    model = Qwen()

    # 测试 _call 方法
    prompt = "你好！"
    response = model._call(prompt)
    print("Response from _call:", response)

    # 测试 _acall 方法（异步）
    async def test_acall():
        response = await model._acall(prompt)
        print("Response from _acall:", response)

    asyncio.run(test_acall())

    # 测试 _stream 方法
    def test_stream():
        generator = model._stream(prompt)
        print("Streaming response:")
        for chunk in generator:
            print(chunk.message.content)

    test_stream()

    # 测试 _astream 方法（异步流式）
    async def test_astream():
        async for chunk in model._astream(prompt):
            print("Async Streaming response:", chunk.message.content)

    asyncio.run(test_astream())
