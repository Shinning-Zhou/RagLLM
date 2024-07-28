from langchain.llms.base import LLM
from typing import Any, List, Optional, Generator, Mapping
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import (
    ChatGenerationChunk,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from threading import Thread
import torch
import asyncio


class Qwen2(LLM):
    model_name: str = "Qwen2-7B-Instruct/"
    model_path: str = "/data1n1/Qwen2-7B-Instruct/"
    device: torch.device = torch.device('cuda:0')
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    max_length: int = 50
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    history_len: int = 3

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map=self.device)

    def _prepare_messages(self, prompt: str) -> List[dict]:
        """准备消息格式。"""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

    def _generate_text(self, model_inputs) -> str:
        """生成文本并解码。"""
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=self.max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    @torch.inference_mode()
    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = self._prepare_messages(prompt)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        return self._generate_text(model_inputs)

    @torch.inference_mode()
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """异步版本的 _call 方法，用于支持 async 调用"""
        messages = self._prepare_messages(prompt)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = await asyncio.to_thread(
            self.model.generate,
            model_inputs.input_ids,
            max_new_tokens=self.max_new_tokens
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @torch.inference_mode()
    def _stream(self, 
                prompt: str, 
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Generator[str, None, None]:
        """同步流式输出接口"""
        messages = self._prepare_messages(prompt)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=self.max_new_tokens)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield ChatGenerationChunk(message=AIMessageChunk(content=new_text))

    @property
    def _llm_type(self) -> str:
        return self.model_name

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取识别参数。"""
        return {
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len
        }




# 测试 Qwen2Model 类
if __name__ == "__main__":
    import asyncio
    # 创建模型实例
    model = Qwen2()

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

