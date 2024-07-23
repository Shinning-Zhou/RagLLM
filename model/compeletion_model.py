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


class ChatModel(BaseChatModel):
    device_name: str = Field(
        "cuda:3", description="Name of the device to run the model."
    )
    model_name: str = Field("Llama2/Llama-2-7b-hf", description="Name of the model.")

    # model_name: str = Field("qwen/Qwen-7B-Chat", description="Name of the model.")
    model: HuggingFacePipeline = Field(
        None, description="Hugging Face pipeline for chat model."
    )
    tokenizer: AutoTokenizer = Field(None, description="Tokenizer for the model.")

    model_kwargs: Dict = Field(
        None,
        description="Keyword arguments for the model's generate method.",
    )

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    def __init__(self, **kwargs):
        super().__init__()
        model_path = "/data1n1/"
        self.device_name = "cuda:3"

        # self.model_name = "Llama2/Llama-2-7b-hf"
        # self.model_name = "Llama2/Llama-2-7b-chat-hf"
        # self.model_name = "mistral-7B-v0.1"
        # self.model_name = "Mistral-7B-Instruct-v0.1"
        self.model_name = "Qwen2-7B-Instruct"


        self.tokenizer = AutoTokenizer.from_pretrained(model_path + self.model_name, trust_remote_code=True)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path + self.model_name,
        #     output_hidden_states=True,
        #     output_attentions=True,
        #     torch_dtype=torch.float16,
        #     device_map=self.device_name,
        #     trust_remote_code=True
        # )
        # self.model_kwargs = {
        #     "max_new_tokens": 128,
        #     "repetition_penalty": 1.5,
        #     "do_sample": True,
        #     "early_stopping": True,
        # }
        
        # def token_generator(query: str) -> List[int]:
        #     inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        #     return inputs['input_ids'].squeeze().tolist()
        pipeline = transformers.pipeline(
            'text-generation',
            model=model_path + self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            max_new_tokens=128,
            repetition_penalty=1.15,
            do_sample=True,
            )
        self.model = HuggingFacePipeline(pipeline=pipeline)
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model_name

    @torch.inference_mode()
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        query = messages[-1].content
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        token_n = self.tokenizer.encode(query, return_tensors="pt").shape[1]
        generated_ids = self.model.generate(**inputs, **self.model_kwargs)
        output = self.tokenizer.batch_decode(generated_ids[:, token_n:])[0].strip()

        message = AIMessage(
            content=output,
            additional_kwargs={},  # 用于添加额外的有效负载（例如，函数调用请求）
            response_metadata={  # 用于响应元数据
                "time_in_seconds": 3,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @torch.inference_mode()
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        query = messages[-1].content
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        token_n = self.tokenizer.encode(query, return_tensors="pt").shape[1]
        generated_ids = await self.model.agenerate(**inputs, **self.model_kwargs)
        output = self.tokenizer.batch_decode(generated_ids[:, token_n:])[0].strip()

        message = AIMessage(
            content=output,
            additional_kwargs={},  # 用于添加额外的有效负载（例如，函数调用请求）
            response_metadata={  # 用于响应元数据
                "time_in_seconds": 3,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # def stream_thr(self, inputs, streamer: TextIteratorStreamer, **kwargs: Any):
    #     with torch.no_grad():
    #         outputs = self.model.chat(
    #             inputs.input_ids,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #             **kwargs
    #             )
    #     return self.tokenizer.decode(
    #         outputs[0],
    #         streamer=streamer,
    #         skip_special_tokens=True
    #         )

    @torch.inference_mode()
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
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
            # chunk = chunk.replace(self.tokenizer.bos_token, "")
            # chunk = chunk.replace(self.tokenizer.eos_token, "")
            if chunk and token_cnt >= token_n:
                # print(f"content: {chunk}")
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
                if run_manager:
                    run_manager.on_llm_new_token(query, chunk=chunk)

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )

        if run_manager:
            run_manager.on_llm_new_token(query, chunk=chunk)

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
