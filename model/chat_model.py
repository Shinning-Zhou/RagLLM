from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
from langchain_core.outputs import (
    ChatGenerationChunk,
)
from langchain.llms.base import LLM
import torch
from typing import Optional, List, Any
from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk

class ChatModel(LLM):
    device_name: str = Field(
        "cuda:2", description="Name of the device to run the model."
    )
    model_name: str = Field("qwen/Qwen-7B-Chat", description="Name of the model.")

    model: AutoModelForCausalLM = Field(
        None, description="Hugging Face pipeline for chat model."
    )
    tokenizer: AutoTokenizer = Field(None, description="Tokenizer for the model.")


    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    def __init__(self, model_name: str = "qwen/Qwen-7B-Chat", device_name: str = "cuda:3"):
        super().__init__()
        model_path = "/data1n1/"
        self.device_name = device_name
        self.model_name = model_name


        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path + model_name,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map=device_name,
            trust_remote_code=True
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path + model_name,
            trust_remote_code=True
        )


    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model_name

    @torch.inference_mode()
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response, _ = self.model.chat(self.tokenizer, prompt, [])
        return response
    @torch.inference_mode()
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response, _ = self.model.chat(self.tokenizer, prompt, [])
        return response

    @torch.inference_mode()
    def _stream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response = self.model.chat_stream(self.tokenizer, prompt, [], stream=True)

        last_chunk = ""
        for chunk in response:
            chunk = chunk.replace(last_chunk, "")
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            last_chunk = chunk


    @torch.inference_mode()
    async def _astream(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response = self.model.chat_stream(self.tokenizer, prompt, [], stream=True)
        async def async_generator(gen):

            last_chunk = ""
            for chunk in gen:
                delta = chunk.replace(last_chunk, "")
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                last_chunk = chunk
        async for resp in async_generator(response):
            yield resp

