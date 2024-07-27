from typing import List
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingModel(Embeddings):
    """
    EmbeddingModel 继承自 Embeddings，实现了封装 transformers 的 langchain embedding 模型。
    """

    device_name: str = Field("cuda:4", description="模型运行的设备名称")
    model_name: str = Field("Mistral-7B-Instruct-v0.1", description="模型名称")
    tokenizer: AutoTokenizer = Field(None, description="tokenizer 对象")
    model: AutoModelForCausalLM = Field(None, description="模型对象")

    class Config:
        """pydantic 设置."""

        allow_population_by_field_name = True

    def __init__(
        self, model_name: str = "Mistral-7B-Instruct-v0.1", device_name: str = "cuda:4"
    ):
        super().__init__()
        # 配置路径和使用设备
        model_path = "/data1n1/"
        self.device_name = device_name
        self.model_name = model_name

        # 加载模型和分词器，并设置模型参数
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path + model_name,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.float16,
            device_map=device_name,
        )

    @torch.inference_mode()
    def generate_response(self, query: str) -> str:
        """
        根据输入的 query 生成回复。
        """
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device_name)
        outputs = self.model.generate(**inputs, max_length=200)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    @torch.inference_mode()
    def get_hidden_states(self, text: str) -> List[torch.Tensor]:
        """
        获取推理完成后的隐藏状态。
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device_name)
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

    def embed_query(self, query: str) -> List[float]:
        """
        计算 query 的 embedding 向量。
        该函数是对 langchain embeddings 模型的封装，为固定接口
        """
        hidden_states = self.get_hidden_states(query)
        embedding = hidden_states[-1].mean(dim=1).squeeze().to("cpu").tolist()
        del hidden_states
        return embedding

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        计算 documents 的 embedding 向量。
        该函数是对 langchain embeddings 模型的封装，为固定接口
        """
        texts = [doc for doc in documents]
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings

    def __del__(self):
        """
        释放模型和 tokenizer 的资源。
        """
        del self.model
        del self.tokenizer
