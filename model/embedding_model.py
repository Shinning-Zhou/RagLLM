from typing import List
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
    def __init__(self):
        super().__init__()
        device_name = "cuda:4"  # Specify a GPU to use
        self.device = torch.device(device_name)

        model_path = "/data1n1/"
        model_name = "Mistral-7B-Instruct-v0.1"

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_path + model_name,
                                                    output_hidden_states=True,
                                                    output_attentions=True,
                                                    torch_dtype=torch.float16,
                                                    device_map=device_name)

        ## Push the GPT model to GPU:
        self.model.to(device_name, dtype=torch.float16)
    
    def generate_response(self, query: str):
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=200)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def get_hidden_states(self, text:str) -> List[torch.Tensor]:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            ## len(hidden_states) = number of layers of the GPT model
            return hidden_states

    def embed_query(self, query: str) -> List[float]:
        hidden_states = self.get_hidden_states(query)
        embedding = hidden_states[-1].mean(dim=1).squeeze().to('cpu').tolist()
        del hidden_states
        return embedding

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        texts = [doc for doc in documents]
        embeddings = [self.embed_query(text) for text in texts]
        return embeddings
    
    def __del__(self):
        del self.model
        del self.tokenizer