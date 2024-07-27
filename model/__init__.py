from .chat_model.chatglm import ChatGlm
from .chat_model.qwen import Qwen
from .chat_model.qwen2 import Qwen2

def get_model(model_name):
    if model_name == 'chat_glm':
        return ChatGlm()
    elif model_name == 'qwen':
        return Qwen()
    elif model_name == 'qwen2':
        return Qwen2()
    else:
        raise ValueError("Invalid model name")
    
