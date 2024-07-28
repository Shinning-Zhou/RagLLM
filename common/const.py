from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field

class Intent(BaseModel):
    """意图"""
    name: str = Field(..., title="意图名称")
    description: str = Field(..., title="意图描述")
    prompt_template: str = Field(..., title="意图模板")

chat_intent = Intent(  # 聊天意图
    name="聊天",
    description="回答一般性问题",
    prompt_template="""{input}"""
)

rag_intent = Intent(  # 问答意图
    name="问答",
    description="回答具体问题",
    prompt_template="""你是一位非常聪明的助手。
    你擅长借助资料以简洁易懂的方式回答问题。
    当你不知道某个问题的答案时，你会坦诚承认。

    这是一个问题：
    {input}"""
)

emoji_intent = Intent(  # 表情包意图
    name="表情包",
    description="回答表情包相关问题",
    prompt_template="""你是一位追逐潮流的年轻人。
    你擅长从图库里找到和描述一致的表情包并在对话中展示。

    这是一个问题：
    {input}
    
    请帮他找到对应的表情包"""
)

router_prompt = PromptTemplate.from_template(  # 路由模板，用来识别意图
    f"""
    你是一位非常聪明的助手，你擅长识别问题的主题。
    你只能从 聊天 问答 表情包 这三个主题中回答问题。
    这几个主题中回答问题。
    """ + """
    这是一个问题：
    {input}

    请告诉我这个问题属于哪个主题，回答格式如下
    ```
    [主题]
    ```
    如果不知道或者不属于你能分辨的主题，请说"[unknown]"。
    """ + f"""
    例如：
    "牛顿第二定律是什么？" -> "[问答]"
    "今天天气怎么样？" -> "[聊天]"
    "你好" -> "[聊天]"
    "介绍一下你自己" -> "[聊天]"
    "帮我找个哭泣的表情包" -> "[表情包]"
    """
)

contextualize_q_system_prompt = (  # 总结模板，让大模型能根据对话历史和问题，生成问题来检索参考资料
    """
    给定聊天历史记录和最新用户的问题，问题可能引用聊天历史记录中的上下文。
    形成独立的问题，无需聊天历史记录即可理解。
    不需要回答，只需重新构思即可。
    请总结50字以内的问题，并保持问题简洁。
    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(  # 对话提示词
    [
        ("system", contextualize_q_system_prompt),  # 放模板
        MessagesPlaceholder("chat_history"),  # 放对话历史
        ("human", "{input}"),  # 放问题
    ]
)


system_prompt = (  # 对话提示词
    """
    你是一名助手
    请使用以下检索到的上下文来回答问题
    如果不知道答案，请说不知道。
    请使用三个句子以内的答案，并保持答案简洁。
    
    
    {context}
    """
)

qa_prompt = ChatPromptTemplate.from_messages(  # 问答提示词
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
