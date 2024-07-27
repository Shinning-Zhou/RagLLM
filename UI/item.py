from pydantic import BaseModel, Field


class Dialog(BaseModel):
    """对话数据模型"""
    session_id: str = Field(..., title="Session ID")
    message: str = Field(..., title="Message")

class Query(BaseModel):
    """查询数据模型"""
    query: str = Field(..., title="Query")
    intent: str = Field("问答", title="Intent")
    search_type: str = Field('mmr', title="Search Type")

class UploadPacket(BaseModel):
    """上传数据模型"""
    dir_path: str = Field(..., title="Directory Path")
