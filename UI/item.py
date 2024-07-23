from pydantic import BaseModel, Field


class Dialog(BaseModel):
    session_id: str = Field(..., title="Session ID")
    message: str = Field(..., title="Message")

class Query(BaseModel):
    query: str = Field(..., title="Query")
    intent: str = Field("问答", title="Intent")
    search_type: str = Field('mmr', title="Search Type")

class UploadPacket(BaseModel):
    dir_path: str = Field(..., title="Directory Path")
