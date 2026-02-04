from typing import List, Literal
from pydantic import BaseModel



class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    conversation: List[Message]
    query: str