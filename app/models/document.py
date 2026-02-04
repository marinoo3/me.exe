from pydantic import BaseModel
from typing import Optional


class Document(BaseModel):
    id: Optional[int] = None
    name: str
    category: str