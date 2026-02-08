from pydantic import BaseModel, ConfigDict
from typing import Optional


class Document(BaseModel):
    id: Optional[int] = None
    name: str
    category: str
    url: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        # Hash the data that defines equality.
        # To make the objects hashable
        return hash((self.id, self.name, self.category))