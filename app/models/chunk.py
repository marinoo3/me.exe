from pydantic import BaseModel, ConfigDict
from typing import Optional
import numpy as np


class Chunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[int] = None
    document_id: int
    content: str
    emb_384d: np.ndarray
    
    source_name: Optional[str] = None
    source_categorie: Optional[str] = None