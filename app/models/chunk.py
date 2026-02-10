from pydantic import BaseModel, ConfigDict, field_serializer
from typing import Optional
import numpy as np

from app.models import Document


class Chunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: Optional[int] = None
    document_id: int
    content: str
    emb_384d: np.ndarray

    distance: Optional[float] = None
    score: Optional[float] = None
    
    source: Optional[Document] = None

    @field_serializer("emb_384d")
    def serialize_embedding(self, v: np.ndarray, _info):
        # Serialize np array for json export
        return v.tolist()
    
    @field_serializer("distance")
    def serialize_distance(self, d: float, _info):
        # Round distance to 3 digits for json export
        return round(d, 3)
    
    @field_serializer("score")
    def serialize_score(self, s: float, _info):
        # Round distance to 2 digits for json export
        return round(s, 2)
