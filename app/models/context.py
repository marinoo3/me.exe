from pydantic import BaseModel, Field, computed_field

import uuid

from app.models import Chunk



class Context(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunks: list[Chunk]

    @computed_field
    @property
    def context(self) -> str:
        ctx = []
        for chunk in self.chunks:
            if chunk.source is not None:
                ctx.append(
                    f"> {chunk.source.name} ({chunk.source.category})"
                    f"\n...{chunk.content}..."
                )

        return "\n\n".join(ctx)

