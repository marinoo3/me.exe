import os
from typing import Optional
from mistralai import Mistral

from app.exceptions import MissingAPIKeyError
from app.models import Chunk, Message





class LLMHandler:
    """
    Handler for Mistral LLM interactions.

    Manages model selection, prompt templates and conversation history.
    """

    temperature = .5
    max_tokens = 5000
    system_prompt = ""

    def __init__(self, model: str = "mistral-medium-latest"):
        """
        Initialize the LLM handler.

        Args:
            model: Model key from MODELS dict.
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")

        if not self.api_key:
            raise MissingAPIKeyError(
                "MISTRAL_API_KEY not found in environment variables"
            )

        self._client = None
        self.model_name = model

    @property
    def client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)

        return self._client
    
    def build_rag_context(self, chunks: list[Chunk]) -> str:
        """
        Create a RAG context / prompt from a list of chunk

        Args:
            chunks (list[Chunk]): The chunks to build a context from

        Returns:
            str: Formatted RAG context:
            ```md
            > name (category)
            content

            [...]
            ```
        """

        context = []
        for chunk in chunks:
            context.append(
                f"> {chunk.source_name} ({chunk.source_categorie})" +
                f"\n...{chunk.content}..."
            )
        
        return "\n\n".join(context)

    
    def chat(
        self,
        message: str,
        history: list[Message],
        rag_context: Optional[str] = None
    ) -> str:
        """
        Generate a response in a conversation context with history.

        Args:
            message (str): The user message.
            history (list[Message], optional): List of previous messages [{"role": "user"|"assistant", "content": "..."}].
            rag_context (str, optional): Optional RAG context prompt.

        Returns:
            LLMResponse with content and usage stats
        """

        messages = []
        messages.append({
            "role": "system", 
            "content": self.system_prompt
        })

        # Add conversation history
        for m in history:
            messages.append({
                "role": m.role,
                "content": m.content
            })

        # Add RAG context if provided
        if rag_context:
            messages.append({
                "role": "system",
                "content": rag_context
            })

        # Add new user message
        messages.append({
            "role": "user",
            "content": message
        })

        response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.choices:
            raise ValueError("Invalid response from Mistral API")

        return response.choices[0].message.content
        