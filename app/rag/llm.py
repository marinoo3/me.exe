from typing import Optional
from mistralai import Mistral
import uuid

from app.exceptions import MissingAPIKeyError
from app.models import Chunk, Message
from app.config import settings





class LLMSession:
    """
    Session handler for Mistral LLM interactions.

    Manages model selection, prompt templates and conversation history.
    """

    model_name: str
    id: str
    temperature = .5
    max_tokens = 5000
    messages = []

    def __init__(self, model: str = "mistral-medium-latest"):
        """
        Initialize the LLM handler.

        Args:
            model: Model key from MODELS dict.
        """
        self.id = str(uuid.uuid4())
        self.api_key = settings.MISTRAL_API_KEY.get_secret_value()
        self.system_prompt = settings.SYSTEM_PROMPT

        if not self.api_key:
            raise MissingAPIKeyError(
                "MISTRAL_API_KEY not found in environment variables"
            )

        self._client = None
        self.model_name = model
        self.messages.append(
            Message(role="system", content=self.system_prompt).model_dump()
        )

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

        # Add RAG context if provided
        if rag_context:
            self.messages.append(
                Message(role="system", content=rag_context).model_dump()
            )

        # Add new user message
        self.messages.append(
            Message(role="user", content=message).model_dump()
        )

        response = self.client.chat.complete(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.choices:
            raise ValueError("Invalid response from Mistral API")
        
        print(self.messages)

        return response.choices[0].message.content #type: ignore
        

class LLMHandler:
    """
    Manage LLMHandler sessions
    """

    sessions: list[LLMSession] = []

    def get_session(self, id: str) -> LLMSession:
        """
        Retrieve a LLMSession from its uuid

        Args:
            id (int): session uuid
        """

        for session in self.sessions:
            if session.id == id:
                return session
            
        raise IndexError(f"No session found with uuid {uuid}")
    
    def create_session(self) -> LLMSession:
        """
        Create and store a LLMSession

        Returns:
            LLMHandler: The LLMSession instance
        """

        session = LLMSession()
        self.sessions.append(session)
        return session