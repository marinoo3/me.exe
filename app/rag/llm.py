from typing import Optional
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage, Messages
import uuid
import json

from app.parser import ParseMD
from app.exceptions import MissingAPIKeyError
from app.models import Chunk, Context
from app.config import settings





class LLMSession:
    """
    Session handler for Mistral LLM interactions.

    Manages model selection, prompt templates and conversation history.
    """

    id: str
    model_name: str
    temperature = .5
    max_tokens = 5000
    messages: list[Messages] = []
    contexts: dict[str, Context] = {}

    def __init__(self, model: str = "mistral-small-latest"):
        """
        Initialize the LLM handler.

        Args:
            model: Mistral model name. Default to 'mistral-small-latest'
        """
        self.id = str(uuid.uuid4())
        self._api_key = settings.MISTRAL_API_KEY.get_secret_value()
        self.system_prompt = settings.SYSTEM_PROMPT

        if not self._api_key:
            raise MissingAPIKeyError(
                "MISTRAL_API_KEY not found in environment variables"
            )

        self._client = None
        self.model_name = model
        self.__init_messages()

    @property
    def client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            self._client = Mistral(api_key=self._api_key)

        return self._client
    
    def __init_messages(self) -> None:
        """
        Create (or erase) the list of message with the initial system prompt
        """
        self.messages = [
            SystemMessage(content=self.system_prompt)
        ]
    
    def build_context(self, query: str, chunks: list[Chunk], store=True) -> Context:
        """
        Create a RAG context / prompt from a list of chunk. Store the context chunks in history

        Args:
            chunks (list[Chunk]): The chunks to build a context from
            query (str): User message query
            store (bool, optional): To store or not the chunks in history. Default to True

        Returns:
            Context: Context object
        """
        context = Context(
            query=query,
            chunks=chunks
        )

        if store:
            self.contexts[context.id] = context
        
        return context
    
    def get_context(self, id: str) -> Context:
        """
        Retrieve a context from its UUID

        Args:
            id (str): Context UUID to retrive

        Returns:
            Context: Context object
        """
        context = self.contexts.get(id)
        if not context:
            raise IndexError(f"No context found with uuid '{id}'")
        
        return context

    def send_message(
        self,
        message: str,
        context: Optional[Context] = None
    ) -> str:
        """
        Generate a response in a conversation context with history.

        Args:
            message (str): The user message.
            history (list[Message], optional): List of previous messages [{"role": "user"|"assistant", "content": "..."}].
            context (str, optional): Optional RAG context prompt.

        Returns:
            LLMResponse with content and usage stats
        """
        # Add RAG context if provided
        if context:
            self.messages.append(
                SystemMessage(content=context.context)
            )

        # Add new user message
        self.messages.append(
            UserMessage(content=message)
        )

        response = self.client.chat.complete(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if not response.choices:
            raise ValueError("Invalid response from Mistral API")
        
        # Add LLM response to conversation
        self.messages.append(
            response.choices[0].message
        )
        
        # Parse Markdown to text and remove emojis
        text_response = ParseMD.from_string(
            str(response.choices[0].message.content),
            remove_emojis=True
        )

        return text_response
    
    def dump_messages(self, format='txt') -> str:
        """
        Format message history to a string or json for export

        Arguments:
            format (str): Which format to use for export ('txt' or 'json')

        Returns:
            str: Dumped messages
        """
        match format:
            case 'txt':
                dumped = [
                    f"------------ {message.role}\n{message.content}" for message in self.messages
                ]
                return '\n\n'.join(dumped)
            case 'json':
                dumped = [
                    message.model_dump() for message in self.messages
                ]
                return json.dumps(dumped, indent=3)
            case _:
                raise ValueError(f"Invalid format parameter '{format}', expected 'str' or 'json'")

    def clear_messages(self) -> None:
        """
        Reset the message history
        """
        self.__init_messages()
        

class LLMHandler:
    """
    Manage LLMHandler sessions
    """

    sessions: dict[str, LLMSession] = {}

    def get_session(self, id: str) -> LLMSession:
        """
        Retrieve a LLMSession from its uuid

        Args:
            id (int): Session uuid
        """
        session = self.sessions.get(id)
        if not session:
            raise IndexError(f"No session found with uuid {id}")
        
        return session
    
    def create_session(self) -> LLMSession:
        """
        Create and store a LLMSession

        Returns:
            LLMHandler: The LLMSession instance
        """
        session = LLMSession()
        self.sessions[session.id] = session
        return session
    
    def delete_session(self, id: str) -> None:
        """
        Delete a LLMSession instance

        Args:
            id (int): Session uuid
        """
        session = self.get_session(id)
        self.sessions.pop(session.id)