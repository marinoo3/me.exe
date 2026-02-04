from app.rag import VectorStore, LLMHandler
from app.exceptions import UnsafeRequestException
from app.models import Message




class RagService:

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def make_query(
            self, 
            query: str, 
            conversation: list[Message] = []
            ) -> str:
        """
        Process RAG, set conversation if provided and make a query to the LLM

        Args:
            query (str): User query
            conversation (list[Message], optional): User conversation history. Defaults to None.

        Returns:
            str: LLM response
        """

        # Retrieve related document chunks (RAG)
        related_chunks = self.vector_store.search(query)

        # Set LLM history, build a prompt and query
        context = self.llm_handler.build_rag_context(related_chunks)
        response = self.llm_handler.chat(
            query, 
            rag_context=context, 
            history=conversation
        )

        return response

