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
            session_id: str
            ) -> str:
        """
        Process RAG, set conversation if provided and make a query to the LLM

        Args:
            query (str): User query
            conversation (list[Message], optional): User conversation history. Defaults to None.

        Returns:
            str: LLM response
        """

        # Retrieve LLM session
        session = self.llm_handler.get_session(session_id)

        # Retrieve related document chunks (RAG)
        related_chunks = self.vector_store.search(query)

        # Set LLM history, build a prompt and query
        context = session.build_rag_context(related_chunks)
        response = session.chat(
            query, 
            rag_context=context
        )

        return response

