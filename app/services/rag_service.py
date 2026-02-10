from app.rag import VectorStore, LLMHandler
from app.exceptions import UnsafeRequestException
from app.models import Context




class RagService:

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def make_query(
            self, 
            query: str, 
            session_id: str
            ) -> tuple[str, Context|None]:
        """
        Process RAG, set conversation if provided and make a query to the LLM

        Args:
            query (str): User query
            session_id (str): UUID of the chat bot session

        Returns:
            str: LLM response
            Context|None: Context object used for query
        """
        # Retrieve LLM session
        session = self.llm_handler.get_session(session_id)

        # Retrieve related document chunks (RAG)
        related_chunks = self.vector_store.search(query)

        # Build RAG prompt from chunks
        context = None
        if related_chunks:
            context = session.build_context(related_chunks)

        # Query LLM
        response = session.send_message(
            query, 
            context=context
        )

        return response, context

