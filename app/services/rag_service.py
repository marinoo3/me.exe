from app.rag import VectorStore, LLMHandler
from app.exceptions import UnsafeRequestException
from app.models import Document




class RagService:

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()

    def make_query(
            self, 
            query: str, 
            session_id: str
            ) -> tuple[str, set[Document]]:
        """
        Process RAG, set conversation if provided and make a query to the LLM

        Args:
            query (str): User query
            session_id (str): UUID of the chat bot session

        Returns:
            str: LLM response
            set[Document]: Unique documents used for RAG
        """

        # Retrieve LLM session
        session = self.llm_handler.get_session(session_id)

        # Retrieve related document chunks (RAG)
        related_chunks = self.vector_store.search(query)
        sources = {
            Document(
                id = chunk.document_id,
                name = chunk.source_name or "",
                category = chunk.source_categorie or ""
            ) for chunk in related_chunks
        }

        # Build RAG prompt from chunks
        context = ""
        if related_chunks:
            context = session.build_rag_context(related_chunks)

        # Query LLM
        response = session.chat(
            query, 
            rag_context=context
        )

        return response, sources

