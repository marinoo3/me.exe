from app.rag import Vectorizer, Reranker
from app.database import db
from app.models import Chunk


class VectorStore:
    
    def __init__(self) -> None:
        self.vectorizer = Vectorizer()
        self.reranker = Reranker()
        self.document_db = db

    def search(self, query: str, k=10) -> list[Chunk]:
        """
        Search nearest documents from a query. Embed the query, search for the nearest chunks
        and return the chunks

        Args:
            query (str): Query to search documents
            k (int, optional): Number of chunks to retrieve. Defaults to 5.

        Returns:
            list[Chunk]: List of document chunk
        """
        
        embeddings = self.vectorizer.generate_embeddings(query)

        with self.document_db.connect() as conn:
            chunks = self.document_db.get_k_nearest(
                embeddings, 
                k=k, 
                conn=conn
            )

        reranked_chunks = self.reranker.rerank(query, chunks)

        return reranked_chunks