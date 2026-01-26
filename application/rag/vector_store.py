from application.rag import Vectorizer
from application.db import DocumentsDB
from application.models import Document


class VectorStore:
    
    def __init__(self) -> None:
        self.vectorizer = Vectorizer()
        self.documents_db = DocumentsDB()

    def search(self, query: str, k=5) -> list[Document]:
        """
        Search nearest documents from a query. Embed the query, search for the nearest chunks
        and return the unique documents

        Args:
            query (str): Query to search documents
            k (int, optional): Number of chunks to retrieve. Defaults to 5.

        Returns:
            list[str]: List of document text content
        """

        embeddings = self.vectorizer.generate_embeddings(query)

        with documents_db.connect() as conn:
            chunk_ids = self.documents_db.get_k_nearest(
                embeddings, 
                k=k, 
                conn=conn
            )
            unique_chunk_ids = set(chunk_ids)

            document_contents: list[str] = []
            for doc_id in unique_chunk_ids:
                content = documents_db.get_by_id(doc_id)
                document_contents.append(content)

            return document_contents