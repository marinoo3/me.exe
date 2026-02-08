from app.database import db
from app.models import Document


class DocumentService:

    def __init__(self) -> None:
        self.document_db = db

    def get_by_ids(self, ids: list[int]) -> list[Document]:
        """
        Retrieve documents by IDs

        Args:
            ids (list[int]): Document IDs

        Returns:
            list[Document]: The list of Documents
        """
        documents: list[Document] = []

        with self.document_db.connect() as conn:
            for doc_id in ids:
                doc = self.document_db.get_document(doc_id, conn)
                if doc:
                    documents.append(doc)

        return documents