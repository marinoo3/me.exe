import os
from typing import Generator

from app.models import Document, Chunk
from app.database import db
from app.rag import Vectorizer
from app.parser import ParsePDF, ParseMD


def load_documents(path: str) -> Generator:
    """
    Load documents from data/files

    Args:
        path (str): files directory (where the documents are stored)

    Yield:
        tuple[Document, str]: Tuple of document and its path
    """
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        for _, _, files in os.walk(category_path):
            for f in files:
                document = Document(
                    name = ''.join(f.split('.')[:-1]), #type: ignore
                    category = category
                )
                yield (document, os.path.join(category_path, f))


def parse_content(path: str) -> str:
    """
    Load a file and parse its content

    Args:
        path (str): The path of the file to parse

    Returns:
        str: the file content
    """
    ext = path.split('.')[-1]
    match ext:
        case 'txt':
            with open(path, 'r') as f:
                return f.read()
        case 'pdf':
            return ParsePDF(path)
        case 'md':
            return ParseMD.from_path(path)
        case _:
            raise ValueError(f"Impossible to parse '{ext}' files ({path})")



if __name__ == '__main__':
    document_db = db
    vectorizer = Vectorizer()

    with document_db.connect() as conn:
        for document, path in load_documents('data/files'):
            print(path)
            # Create document in DB
            doc_id = document_db.add_document(document, conn=conn)
            # Read and chunk its content
            document_content = parse_content(path)
            chunks_content = vectorizer.chunk_text(document_content)
            # Vectorize and create the chunks in DB
            for content in chunks_content:
                emb_384d = vectorizer.generate_embeddings(content)
                chunk = Chunk(
                    document_id = doc_id,
                    content = content,
                    emb_384d = emb_384d
                )
                document_db.add_chunk(chunk, conn=conn)
