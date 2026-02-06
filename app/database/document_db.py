import os
import numpy as np

from app.models import Document, Chunk

import sqlite3
import sqlite_vec



class DocumentDB:

    path = 'db/document.db'

    def __init__(self) -> None:
        root = os.environ.get('DATA_PATH', 'data/')
        self.db_path = os.path.join(root, self.path)

    @staticmethod
    def _vecf32_converter(blob:bytes) -> np.ndarray:
        """
        Convert blob to vector

        Args:
            blob (bytes): Blob to convert

        Returns:
            np.ndarray: f32 vector
        """

        return np.frombuffer(blob, dtype=np.float32).copy()
    
    def connect(self) -> sqlite3.Connection:
        """
        Create a connection to the database

        Returns:
            sqlite3.Connection: DB connection
        """

        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn) 
        return conn
    
    # --------- GET methods
    
    def get_k_nearest(
                self,
                embedding: np.ndarray,
                k: int,
                conn: sqlite3.Connection
            ) -> list[Chunk]:
        """
        Get the k nearest document chunks from a given embedding

        Args:
            embeddings (np.ndarray): Embedding to compute cosinus distance with
            k (int): Number of chunks to retrieve
            conn (sqlite3.Connection): DB conneciton

        Return:
            list[Chunk]: The list of nearest chunks
        """

        embedding_blob = embedding.astype("float32").tobytes()

        with conn:
            conn.row_factory = sqlite3.Row

            query = f"""
            SELECT
                c.document_id,
                c.content,
                c.emb_384d,
                vec_distance_cosine(c.emb_384d, vec_f32(?)) AS query_distance,
                d.name as source_name,
                d.category as source_category
            FROM Chunk AS c
            JOIN Document AS d ON c.document_id = d.id
            ORDER BY query_distance
            LIMIT {k}
            """

            rows = conn.execute(query, (embedding_blob,))

            chunks = [
                Chunk(
                    document_id=row["document_id"],
                    content=row["content"],
                    emb_384d=np.frombuffer(row["emb_384d"], dtype=np.float32).copy(),
                    source_name=row["source_name"],
                    source_categorie=row["source_category"]
                )
                for row in rows
            ]

            return chunks
        
    # --------- UPDATE methods

    def add_document(self, document: Document, conn: sqlite3.Connection) -> int:
        """
        Write a new document in the database

        Args:
            document (Document): Document to add

        Returns:
            int: The created document ID
        """

        with conn:
            cur = conn.execute(
                """
                INSERT INTO Document (name, category)
                VALUES (?, ?)
                """,
                (document.name, document.category)
            )

            if cur.lastrowid is None:
                raise ValueError(f"Failed to insert {document} into Document table")

            return cur.lastrowid

    def add_chunk(self, chunk: Chunk, conn: sqlite3.Connection) -> int:
        """
        Write a list of chunks in the database

        Args:
            chunks Chunk: List of chunks to add

        Return:
         int: The created chunk ID
        """

        with conn:
            cur = conn.execute(
                """
                INSERT INTO Chunk (emb_384d, content, document_id)
                VALUES (?, ?, ?)
                """,
                (chunk.emb_384d, chunk.content, chunk.document_id)
            )

            if cur.lastrowid is None:
                raise ValueError(f"Failed to insert {chunk} into Chunk table")

            return cur.lastrowid


db = DocumentDB()