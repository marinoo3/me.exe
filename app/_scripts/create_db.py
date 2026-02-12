import sqlean as sqlite3
import sqlite_vec

connection = sqlite3.connect("data/db/document.db")
connection.enable_load_extension(True)
sqlite_vec.load(connection)
connection.execute("PRAGMA foreign_keys = ON;")
cursor = connection.cursor()


schema_statements = [
    """
    CREATE TABLE IF NOT EXISTS Document (
        id          INTEGER PRIMARY KEY,
        name        TEXT,
        category    TEXT,
        url         TEXT
    );
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS Chunk
    USING vec0(
        emb_384d        FLOAT[384],
        emb_3d          FLOAT[3],
        content         TEXT,
        document_id     INTEGER
    );
    """
]


for stmt in schema_statements:
    cursor.execute(stmt)

connection.commit()
connection.close()