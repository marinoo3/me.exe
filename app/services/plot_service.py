from app.database import db
from app.rag import Vectorizer, Reductor
from app.plots import Scatter3D
from app.models import Chunk



class PlotService:
    
    def __init__(self) -> None:
        self.scatter_3d = Scatter3D()
        self.vectorizer = Vectorizer()
        self.reductor = Reductor()
        self.document_db = db

    def project(self, query: str, chunks: list[Chunk]) -> str:
        """
        Create a 3d scatterplot of query and RAG chunks embeddings

        Args:
            query (str): User LLM query
            chunks (list[Chunk]): List of chunks used for RAG

        Returns:
            str: Plotly json plot
        """
        query_emb = self.vectorizer.generate_embeddings(query)
        query_emb_3d = self.reductor.transform(query_emb)

        with self.document_db.connect() as conn:
            all_chunks = self.document_db.get_chunks(conn)
        
        selected_chunk_ids = [chunk.id for chunk in chunks if chunk.id is not None]

        fig_json = self.scatter_3d.plot(
            chunks=all_chunks,
            query_emb=query_emb_3d[0],
            highlight_ids=selected_chunk_ids
        )

        return fig_json