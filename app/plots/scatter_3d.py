from app.models import Chunk
from app.plots.base_plot import BasePlot

import numpy as np
import plotly.graph_objects as go


class Scatter3D(BasePlot):

    def plot(
            self, 
            chunks: list[Chunk], 
            query_emb: np.ndarray, 
            highlight_ids: list[int]
        ) -> str:
        """
        Create a 3d scatterplot of query and RAG chunks embeddings

        Args:
            chunks (list[Chunk]): Chunks to project
            query_emb (np.ndarray): Query embeddings to project
            highlight_ids (list[int]): Chunk IDs to highlight

        Returns:
            str: JSON formated plotly object
        """
        xs = [chunk.emb_3d[0] for chunk in chunks]
        ys = [chunk.emb_3d[1] for chunk in chunks]
        zs = [chunk.emb_3d[2] for chunk in chunks]

        colors = ["#00ff00" if chunk.id in highlight_ids else "lightgray" for chunk in chunks]

        fig = go.Figure()

        # All chunks
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=colors,
                    line=dict(width=1, color="black")
                ),
                text=[f"chunk_id: {chunk.id}" for chunk in chunks],
                name="Chunks"
            )
        )

        # Query point
        fig.add_trace(
            go.Scatter3d(
                x=[query_emb[0]],
                y=[query_emb[1]],
                z=[query_emb[2]],
                mode="markers",
                marker=dict(
                    size=2.5, 
                    color="#0000ff", 
                    line=dict(width=1, color="black"),
                    symbol="diamond"),
                name="Query"
            )
        )

        fig.update_layout(
            width=62.5,
            height=62.5,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            hovermode=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),  # JS uses paper_color; in Python set font color
            scene=dict(
                bgcolor="black",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
                xaxis=dict(
                    title="",
                    showgrid=True,
                    gridcolor="#6f6f6f",
                    gridwidth=1,
                    backgroundcolor="black",
                    spikecolor="#00ff00",
                    spikethickness=1,
                    zeroline=False,
                    tickfont=dict(size=6),
                ),
                yaxis=dict(
                    title="",
                    showgrid=True,
                    backgroundcolor="black",
                    gridcolor="#6f6f6f",
                    gridwidth=1,
                    spikecolor="#00ff00",
                    spikethickness=1,
                    zeroline=False,
                    tickfont=dict(size=6),
                ),
                zaxis=dict(
                    title="",
                    showgrid=True,
                    gridcolor="#6f6f6f",
                    gridwidth=1,
                    backgroundcolor="black",
                    spikecolor="#00ff00",
                    spikethickness=1,
                    zeroline=False,
                    tickfont=dict(size=6),
                ),
            )
        )

        return self._get_json(fig)
