from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from io import BytesIO

from typing import Literal

from app.services import RagService, PlotService
from app.models import ChatRequest, SessionRequest


router = APIRouter(
    prefix="/api",
    tags=["api"]
)


@router.post("/create_session", summary="Create a chatbot session")
async def create_session(request: Request):
    """
    Create a LLM session

    Args:
        request (Request): Default request argument

    Returns:
        json: {
            session_id (str): The UUID of the created session
        }
    """
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.create_session()

    return {
        'session_id': session.id
    }

@router.post("/clear_session", summary="Clear a chatbot conversation session")
async def clear_session(body: SessionRequest, request: Request):
    """
    Clear a session conversation history

    Args:
        body (SessionRequest): Session payload (session_id)
        request (Request): Default request argument

    Returns:
        json: {
            success (bool): Success confirmation
        }
    """
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.get_session(body.session_id)
    session.clear_messages()

    return {
        'success': True
    }

@router.get("/download_session", summary="Download a conversation in .txt or .json")
async def download_session(
        request: Request, 
        session_id: str, 
        format: Literal["txt", "json"] = "txt"
    ):
    """
    Download a session conversation in .txt or .json format

    Args:
        request (Request): Default request argument
        session_id (str): LLM session ID
        format (str): Format for export ('txt' or 'json')

    Returns:
        StreamingResponse: txt or json file
    """
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.get_session(session_id)

    content = session.dump_messages(format)
    buffer = BytesIO(content.encode("utf-8"))
    buffer.seek(0)

    if format == 'json':
        media_type = "application/json"
        filename = f"session_{session_id}.json"
    else:
        media_type = "text/plain"
        filename = f"session_{session_id}.txt"
        

    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.post("/send", summary="Query the chatbot with RAG")
async def send(body: ChatRequest, request: Request):
    """
    Send a message to LLM with RAG context

    Args:
        body (ChatRequest): Chat message payload (role, content)
        request (Request): Default request argument

    Returns:
        json: {
            response (str): LLM response,
            context: {
                id (str): UUID of the context used for query,
                length (int): amount of documents in context
            }
        }
    """
    rag_service: RagService = request.app.state.rag_service
    response, context = rag_service.make_query(
        body.query,
        session_id=body.session_id
    )

    return {
        'response': response,
        'context': {
            'id': context.id if context else None,
            'length': len(context.chunks) if context else None
        }
    }

@router.get("/get_context", summary="Retrieve RAG chunks from a context ID")
async def get_context(request: Request, session_id: str, context_id: str):
    """
    Retrieve a list of chunks used for RAG from a context ID

    Args:
        request (Request): Default request argument
        session_id (str): LLM session ID
        context_id (str): Context ID

    Returns:
        json: {
            chunks (list[Chunk]): Chunks objects
        }
    """
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.get_session(session_id)
    context = session.get_context(context_id)

    return {
        'chunks': [
            chunk.model_dump() for chunk in context.chunks
        ]
    }

@router.get("/plot_context", summary="Project user query and context chunks in 3d space")
def plot_context(request: Request, session_id: str, context_id: str):
    """
    Create a 3d scatter plot of a RAG context

    Args:
        request (Request): Default request argument
        session_id (str): LLM session ID
        context_id (str): Context ID

    Returns:
        json: {
            3d_scatter (json): Plotly json plot
        }
    """
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.get_session(session_id)
    context = session.get_context(context_id)

    plot_service: PlotService = request.app.state.plot_service
    plot = plot_service.project(
        query=context.query,
        chunks=context.chunks
    )

    return {
        '3d_scatter': plot
    }