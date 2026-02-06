from fastapi import APIRouter, Request

from app.services import RagService
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

@router.post("/clear_session", summary="Clear a chatbot session conversation")
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
    rag_service.llm_handler.clear_session(body.session_id)

    return {
        'success': True
    }

@router.post("/send", summary="Query the chatbot")
async def send(body: ChatRequest, request: Request):
    """
    Send a message to LLM with RAG context

    Args:
        body (ChatRequest): Chat message payload (role, content)
        request (Request): Default request argument

    Returns:
        json: {
            response (str): LLM response
        }
    """
    rag_service: RagService = request.app.state.rag_service
    response = rag_service.make_query(
        body.query,
        session_id=body.session_id
    )

    return {
        'response': response
    }