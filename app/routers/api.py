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
        session_id (str): Session ID
        context_id (str): Context ID

    Returns:
        json: {
            chunks (list[Chunk]): Chunks objects
        }
    """
    print('context_1', flush=True)
    rag_service: RagService = request.app.state.rag_service
    print('context_3', flush=True)
    session = rag_service.llm_handler.get_session(session_id)
    print('context_4', flush=True)
    context = session.get_context(context_id)
    print('context_5', flush=True)

    return {
        'chunks': [
            chunk.model_dump() for chunk in context.chunks
        ]
    }