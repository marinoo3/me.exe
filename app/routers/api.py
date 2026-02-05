from fastapi import APIRouter, Request

from app.services import RagService
from app.models import ChatRequest


router = APIRouter(
    prefix="/api",
    tags=["api"]
)

@router.post("/create_session", summary="Create a chatbot session")
async def create_session(request: Request):
    rag_service: RagService = request.app.state.rag_service
    session = rag_service.llm_handler.create_session()
    return session.id

@router.post("/send", summary="Query the chatbot")
async def send(body: ChatRequest, request: Request):
    # Access body data
    session_id = body.session_id
    query = body.query
    # Process RAG and query LLM
    rag_service: RagService = request.app.state.rag_service
    response = rag_service.make_query(
        query,
        session_id=session_id
    )
    # Send response
    return response


@router.get("/test")
def test():
    return "Hello world"