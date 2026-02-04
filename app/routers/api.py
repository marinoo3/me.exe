from fastapi import APIRouter, Request

from app.services import RagService
from app.models import ChatRequest


router = APIRouter()




@router.get("ask", summary="Query my chatbot")
async def ask(body: ChatRequest, request: Request):
    # Access body data
    conversation = body.conversation
    query = body.query
    # Process RAG and query LLM
    rag_service: RagService = request.app.state.rag_service
    response = rag_service.make_query(
        query,
        conversation=conversation
    )
    # Send response
    return response