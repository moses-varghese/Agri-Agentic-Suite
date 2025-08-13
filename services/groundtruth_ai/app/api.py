# from fastapi import APIRouter, Form
# from twilio.twiml.messaging_response import MessagingResponse
# from .services.rag_service import rag_service

# router = APIRouter()

# @router.post("/sms", tags=["SMS"])
# async def handle_sms(From: str = Form(...), Body: str = Form(...)):
#     """
#     Endpoint to handle incoming SMS from Twilio.
#     """
#     print(f"📲 Received SMS from {From}: '{Body}'")
    
#     # Get the AI-powered response
#     ai_response = await rag_service.get_response(Body)
    
#     # Create a TwiML response
#     twiml_response = MessagingResponse()
#     twiml_response.message(ai_response)
    
#     return str(twiml_response)



from fastapi import APIRouter
from pydantic import BaseModel
from .services.rag_service import rag_service

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@router.post("/groundtruth", response_model=QueryResponse, tags=["RAG Query"])
async def handle_query(request: QueryRequest):
    """
    Endpoint to handle an incoming query. Replaces the /sms endpoint.
    """
    print(f"❓ Received query: '{request.query}'")
    ai_response = await rag_service.get_response(request.query)
    return QueryResponse(answer=ai_response)