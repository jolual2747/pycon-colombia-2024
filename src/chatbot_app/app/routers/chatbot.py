from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from app.services.chatbot import ChatBot, create_gen
from app.schemas.query import Query
from app.schemas.responses import ChatResponse
from app.services.utils import create_vector_database_from_pdf, AsyncCallbackHandler

chatbot_router = APIRouter()
retriever = create_vector_database_from_pdf("./app/tmp/data.pdf")
chatbot_service = ChatBot(retriever, "PyCon Colombia 2024")
chatbot_service.create_agent()

@chatbot_router.post("/chat", tags = ["chatbot"], response_model = ChatResponse)
def chat(query: Query) -> ChatResponse:
    """Chat with LLM Chatbot.

    Args:
        query (Query): User's query.

    Returns:
        ChatResponse: AI Message.
    """
    answer = chatbot_service.agent({"input": query.text})
    return JSONResponse(content=jsonable_encoder(ChatResponse(ai_answer=answer["output"])))

@chatbot_router.post("/chat_stream", tags = ["chatbot"])
async def chat_stream(query: Query = Body(...)) -> StreamingResponse:
    """Chat with LLM Chatbot with Streaming Response in asynchronous way.

    Args:
        query (Query, optional): User's query. Defaults to Body(...).

    Returns:
        StreamingResponse: Streaming AI Message.
    """
    stream_it = AsyncCallbackHandler()
    gen = create_gen(chatbot_service.agent, query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")