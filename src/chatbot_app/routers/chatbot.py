from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from chatbot_app.services.chatbot import ChatBot, create_gen
from chatbot_app.schemas.query import Query
from chatbot_app.schemas.responses import ChatResponse
from chatbot_app.services.utils import create_vector_database_from_pdf, AsyncCallbackHandler

chatbot_router = APIRouter()
retriever = create_vector_database_from_pdf("./chatbot_app/tmp/data.pdf")
chatbot_service = ChatBot(retriever, "Tuya S.A")
chatbot_service.create_agent()

@chatbot_router.post("/chat", tags = ["chatbot"], response_model = ChatResponse)
def chat(query: Query) -> ChatResponse:
    answer = chatbot_service.agent({"input": query.text})
    print(answer.keys(  ))
    return JSONResponse(content=jsonable_encoder(ChatResponse(ai_answer=answer["output"])))

@chatbot_router.post("/chat_stream", tags = ["chatbot"])
async def chat_stream(query: Query = Body(...)) -> StreamingResponse:
    stream_it = AsyncCallbackHandler()
    gen = create_gen(chatbot_service.agent, query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")