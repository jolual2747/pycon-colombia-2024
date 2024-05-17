from fastapi import FastAPI
from chatbot_app.routers.chatbot import chatbot_router
from chatbot_app.services.utils import create_vector_database_from_pdf

app = FastAPI(
    title="Chatbot Streaming",
    version="0.1.0",
)

# retriever = create_vector_database_from_pdf("./chatbot_app/tmp/data.pdf")

app.include_router(chatbot_router)

@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}