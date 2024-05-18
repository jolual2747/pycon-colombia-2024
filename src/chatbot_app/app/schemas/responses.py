from pydantic import BaseModel

class ChatResponse(BaseModel):
    ai_answer: str