# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from whistle_wise import build_chain

import traceback


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = build_chain()

from pydantic import BaseModel
from typing import List, Tuple

class Query(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []


@app.post("/ask")
def ask_question(query: Query):
    try:
        response = qa_chain.invoke({
            "input": query.question,
            "chat_history": query.chat_history
        })
        return {"response": response}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
