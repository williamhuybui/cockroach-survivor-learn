import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    text: str


@app.post("/chat")
def send_message(message: Message):
    payload = {
        "model": "gemma3",
        "messages": [
            {
                "role": "user",
                "content": message.text
            }
        ],
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return {"message": response.json()["message"]["content"]}