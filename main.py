import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_TOKEN")
)

app = FastAPI()

BAD_LABELS = ["toxic", "obscene", "insult", "threat", "identity_hate"]
THRESHOLD = 0.7


class CommentRequest(BaseModel):
    text: str


@app.get("/ping")
def ping():
    return {"message": "healthy"}  # For uptime monitoring


@app.post("/moderate")
def moderate_comment(req: CommentRequest):
    result = client.text_classification(
        req.text,
        model="unitary/toxic-bert"
    )

    for item in result:
        if item.label in BAD_LABELS and item.score >= THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail="Your comment contains inappropriate content"
            )

    return {"message": "OK"}
