# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # set this in environment variables
HF_MODEL = "prithivida/parrot_paraphraser_on_T5"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

app = FastAPI(title="Parrot Paraphraser API", version="1.0")

class ParaphraseRequest(BaseModel):
    text: str

def hf_paraphrase(text: str):
    payload = {"inputs": text, "parameters": {"max_new_tokens": 128}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.status_code} | {response.text}")
    data = response.json()
    return data[0]["generated_text"] if isinstance(data, list) else str(data)

@app.post("/paraphrase")
async def paraphrase(req: ParaphraseRequest):
    text = req.text.strip()
    try:
        result = hf_paraphrase(text)
    except Exception as e:
        return {"error": str(e)}
    return {"original": text, "paraphrased": result}

@app.get("/health")
def health():
    return {"status": "API running", "model": HF_MODEL}
