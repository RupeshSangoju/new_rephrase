from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

# -----------------------------
# HuggingFace API setup
# -----------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # set this in Render's environment variables
HF_MODEL = "prithivida/parrot_paraphraser_on_T5"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Parrot Paraphraser API (HF)",
    description="Paraphrase text using HuggingFace Inference API",
    version="1.0.0"
)

# -----------------------------
# Request schema
# -----------------------------
class ParaphraseRequest(BaseModel):
    text: str

# -----------------------------
# HuggingFace paraphrasing function
# -----------------------------
def hf_paraphrase(text: str):
    payload = {"inputs": text, "parameters": {"max_new_tokens": 128}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.status_code} | {response.text}")
    data = response.json()
    # HuggingFace model returns a list of dicts with 'generated_text'
    return data[0]["generated_text"] if isinstance(data, list) else str(data)

# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.post("/paraphrase")
async def paraphrase(request: ParaphraseRequest):
    input_text = request.text.strip()
    try:
        paraphrased_text = hf_paraphrase(input_text)
    except Exception as e:
        return {"error": str(e)}
    return {
        "original": input_text,
        "paraphrased": paraphrased_text
    }

@app.get("/health")
def health():
    return {"status": "API is running", "model": HF_MODEL}
