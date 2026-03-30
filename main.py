from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from difflib import SequenceMatcher
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load WHO dataset
with open("icd10_who_2019_clean.json", "r", encoding="utf-8") as f:
    ICD_DATA = json.load(f)["items"]


class Query(BaseModel):
    text: str


def normalize(text):
    return text.lower().strip()


def score(query, item):
    q = normalize(query)
    title = normalize(item["title"])

    score = 0

    if q in title:
        score += 5

    score += SequenceMatcher(None, q, title).ratio() * 3

    return score


@app.get("/")
def home():
    return {
        "status": "ICD API running",
        "records": len(ICD_DATA)
    }


@app.post("/predict")
def predict(q: Query):
    ranked = sorted(
        ICD_DATA,
        key=lambda x: score(q.text, x),
        reverse=True
    )

    best = ranked[0]

    return {
        "suggestions": [
            {
                "code": best["code"],
                "description": best["title"]
            }
        ]
    }
