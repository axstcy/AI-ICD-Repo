from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from difflib import SequenceMatcher
import json
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path("icd10_who_2019_clean.json")


class Query(BaseModel):
    text: str


def normalize(text):
    return (text or "").lower().strip()


def load_icd_data():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    items = bundle.get("items", [])

    cleaned = []
    for item in items:
        code = item.get("code")
        title = item.get("title")

        if code and title:
            cleaned.append(
                {
                    "code": code,
                    "title": title,
                    "chapter_title": item.get("chapter_title") or "",
                    "block_title": item.get("block_title") or "",
                    "is_terminal": item.get("is_terminal", False),
                }
            )

    return cleaned


ICD_DATA = load_icd_data()


def score(query, item):
    q = normalize(query)
    title = normalize(item.get("title"))
    chapter = normalize(item.get("chapter_title"))
    block = normalize(item.get("block_title"))
    code = normalize(item.get("code"))

    total = 0.0

    if q in title:
        total += 5.0
    if q in block:
        total += 1.5
    if q == code:
        total += 10.0

    total += SequenceMatcher(None, q, title).ratio() * 3.0
    return total


@app.get("/")
def home():
    return {
        "status": "ICD API running",
        "records_loaded": len(ICD_DATA),
    }


@app.post("/predict")
def predict(q: Query):
    query_text = normalize(q.text)

    if not query_text:
        return {"suggestions": []}

    ranked = sorted(
        ICD_DATA,
        key=lambda item: score(query_text, item),
        reverse=True,
    )

    best = ranked[:1]

    return {
        "suggestions": [
            {
                "code": item["code"],
                "description": item["title"],
            }
            for item in best
        ]
    }
