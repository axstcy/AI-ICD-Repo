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


def load_raw_bundle():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


RAW_BUNDLE = load_raw_bundle()


def extract_items(bundle):
    if isinstance(bundle, dict):
        if isinstance(bundle.get("items"), list):
            return bundle["items"]
        if isinstance(bundle.get("data"), list):
            return bundle["data"]
        return []
    if isinstance(bundle, list):
        return bundle
    return []


def load_icd_data():
    items = extract_items(RAW_BUNDLE)
    cleaned = []

    for item in items:
        if not isinstance(item, dict):
            continue

        code = (
            item.get("code")
            or item.get("code_no_asterisk")
            or item.get("code_no_dot")
            or ""
        )
        title = (
            item.get("title")
            or item.get("description")
            or item.get("label")
            or ""
        )

        code = str(code).strip()
        title = str(title).strip()

        if code and title:
            cleaned.append(
                {
                    "code": code,
                    "title": title,
                    "chapter_title": str(item.get("chapter_title") or "").strip(),
                    "block_title": str(item.get("block_title") or "").strip(),
                }
            )

    return cleaned


ICD_DATA = load_icd_data()


def score(query, item):
    q = normalize(query)
    title = normalize(item.get("title"))
    code = normalize(item.get("code"))

    total = 0.0

    # Exact match boost
    if q == title:
        total += 10

    # Partial match
    if q in title:
        total += 6

    # Prefer general usable hospital codes (J, A, etc.)
    if code.startswith("j"):
        total += 2

    # Prefer more specific ICD codes (with decimal)
    if "." in code:
        total += 3

    # Penalize neonatal codes slightly
    if code.startswith("p"):
        total -= 1

    # Fuzzy similarity
    total += SequenceMatcher(None, q, title).ratio() * 3.0

    return total



@app.get("/")
def home():
    raw_items = extract_items(RAW_BUNDLE)
    return {
        "status": "ICD API running",
        "raw_items_count": len(raw_items),
        "records_loaded": len(ICD_DATA),
        "raw_sample": raw_items[:2],
        "cleaned_sample": ICD_DATA[:2],
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
