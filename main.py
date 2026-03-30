from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict
import json
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path("icd10_who_2019_clean.json")

# WHO-compliant input synonyms only
SYNONYMS = {
    "uti": "urinary tract infection",
    "urti": "upper respiratory tract infection",
    "age": "acute gastroenteritis",
    "pna": "pneumonia",
}


class Query(BaseModel):
    text: str


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace(",", " ")
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_synonyms(text: str) -> str:
    t = normalize(text)
    return SYNONYMS.get(t, t)


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t]


def extract_items(bundle):
    if isinstance(bundle, dict):
        if isinstance(bundle.get("items"), list):
            return bundle["items"]
        if isinstance(bundle.get("data"), list):
            return bundle["data"]
    elif isinstance(bundle, list):
        return bundle
    return []


def choose_best_description(item: dict) -> str:
    """
    Prefer fuller text fields when available.
    Keep this WHO-based only.
    """
    title = str(item.get("title") or "").strip()

    # Some extracted files may contain useful alternate fields.
    description = str(item.get("description") or "").strip()
    label = str(item.get("label") or "").strip()

    candidates = [description, label, title]
    candidates = [c for c in candidates if c]

    if not candidates:
        return ""

    return max(candidates, key=len)


def load_icd_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found in project root")

    with DATA_FILE.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    raw_items = extract_items(bundle)
    cleaned = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        code = str(item.get("code") or "").strip()
        description = choose_best_description(item)

        if not code or not description:
            continue

        norm_desc = normalize(description)

        cleaned.append(
            {
                "code": code,
                "description": description,
                "normalized_description": norm_desc,
                "tokens": set(tokenize(description)),
                "title": str(item.get("title") or "").strip(),
                "chapter_title": str(item.get("chapter_title") or "").strip(),
                "block_title": str(item.get("block_title") or "").strip(),
                "is_terminal": bool(item.get("is_terminal", False)),
            }
        )

    return cleaned


ICD_DATA = load_icd_data()

# Exact phrase index
EXACT_MAP = {}
for item in ICD_DATA:
    EXACT_MAP[item["normalized_description"]] = item

# Token index for faster candidate search
TOKEN_INDEX = defaultdict(set)
for idx, item in enumerate(ICD_DATA):
    for token in item["tokens"]:
        TOKEN_INDEX[token].add(idx)


def score(query: str, item: dict) -> float:
    q = normalize(query)
    desc = normalize(item.get("description"))
    code = normalize(item.get("code"))

    total = 0.0

    q_words = set(q.split())
    desc_words = set(desc.split())

    # Exact and partial matches
    if q == desc:
        total += 30
    if q in desc:
        total += 12

    # Word overlap
    total += len(q_words & desc_words) * 5

    # Fuzzy similarity
    total += SequenceMatcher(None, q, desc).ratio() * 6.0

    # Prefer more specific WHO codes
    if "." in code:
        total += 3

    # Prefer terminal WHO entries
    if item.get("is_terminal"):
        total += 2

    # Explicit unspecified preference
    if "unspecified" in q:
        if "unspecified" in desc:
            total += 12
        else:
            total -= 5

    # Penalize obviously wrong disease families
    if "gastroenteritis" in q and "bronchitis" in desc:
        total -= 20
    if "bronchitis" in q and "gastroenteritis" in desc:
        total -= 20

    # Pneumonia tuning
    if "pneumonia" in q:
        if "unspecified" in desc or "organism unspecified" in desc:
            total += 8

        subtype_words = {
            "congenital",
            "neonatal",
            "aspiration",
            "lobar",
            "hypostatic",
            "bronchopneumonia",
            "viral",
            "bacterial",
        }

        if any(word in desc for word in subtype_words):
            total -= 4

    return total


def get_candidates(query: str):
    q = normalize(query)

    # Exact phrase first
    if q in EXACT_MAP:
        return [EXACT_MAP[q]]

    # Token-based narrowing
    q_tokens = tokenize(query)
    candidate_ids = set()

    for token in q_tokens:
        candidate_ids.update(TOKEN_INDEX.get(token, set()))

    if candidate_ids:
        return [ICD_DATA[i] for i in candidate_ids]

    # Fallback to full dataset
    return ICD_DATA


@app.get("/")
def home():
    return {
        "status": "ICD API running",
        "records_loaded": len(ICD_DATA),
        "indexed_tokens": len(TOKEN_INDEX),
        "sample": ICD_DATA[:3],
    }


@app.post("/predict")
def predict(q: Query):
    query_text = apply_synonyms(q.text)

    if not query_text:
        return {"suggestions": []}

    candidates = get_candidates(query_text)
    ranked = sorted(
        candidates,
        key=lambda item: score(query_text, item),
        reverse=True,
    )

    best = ranked[:1]

    return {
        "suggestions": [
            {
                "code": item["code"],
                "description": item["description"],
            }
            for item in best
        ]
    }
