from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

with open("icd10_who_2019_clean.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

items = raw["items"]
codes = [x for x in items if x.get("is_terminal") is True]

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

SYNONYMS = {
    "uti": "urinary tract infection",
    "pna": "pneumonia",
    "ptb": "pulmonary tuberculosis",
    "tb": "tuberculosis",
    "hpn": "hypertension",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
    "age": "gastroenteritis",
    "urti": "upper respiratory tract infection",
}

def normalize(text: str):
    text = text.lower().strip()
    for short, full in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)
    return text

for item in codes:
    desc = item.get("code_no_dot") or item.get("title") or ""
    item["description"] = desc
    item["tokens"] = set(tokenize(desc))

def score(query_tokens, item_tokens):
    overlap = len(query_tokens & item_tokens)
    if overlap == 0:
        return 0.0

    precision = overlap / max(len(query_tokens), 1)
    recall = overlap / max(len(item_tokens), 1)

    exact_bonus = 0.0
    if query_tokens == item_tokens:
        exact_bonus = 0.5

    unspecified_penalty = 0.0
    if "unspecified" in item_tokens and "unspecified" not in query_tokens:
        unspecified_penalty = 0.08

    return (precision * 0.7) + (recall * 0.3) + exact_bonus - unspecified_penalty

class Query(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "status": "ICD API running",
        "records_loaded": len(codes)
    }

@app.post("/predict")
def predict(q: Query):
    text = normalize(q.text)
    q_tokens = set(tokenize(text))

    results = []
    for item in codes:
        s = score(q_tokens, item["tokens"])
        if s > 0.2:
            results.append({
                "code": item["code"],
                "description": item["description"],
                "score": round(s, 3)
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": q.text,
        "normalized_query": text,
        "suggestions": results[:5]
    }
