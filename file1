from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')

ICD_DATA = [
    {"code": "J18.9", "description": "Pneumonia, unspecified organism"},
    {"code": "J18.0", "description": "Bronchopneumonia, unspecified"},
    {"code": "J20.9", "description": "Acute bronchitis"},
    {"code": "J45.909", "description": "Asthma"}
]

descriptions = [item["description"] for item in ICD_DATA]
embeddings = model.encode(descriptions, convert_to_tensor=True)

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ICD API running"}

@app.post("/predict")
def predict_icd(query: Query):
    query_embedding = model.encode(query.text, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_result = scores.argmax()

    return {
        "suggestions": [
            {
                "code": ICD_DATA[top_result]["code"],
                "description": ICD_DATA[top_result]["description"],
                "confidence": float(scores[top_result])
            }
        ]
    }
