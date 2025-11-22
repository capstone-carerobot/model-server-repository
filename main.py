from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
from openai import OpenAI
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 시 특정 도메인으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def dummy_predict(utterance: str):
    # 여기에 실제 모델 예측 로직을 구현
    if "bad" in utterance.lower():
        return "Negative"
    elif "good" in utterance.lower():
        return "Positive"
    else:
        return "Neutral"

# 근거 문장 생성 gpt api
def get_reason(utterance: str, prediction: int, attn: Dict):
    client = OpenAI()
    
    response = client.responses.create(
        model="gpt-5",
        input=f"Explain why the sentiment of the following utterance is {prediction}:\n\n{utterance}\n\nAttention details: {attn}",
    )
    
    return response.output_text

# 정서 상태 예측 gpt api
def get_sentiment_level(utterance: str):
    client = OpenAI()
    
    response = client.responses.create(
        model="gpt-5",
        input=f"Determine the sentiment level of the following utterance (0, 1, 2, 3):\n\n{utterance}",
    )
    
    return response.output_text

@app.post("/internal/analyze")
async def predict(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        utterance = data.get("utterance")

        if utterance is None:
            return {"status": "error", "message": "utterance is required"}

        prediction_emergency = dummy_predict(utterance)
        prediction_sentiment = get_sentiment_level(utterance)
        attn = {}
        reason = get_reason(utterance, prediction_sentiment, attn, prediction_emergency)

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {
        "id": user_id,
        "utterance": utterance,
        "prediction_emergency": prediction_emergency,
        "prediction_sentiment": prediction_sentiment,
        "reason": reason
    }

@app.get("/")
def root():
    return {"message": "FastAPI model server is running!"}