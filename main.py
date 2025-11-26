from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
from openai import OpenAI
from typing import Dict
from model import RiskModel
from dotenv import load_dotenv
import os

load_dotenv()
# MECAB_DICT_PATH = os.getenv("MECAB_DICT_PATH")
# JAVA_HOME = os.getenv("JAVA_HOME")
GPT_API_KEY = os.getenv("GPT_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 시 특정 도메인으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_LEN = 128
MODEL_PATH = "1118_15epochs"

# 근거 문장 생성 gpt api
def get_reason(utterance: str, prediction_mental: int, prediction_physical: int, word_importance_physical: Dict, word_importance_mental: Dict):
    client = OpenAI()
    
    response = client.responses.create(
        model="gpt-5-nano",
        input=f"""Given the following utterance and explain, provide the reason why the risk levels are respectively {prediction_mental} (mental) and {prediction_physical} (physical) in KOREAN:
        \n\n{utterance}\n\nAttention details (physical): {word_importance_physical}, Attention details (mental): {word_importance_mental}.
        Make sure your explanation is SIMPLY one to two sentences long (up to 50 words), focusing on key words from the attention details.""",
    )
    
    return response.output_text

# 정서 상태 예측 gpt api
def get_mental_level(utterance: str):
    client = OpenAI()
    
    response = client.responses.create(
        model="gpt-5-nano",
        input=f"""Determine the mental risk level of the following utterance (0, 1, or 2). Provide your answer in the following format:
                {{mental risk level: one out of the three, word importance: [(word: importance score), (...), ...]}}.
                Make sure the total word importance score equals 1, and output only the response in this format.
                Note that mental risk is different from physical risk.
                Here is the sentence:{utterance}""",
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


        risk_model = RiskModel(MODEL_PATH, MAX_LEN)
        prediction_physical = risk_model.predict_physical_level(utterance)
        utterance_decoded = risk_model.decode_input(utterance)
        word_importance_physical = risk_model.explain_text(utterance_decoded, filter_pos=True)
        
        
        response_mental = get_mental_level(utterance)
        print(response_mental)
        prediction_mental, word_importance_mental = risk_model.parse_mental_level_response(response_mental)
        
        reason = get_reason(utterance, prediction_mental, prediction_physical, word_importance_physical, word_importance_mental)

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {
        "id": user_id,
        "utterance": utterance,
        "prediction_physical": prediction_physical,
        "prediction_mental": prediction_mental,
        "word_importance_mental": word_importance_mental,
        "word_importance_physical": word_importance_physical,
        "reason": reason
    }

@app.get("/")
def root():
    return {"message": "FastAPI model server is running!"}