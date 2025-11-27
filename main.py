from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
from openai import OpenAI
from typing import Dict
from model import RiskModel
from dotenv import load_dotenv
import os
import json

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

# 근거 문장 생성 gpt api physical
def get_reason(utterance: str, prediction_physical: int, prediction_mental: int,  word_importance_physical: Dict ,word_importance_mental: Dict):
    
    # important_words_physical = ', '.join([list(item.keys())[0] for item in word_importance_physical])
    
    reason = f"""[멘탈 위험도]:\n멘탈 위험도는 {prediction_mental}입니다. 판단에 중요하게 작용한 단어는 {word_importance_mental}입니다.
    \n [신체 위험도]:\n신체 위험도는 {prediction_physical}입니다. 판단에 중요하게 작용한 단어는 {word_importance_physical}입니다."""
    
    return reason




# 정서 상태 예측 gpt api
def get_mental_level(utterance: str):
    client = OpenAI()
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
## Emotional Risk Classification Task

You are an expert mental-health conversation classifier.
Your task is to classify the *psychological or emotional risk level* of a Korean user's input.

---

# 1) abeling Rules (Meaning of Each Label)

## 🔹 Label 1 = Low Risk
Normal or mildly negative emotions with no danger signals.
**Korean Example Sentences:**
- "오늘 좀 피곤하네."
- "살짝 속상하지만 괜찮아."
- "요즘 스트레스는 있지만 버틸만 해."

---

## 🔹 Label 2 = Medium Risk
Clear emotional struggle, anxiety, or elevated stress, but no explicit self-harm danger.
**Korean Example Sentences:**
- "요즘 아무것도 하고 싶지 않아."
- "잠을 거의 못 자고 계속 불안해."
- "모든 게 무너질 것 같아."

---

## 🔹 Label 3 = High Risk
Severe distress, self-harm ideation, suicidal statements, or extreme emotional collapse.
**Korean Example Sentences:**
- "정말 죽고 싶다."
- "사라지고 싶어. 더는 못 버티겠어."
- "내가 없어지는 게 더 나을 것 같아."

---

# 2) How to Select the Top Words for “reason” (Morphological Basis)

The `reason` must follow these rules:

- It must be a **list of up to 3 items** (1 to 3).
- Each item must be a **morphologically meaningful word or phrase** extracted from the user's input.
- **Exclude function-word morphemes**, such as:
  - 조사 (은/는/이/가/을/를/에/에서/과/와…)
  - 전치사·접속사·어미·감탄사
- Include only **content-word morphemes**, such as:
  - 명사, 동사·형용사 어간, 의미 있는 관용구나 구
- Items must represent the **most influential morphemes** for determining the risk level.
- Sort items by **importance (most influential → least influential)**.

### ✔ Examples:
- Label 1: ["피곤", "속상", "버틸"]
- Label 2: ["아무것도 하고 싶지 않", "불안", "무너질"]
- Label 3: ["죽고 싶", "사라지고 싶", "못 버티"]

---
# 3) Output Format (STRICT)

Return **only** the following JSON object:

{{
  "label": <0 or 1 or 2>,
  "reason": ["<morpheme1>", "<morpheme2>", "<morpheme3>"]
}}

No markdown, no English, no additional text outside the JSON.

Here's the utterance for you to classify: {utterance}
""",
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
        word_importance_physical = risk_model.explain_text_with_konlpy(utterance_decoded)
        # 1. (key, value) 튜플로 변환
        kv_pairs = [(k, v) for d in word_importance_physical for k, v in d.items()]
        # 2. value 기준으로 내림차순 정렬
        kv_pairs_sorted = sorted(kv_pairs, key=lambda x: x[1], reverse=True)
        # 3. 상위 3개 key만 추출
        top3_keys = [k for k, v in kv_pairs_sorted[:3]]
        
        
        response_mental = get_mental_level(utterance) #{"label": 2, "reason": "요즘 아무것도 하고 싶지 않아."}
        try:
            parsed_response = json.loads(response_mental)
            print(parsed_response["label"])
            print(parsed_response["reason"])
            prediction_mental = parsed_response["label"]
            word_importance_mental = parsed_response["reason"]
        except:
            raise RuntimeError("Failed to parse mental level response")
        # prediction_mental, word_importance_mental = risk_model.parse_mental_level_response(response_mental)
        
        reason = get_reason(utterance, prediction_physical, prediction_mental, top3_keys, word_importance_mental)

        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {
        "id": user_id,
        "utterance": utterance,
        "prediction_physical": prediction_physical,
        "prediction_mental": prediction_mental,
        "word_importance_physical": word_importance_physical,
        "reason": reason
    }

@app.get("/")
def root():
    return {"message": "FastAPI model server is running!"}