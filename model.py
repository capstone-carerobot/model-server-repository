from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from konl import extract_important_word
from typing import Tuple, Dict
import re
import torch.nn as nn
import torch.nn.functional as F
import torch
import os


class RiskModel:
    def __init__(self, model_path: str, max_len: int = 128):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # 평가 모드로 전환

    def encode(self, text: str):
        enc = self.tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']
    
    def predict_physical_level(self, text: str) -> int:
        input_ids, attention_mask = self.encode(text)
        with torch.no_grad():
             # Batch size 만큼 들어오는 걸로 모델이 설계가 되어 있으므로 shape 줄이면 안 됨
            # 설계 단계에서 input_ids는 (MAX_LEN)이지만 들어갈 땐 (BATCH_SIZE, MAX_LEN) 형태로 들어가기 때문에
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return int(torch.argmax(logits, dim=1))
        
        
    def decode_input(self, text: str):
        input_ids, _ = self.encode(text)[0], self.encode(text)[1]
        decoded_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        return decoded_text


    def explain_text_with_konlpy(self, text: str, filter_pos: bool = True):
        explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
        try:
            word_attributions = explainer(text)
            print('[Explainer] Word attributions computed.')
            # 소수점 3자리 반올림
            word_attributions = [(w, round(s, 3)) for w, s in word_attributions]
            print('[Explainer] Word attributions rounded.')
            # 서브워드 제거
            word_attributions = [(w, s) for w, s in word_attributions if not w.startswith("##")]
            print('[Explainer] Subword tokens removed.')
            
            if filter_pos:
                word_attributions = extract_important_word(word_attributions)
                print('[Explainer] Filtered by POS tags.')
            
            return word_attributions
        
        except Exception as e:
            raise RuntimeError(f"Explainer failed: {e}")
        
    def naive_explain_text(self, text: str):
        explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
        try:
            word_attributions = explainer(text)
            print('[Explainer] Word attributions computed.')
            # 소수점 3자리 반올림
            word_attributions = [(w, round(s, 3)) for w, s in word_attributions]
            print('[Explainer] Word attributions rounded.')
            # 서브워드 제거
            word_attributions = [(w, s) for w, s in word_attributions if not w.startswith("##")]
            print('[Explainer] Subword tokens removed.')
            
            return word_attributions
        
        except Exception as e:
            raise RuntimeError(f"Explainer failed: {e}")
        
    
    def parse_mental_level_response(self, response_text: str) -> Tuple[int, Dict]:
        """
        GPT 응답에서 mental risk level 파싱
        """
        try:
            # mental risk level 추출
            level_match = re.search(r"mental risk level:\s*(\d+)", response_text)
            mental_risk_level = int(level_match.group(1)) if level_match else None

            # word importance 추출
            word_importance_matches = re.findall(r"\(([^:]+):\s*([0-9.]+)\)", response_text)
            word_importance = [{word.strip(): float(score)} for word, score in word_importance_matches]
            return mental_risk_level, word_importance
        
        except Exception as e:
            raise ValueError(f"Failed to parse mental risk level: {e}")