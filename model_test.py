from transformers import AutoModelForSequenceClassification, AutoTokenizer
from explainer import explain_sample
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys

print(os.listdir())
print(sys.executable)

MAX_LEN = 128

model_path = "1118_15epochs"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

input = ["숨이 막혀요 너무 힘들어요 [SEP] 응급실 가야하나"]
# enc = tokenizer(
#     input,
#     padding = 'max_length',
#     truncation = True,
#     max_length = MAX_LEN,
#     return_tensors = 'pt'
# )
# model_test.py (핵심 부분)
enc = tokenizer(input, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
input_ids = enc['input_ids']
attn_masks = enc['attention_mask']
outputs = model(input_ids=input_ids, attention_mask=attn_masks)
logits = outputs.logits
pred = torch.argmax(logits, dim=1).item()

print(enc['input_ids'].shape) # (1, 128)
print(enc['attention_mask'].shape) # (1, 128)

# Batch size 만큼 들어오는 걸로 모델이 설계가 되어 있으므로 shape 줄이면 안 됨
# 설계 단계에서 input_ids는 (MAX_LEN)이지만 들어갈 땐 (BATCH_SIZE, MAX_LEN) 형태로 들어가기 때문에
# input_ids, attn_masks = enc['input_ids'], enc['attention_mask'] 
# outputs = model(input_ids=input_ids, 
#                 attention_mask=attn_masks)
# logits = outputs.logits
# pred = torch.argmax(logits, dim=1).item()

print(outputs) 
print(logits)
print(f'위험도 레벨: {pred}')
text_to_explain = tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
explain_sample(model, tokenizer, text_to_explain)