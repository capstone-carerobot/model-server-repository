# 🤖 Senior-care Robot Modeling
This project develops an intelligent senior-care monitoring system designed for elderly individuals living alone in Daejeon, South Korea. By analyzing raw utterance data, the fine-tuned model predicts both the physical and mental risk levels of a senior, classified into a 3-tier ordinal scale: Low (0), Medium (1), and High (2). To ensure proactive emergency response, the architecture is engineered to trigger an immediate automated alarm to the centralized monitoring center when a 'High (2)' risk state is inferred, enabling rapid, life-saving interventions.

## Project Overview
As aging populations expand, social isolation among the elderly has emerged as a critical issue, increasing the risks of unmonitored medical emergencies and lonely deaths. This project enhances senior-care companion robots by implementing an intelligent NLP framework capable of analyzing daily conversational logs.

While basic systems rely on superficial keyword matching, this architecture deploys a fine-tuned **KLUE-BERT** sequence classifier. The core objective is to securely parse speech text, accurately classify complex emotional and physical vulnerability (low, medium, and high), and trigger proactive emergency backend alerts.

## Objectives

* Classify risk levels (0–2) from care robot dialogue data with higher precision than keyword-based methods
* Provide explainable evidence for each risk prediction using token-level attention scores and GPT-generated rationale
* Support real-time monitoring dashboard for care center operators

## Data Preprocessing

**Data Sources**
- AI-Hub emergency call transcripts (labeled: high / medium / low → remapped to 3 / 2 / 1)
- General conversation corpus (labeled as 0: no risk)

**Steps**
1. Merged emergency call data with general conversation data
2. Remapped original 3-class labels (high/medium/low) to numeric (3/2/1); added label 0 for everyday dialogue
3. Filtered out irrelevant utterances (greetings, filler phrases, hold messages) from emergency data
4. Categorized emergency types into: first aid, rescue, fire, and other — excluded utterances outside these categories
5. After initial training, risk levels 2 and 3 showed poor vector separation due to semantic similarity → **merged into a single high-risk class**, resulting in a final **3-label schema (0, 1, 2)**

**Mental Risk Classification — Approach & Decision**
- Initially collected psychological counseling dialogue data from AI-Hub and attempted to fine-tune a classification model for mental risk detection
- Due to token length constraints, applied SBERT-based summarization as a preprocessing step before training
- However, the class imbalance in counseling data proved too severe to overcome, making stable model training infeasible
- Replaced the fine-tuned model approach with **GPT API (GPT-4o-mini)**, which handles mental risk classification without requiring labeled training data

## Model

### Architecture

The model pipeline classifies both **mental** and **physical** risk in parallel, then synthesizes a final risk level.

```
Input Text (utterance sequence)
    │
    ├─► [Physical Risk Classification]
    │       KLUE-BERT (fine-tuned)
    │       → Physical Risk Label + Token-level Importance Scores
    │
    └─► [Mental Risk Classification]
            LLM API (e.g., GPT-4o-mini)
            → Mental Risk Label + Top-3 Important Tokens
    │
    └─► Interpret & Combine Results → Final Risk Level (0 / 1 / 2)
```

### KLUE-BERT

- Pre-trained Korean language model from the KLUE benchmark
- Trained on ~62 GB of Korean corpora (news, web, wiki, petitions, etc.)
- Uses morpheme-based subword tokenization suited for Korean morphological structure
- Fine-tuned on the preprocessed utterance dataset for risk classification

### Attention (Scaled Dot-Product)

- Applied to identify which tokens most strongly signal risk
- Computes similarity between encoder hidden states (Key) and decoder states (Query)
- Outputs a weighted context vector, enabling the model to dynamically focus on risk-relevant phrases
- Token-level attention scores are surfaced to dashboard users as explainable evidence



### Risk Level Schema

| Label | Meaning |
|-------|---------|
| 0 | Normal / everyday conversation |
| 1 | Low-risk signal |
| 2 | High-risk — triggers immediate alert |



## Model Performance

**Test Set: Accuracy 0.8589 / Macro-F1 0.8724**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 1 | 0.998 | 1.000 | 0.999 |
| 2 | 0.843 | 0.906 | 0.875 |
| 3 | 0.807 | 0.689 | 0.743 |

**Confusion Matrix**

| Actual \ Predicted | 1 | 2 | 3 |
|---|---|---|---|
| **1** | 3099 | 0 | 0 |
| **2** | 4 | 9235 | 914 |
| **3** | 2 | 1724 | 3820 |

- Class 1 (low risk) is classified almost perfectly
- Classes 2 and 3 show some confusion due to semantic proximity, which motivated the label-merging decision in preprocessing
- Macro-F1 was chosen over standard F1 to equally weight all classes and capture minority-class performance



## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | KLUE-BERT, PyTorch, Jupyter |
| Inference Server | FastAPI |
| LLM Integration | GPT-4o-mini API |
| Infrastructure | AWS EC2 (t2.micro), AWS RDS (MySQL) |
| Security | Nginx Reverse Proxy, Let's Encrypt SSL, HTTPS / WSS |

