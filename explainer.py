from transformers_interpret import SequenceClassificationExplainer

def explain_sample(model, tokenizer, text):
    """
    특정 문장에 대해 단어 중요도를 계산하고 출력
    """
    explainer = SequenceClassificationExplainer(model, tokenizer)
    try:
        word_attributions = explainer(text)
        result = [(w, round(s, 3)) for w, s in word_attributions]
        print(f"[Input Text]: {text}\n")
        print("[Word Importances]:")
        for w, s in result:
            print(f"{w:15s} -> {s}")
        return result
    except Exception as e:
        print(f"[Explainer Error: {e}]")
        return None