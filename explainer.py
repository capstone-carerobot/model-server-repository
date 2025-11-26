from transformers_interpret import SequenceClassificationExplainer
from konl import extract_important_word

def explain_sample_filtered(model, tokenizer, text):
    """
    중요도 기반으로 명사 + 형용사만 출력
    """
    explainer = SequenceClassificationExplainer(model, tokenizer)
    try:
        word_attributions = explainer(text)
        # 단어별 중요도
        result = [(w, round(s, 3)) for w, s in word_attributions]

        #  ##로 시작하는 서브워드 제거
        result = [(w, s) for w, s in result if not w.startswith("##")]
        
        # 명사 + 형용사만 추출
        result = extract_important_word(result)

        print(f"[Input Text]: {text}\n")
        print("[Meaningful Word Importances]:")
        for w, s in result:
            print(f"{w:15s} -> {s}")
        return result

    except Exception as e:
        print(f"[Explainer Error: {e}]")
        return None


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