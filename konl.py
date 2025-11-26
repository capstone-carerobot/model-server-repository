import os
import jpype
from konlpy.tag import Mecab

#-------------------------- Konlpy Mecab JVM Setup --------------------------#
konlpy_jar_path = os.path.join(
    os.path.dirname(__file__),
    "../.venv/lib/python3.9/site-packages/konlpy/java"
)
jvm_path = jpype.getDefaultJVMPath()    # ★ 직접 path 하드코딩하지 않기 ★

# JVM 시작 (classpath 따로 지정)
if not jpype.isJVMStarted():
    jpype.startJVM(
        jvm_path,
        "-Dfile.encoding=UTF-8",
        classpath=f"{konlpy_jar_path}/*"
    )

# 환경변수로 (배포시 고정 경로 사용하지 않도록)
dictpath = os.environ.get("MECAB_PATH")

#-------------------------- Morphological Analysis --------------------------#
pos_dict = {
    'NNG': '일반 명사',
    'NNP': '고유 명사',
    'NNB': '의존 명사',
    'NR': '수사',
    'NP': '대명사',
    'VV': '동사',
    'VA': '형용사', # 한국어에서 동사/형용사는 서술어 역할 (중요)
    'VX': '보조 동사/형용사', # 보조 동사/형용사: '읽어 보다'에서 '~어 보다'
    'MM': '관형사',
    'MAG': '일반 부사',
    'MAJ': '접속 부사',
    'IC': '감탄사',
}

mecab = Mecab(dicpath=dictpath)

def extract_important_word(result, mecab=mecab):
    meaningful_words = []
    for word, score in result:
        pos = mecab.pos(word)
        if any(pos in pos_dict.keys() for _, pos in pos):
            meaningful_words.append({word: score})
    return meaningful_words

jpype.shutdownJVM()