<<<<<<< HEAD
import os
import jpype
import jpype.imports
from konlpy.tag import Okt
from konlpy.tag import Mecab


konlpy_jar_path = os.path.join(
    os.path.dirname(__file__),
    "../.venv/lib/python3.9/site-packages/konlpy/java"
)
# 확인한 libjvm.dylib 경로
# jvm_path = r"/Library/Java/JavaVirtualMachines/openjdk-19/Contents/Home/lib/server/libjvm.dylib"
jvm_path = jpype.getDefaultJVMPath()    # ★ 직접 path 하드코딩하지 않기 ★
# JVM 시작 (classpath 따로 지정)
jpype.startJVM(
    jvm_path,
    "-Dfile.encoding=UTF-8",
    classpath=f"{konlpy_jar_path}/*"
)
=======
import jpype
from konlpy.tag import Okt

# 확인한 libjvm.dylib 경로
jvm_path = r"/Library/Java/JavaVirtualMachines/jdk-1.8.jdk/Contents/Home/lib/server/libjvm.dylib"

>>>>>>> dfcaf7c (Initial commit)
# JVM 시작
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, "-Dfile.encoding=UTF-8")

<<<<<<< HEAD
mecab = Mecab(dicpath="/usr/local/Cellar/mecab-ko-dic/2.1.1-20180720/lib/mecab/dic/mecab-ko-dic")
text = "숨이 막혀요 너무 힘들어요 응급실 가야하나"
print(mecab.morphs(text))

print("Morphs:", mecab.morphs(text))
print("POS:", mecab.pos(text))
print("Nouns:", mecab.nouns(text))

jpype.shutdownJVM()
=======
# Okt 사용
okt = Okt()
text = "숨이 막혀요 너무 힘들어요 응급실 가야하나"

print("Morphs:", okt.morphs(text))
print("POS:", okt.pos(text))
print("Nouns:", okt.nouns(text))


>>>>>>> dfcaf7c (Initial commit)
# from pororo import Pororo

# Pororo.available_models("dp")
# dp = Pororo(task="dep_parse", lang="ko")

# dp('중간고사 점수 내가 반에서 제일 잘 받음')