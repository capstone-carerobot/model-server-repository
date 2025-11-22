import jpype
from konlpy.tag import Okt

# 확인한 libjvm.dylib 경로
jvm_path = r"/Library/Java/JavaVirtualMachines/jdk-1.8.jdk/Contents/Home/lib/server/libjvm.dylib"

# JVM 시작
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, "-Dfile.encoding=UTF-8")

# Okt 사용
okt = Okt()
text = "숨이 막혀요 너무 힘들어요 응급실 가야하나"

print("Morphs:", okt.morphs(text))
print("POS:", okt.pos(text))
print("Nouns:", okt.nouns(text))


# from pororo import Pororo

# Pororo.available_models("dp")
# dp = Pororo(task="dep_parse", lang="ko")

# dp('중간고사 점수 내가 반에서 제일 잘 받음')