"""실행: python3 test_autoedit.py  (ffmpeg 필요)"""
import subprocess
import sys
import tempfile
from pathlib import Path

from autoedit import keep_segments, probe_duration, silence_cuts, speech_cuts


def keeps(silences, duration):
    return keep_segments(silence_cuts(silences, duration), duration)


# 무음 컷: 여백(뒤 0.20, 앞 0.12)을 남기고 자른다
assert keeps([], 10) == [(0.0, 10)]
assert keeps([(2.0, 4.0)], 6.0) == [(0.0, 2.2), (3.88, 6.0)]
assert keeps([(2.0, 2.3)], 6.0) == [(0.0, 6.0)]                 # 여백보다 짧으면 안 자름
assert keeps([(4.0, 6.0)], 6.0) == [(0.0, 4.2), (5.88, 6.0)]    # 끝이 무음
# 겹치거나 포함된 컷은 합쳐진다
assert keep_segments([(1, 3, "ng"), (2, 2.5, "filler"), (2.8, 4, "silence")], 5) == [(0.0, 1), (4, 5)]


def seg(start, end, text):
    return {"start": start, "end": end, "text": text,
            "words": [{"start": start, "end": end, "text": text}]}


# 음성 컷: 추임새 단어, 다시 말한 문장의 앞 테이크
assert speech_cuts([seg(0, 0.4, "음..."), seg(0.4, 2, "그 사람은 어제 왔습니다")]) == [(0, 0.4, "filler")]
assert speech_cuts([seg(0, 2, "오늘은 날씨가 좋습니다"), seg(3, 5, "오늘은 날씨가 참 좋습니다"),
                    seg(5, 7, "다음 주제로 넘어가겠습니다")]) == [(0, 3, "ng")]
assert speech_cuts([seg(0, 1, "네."), seg(1, 2, "네.")]) == []   # 짧은 맞장구는 NG로 보지 않음

# 전체 흐름: 소리 2초 + 무음 2초 + 소리 2초 -> 약 4.32초
with tempfile.TemporaryDirectory() as d:
    src, out = Path(d, "in.mp4"), Path(d, "in.cut.mp4")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=6:r=30:s=320x240",
         "-f", "lavfi", "-i", "sine=f=440:d=6", "-af", "volume='if(between(t,2,4),0,1)':eval=frame",
         "-pix_fmt", "yuv420p", str(src)], check=True)
    before = src.read_bytes()
    cmd = [sys.executable, str(Path(__file__).with_name("autoedit.py")), str(src)]
    subprocess.run(cmd, check=True)
    assert abs(probe_duration(out) - 4.32) < 0.15, probe_duration(out)
    assert src.read_bytes() == before                      # 원본 불변
    assert subprocess.run(cmd).returncode != 0             # 기존 출력 덮어쓰기 거부

print("OK")
