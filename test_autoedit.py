"""실행: python3 test_autoedit.py  (ffmpeg 필요)"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fractions import Fraction

from autoedit import edl, keep_segments, probe_duration, silence_cuts, speech_cuts, srt, subtitles


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

# 자막: 잘린 단어는 빠지고, 시각은 컷된 영상 기준, 18자를 넘으면 나눈다
words = [(0.0, 0.5, "음"), (10.0, 10.5, "오늘은"), (10.5, 11.0, "날씨가"), (11.0, 11.5, "정말"),
         (11.5, 12.0, "좋습니다"), (12.0, 12.5, "그렇죠")]
segs = [{"start": 0, "end": 12.5, "text": "", "words": [{"start": a, "end": b, "text": w} for a, b, w in words]}]
cues = subtitles(segs, [(10.0, 12.5)])   # 앞 10초가 잘린 영상
assert cues == [(0.0, 2.0, "오늘은 날씨가 정말 좋습니다"), (2.0, 2.5, "그렇죠")], cues
assert srt(cues[:1]) == "1\n00:00:00,000 --> 00:00:02,000\n오늘은 날씨가 정말 좋습니다\n\n"
assert srt([(3661.5, 3662.0, "a")]).splitlines()[1] == "01:01:01,500 --> 01:01:02,000"

# EDL: 프레임 단위 타임코드, 레코드 쪽은 빈틈 없이 이어진다
assert edl("a.mp4", [(0, 2.2), (3.88, 6.0)], Fraction(30)).splitlines()[3:7] == [
    "001  AX       AA/V  C        00:00:00:00 00:00:02:06 00:00:00:00 00:00:02:06",
    "* FROM CLIP NAME: a.mp4", "",
    "002  AX       AA/V  C        00:00:03:26 00:00:06:00 00:00:02:06 00:00:04:10"]
# 29.97fps: 3600초 = 107892프레임 = 30 기준 논드롭 00:59:56:12
assert "00:59:56:12" in edl("a.mp4", [(0, 3600)], Fraction(30000, 1001))

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

    # 검수 후 재적용: 자동 컷을 버리고 앞 1초만 자르도록 고치면 5초가 된다
    report = Path(d, "in.cut.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert [c["reason"] for c in data["cuts"]] == ["silence"]
    data["cuts"] = [{"start": 0, "end": 1}]
    report.write_text(json.dumps(data), encoding="utf-8")
    final = Path(d, "final.mp4")
    subprocess.run(cmd + ["--cuts", str(report), "-o", str(final)], check=True)
    assert abs(probe_duration(final) - 5.0) < 0.15, probe_duration(final)
    data["cuts"] = [{"start": 3, "end": 99}]               # 영상 길이를 넘는 컷은 거부
    report.write_text(json.dumps(data), encoding="utf-8")
    assert subprocess.run(cmd + ["--cuts", str(report), "-o", str(Path(d, "bad.mp4"))]).returncode != 0
    assert not Path(d, "bad.mp4").exists()

    # 서로 떨어진 컷 6000개(필터 식 약 160KB): 명령줄 길이 제한에 걸리지 않는다
    many = {"cuts": [{"start": i / 1000 + 0.0003, "end": i / 1000 + 0.0007} for i in range(6000)]}
    report.write_text(json.dumps(many), encoding="utf-8")
    subprocess.run(cmd + ["--cuts", str(report), "-o", str(Path(d, "many.mp4"))], check=True)
    assert 1 < probe_duration(Path(d, "many.mp4")) <= 6.1

    # 가변 프레임레이트 원본: 컷 뒤에도 영상과 소리 길이가 같다
    vfr, vout = Path(d, "vfr.mkv"), Path(d, "vfr.cut.mkv")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=8:r=30:s=320x240",
         "-f", "lavfi", "-i", "sine=f=440:d=8", "-af", "volume='if(between(t,2,6),0,1)':eval=frame",
         "-vf", "select='lt(mod(n,30),10)+not(mod(n,3))'", "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(vfr)],
        check=True)
    subprocess.run([sys.executable, cmd[1], str(vfr)], check=True)
    lengths = [float(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", s, "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(vout)], capture_output=True, text=True, check=True).stdout.split()[-1].strip(",")) for s in ("v:0", "a:0")]
    assert abs(lengths[0] - lengths[1]) < 0.15, lengths

print("OK")
