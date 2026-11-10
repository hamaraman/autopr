"""실행: python3 test_autoedit.py  (ffmpeg 필요)"""
import subprocess
import sys
import tempfile
from pathlib import Path

from autoedit import keep_segments, probe_duration

# 컷 계산: 여백(뒤 0.20, 앞 0.12)을 남기고 자른다
assert keep_segments([], 10) == [(0.0, 10)]
assert keep_segments([(2.0, 4.0)], 6.0) == [(0.0, 2.2), (3.88, 6.0)]
assert keep_segments([(2.0, 2.3)], 6.0) == [(0.0, 6.0)]            # 여백보다 짧으면 안 자름
assert keep_segments([(4.0, 6.0)], 6.0) == [(0.0, 4.2), (5.88, 6.0)]  # 끝이 무음

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
