#!/usr/bin/env python3
"""무음 구간을 잘라낸 새 영상을 만든다. 원본은 건드리지 않는다.

사용: python3 autoedit.py 입력.mp4 [-o 출력.mp4] [--noise -35] [--min-silence 0.6]
필요: ffmpeg, ffprobe (PATH)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

PAD_AFTER = 0.20   # 말이 끝난 뒤 남기는 여백(초)
PAD_BEFORE = 0.12  # 말이 시작되기 전 남기는 여백(초)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"실패: {' '.join(cmd[:2])} ...\n{p.stderr[-800:]}")
    return p


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)]).stdout.strip()
    return float(out)


def detect_silences(path, noise_db, min_silence, duration):
    log = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
               "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
               "-f", "null", "-"]).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", log)]
    ends += [duration] * (len(starts) - len(ends))  # 파일이 무음으로 끝난 경우
    return list(zip(starts, ends))


def keep_segments(silences, duration):
    """무음 목록 -> 남길 구간 목록. 무음 양끝에 여백을 남겨 말이 잘리지 않게 한다."""
    keeps, pos = [], 0.0
    for start, end in silences:
        cut_from, cut_to = max(start, 0.0) + PAD_AFTER, min(end, duration) - PAD_BEFORE
        if cut_to <= cut_from:
            continue
        if cut_from > pos:
            keeps.append((pos, cut_from))
        pos = cut_to
    if duration > pos:
        keeps.append((pos, duration))
    return keeps


def render(src, dst, keeps):
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keeps)
    # shortcut: 고정 프레임레이트 가정 + 구간이 수천 개면 명령줄 길이 초과(특히 Windows).
    # 가변 fps 원본이나 2시간 이상 영상을 다룰 때 구간별 분할 렌더링으로 교체.
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(src),
         "-vf", f"select='{expr}',setpts=N/FRAME_RATE/TB",
         "-af", f"aselect='{expr}',asetpts=N/SR/TB", str(dst)])


def main():
    ap = argparse.ArgumentParser(description="무음 구간 자동 컷")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--noise", type=float, default=-35, help="무음 기준 dB (기본 -35)")
    ap.add_argument("--min-silence", type=float, default=0.6, help="최소 무음 길이 초 (기본 0.6)")
    a = ap.parse_args()

    if not a.input.is_file():
        sys.exit(f"입력 파일이 없습니다: {a.input}")
    out = a.output or a.input.with_name(f"{a.input.stem}.cut{a.input.suffix}")
    if out.exists():  # 원본·기존 결과물을 덮어쓰지 않는다
        sys.exit(f"출력 파일이 이미 있습니다: {out}")

    duration = probe_duration(a.input)
    keeps = keep_segments(detect_silences(a.input, a.noise, a.min_silence, duration), duration)
    kept = sum(b - s for s, b in keeps)
    if duration - kept < 0.01:
        print("잘라낼 무음 구간이 없습니다.")
        return
    render(a.input, out, keeps)
    print(f"{out}  {duration:.1f}초 -> {kept:.1f}초 (구간 {len(keeps)}개, {duration - kept:.1f}초 제거)")


if __name__ == "__main__":
    main()
