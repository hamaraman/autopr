#!/usr/bin/env python3
"""무음 구간을 잘라낸 새 영상을 만든다. 원본은 건드리지 않는다.

사용: python3 autoedit.py 입력.mp4 [-o 출력.mp4] [--noise -35] [--min-silence 0.6]
      python3 autoedit.py 입력.mp4 --speech   # 추임새·반복 발화(NG)도 컷 + 전사 JSON 저장
필요: ffmpeg, ffprobe (PATH). --speech는 추가로 pip install faster-whisper
"""
import argparse
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

FILLERS = {"음", "어", "으음", "어어", "음음"}  # "그", "저"는 지시어와 구분이 안 돼 제외
NG_SIMILARITY = 0.8
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


def silence_cuts(silences, duration):
    """무음 양끝에 여백을 남겨 말이 잘리지 않게 한다."""
    cuts = []
    for start, end in silences:
        a, b = max(start, 0.0) + PAD_AFTER, min(end, duration) - PAD_BEFORE
        if b > a:
            cuts.append((a, b, "silence"))
    return cuts


def transcribe(path, model, spans):
    """발화 구간(spans)마다 따로 인식한다. 통째로 넣으면 Whisper가 반복된 문장을 빼먹는다."""
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("--speech에는 faster-whisper가 필요합니다: pip install faster-whisper")
    # ffmpeg로 직접 디코딩: 영상 컨테이너를 그대로 받을 수 있고 PyAV 버전 차이를 피한다
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], capture_output=True)
    if pcm.returncode != 0:
        sys.exit(f"오디오 디코딩 실패:\n{pcm.stderr.decode(errors='replace')[-800:]}")
    audio, whisper = np.frombuffer(pcm.stdout, np.float32), WhisperModel(model, device="auto", compute_type="auto")
    segments = []
    for a, b in spans:
        parts, _ = whisper.transcribe(audio[int(a * 16000):int(b * 16000)], language="ko", word_timestamps=True)
        words = [{"start": a + w.start, "end": a + w.end, "text": w.word.strip()} for s in parts for w in s.words]
        if words:
            segments.append({"start": a, "end": b, "text": " ".join(w["text"] for w in words), "words": words})
    return segments


def speech_cuts(segments):
    """추임새 단어와, 바로 뒤에서 다시 말한 문장(NG)의 앞 테이크를 자른다."""
    # shortcut: Whisper 단어 시각은 ±0.2초 오차, 문장 단위는 무음으로 나뉜 발화 구간.
    # 컷이 말을 먹기 시작하면 강제 정렬(wav2vec2)과 문장 분리를 추가.
    cuts = [(w["start"], w["end"], "filler") for s in segments for w in s["words"]
            if w["text"].strip(" .,!?~…") in FILLERS]
    for a, b in zip(segments, segments[1:]):
        if len(a["text"]) >= 5 and SequenceMatcher(None, a["text"], b["text"]).ratio() >= NG_SIMILARITY:
            cuts.append((a["start"], b["start"], "ng"))
    return cuts


def keep_segments(cuts, duration):
    """자를 구간 목록(겹쳐도 됨) -> 남길 구간 목록."""
    keeps, pos = [], 0.0
    for a, b, _ in sorted(cuts):
        if a > pos:
            keeps.append((pos, a))
        pos = max(pos, b)
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
    ap.add_argument("--speech", action="store_true", help="음성 인식으로 추임새·반복 발화(NG)도 컷")
    ap.add_argument("--model", default="small", help="Whisper 모델 (기본 small, 정확도 우선이면 large-v3)")
    a = ap.parse_args()

    if not a.input.is_file():
        sys.exit(f"입력 파일이 없습니다: {a.input}")
    out = a.output or a.input.with_name(f"{a.input.stem}.cut{a.input.suffix}")
    if out.exists():  # 원본·기존 결과물을 덮어쓰지 않는다
        sys.exit(f"출력 파일이 이미 있습니다: {out}")

    duration = probe_duration(a.input)
    silences = detect_silences(a.input, a.noise, a.min_silence, duration)
    cuts = silence_cuts(silences, duration)
    spans = keep_segments([(s, e, "") for s, e in silences], duration)  # 무음의 여집합 = 발화 구간
    segments = transcribe(a.input, a.model, spans) if a.speech else []
    cuts += speech_cuts(segments)
    keeps = keep_segments(cuts, duration)
    kept = sum(b - s for s, b in keeps)
    if duration - kept < 0.01:
        print("잘라낼 구간이 없습니다.")
        return
    render(a.input, out, keeps)
    if a.speech:  # 무엇을 왜 잘랐는지 검수할 수 있게 남긴다 (시각은 원본 기준)
        report = {"source": str(a.input), "duration": duration, "segments": segments,
                  "cuts": [{"start": c[0], "end": c[1], "reason": c[2]} for c in sorted(cuts)]}
        out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}  {duration:.1f}초 -> {kept:.1f}초 (구간 {len(keeps)}개, {duration - kept:.1f}초 제거)")


if __name__ == "__main__":
    main()
