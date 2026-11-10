#!/usr/bin/env python3
"""무음 구간을 잘라낸 새 영상을 만든다. 원본은 건드리지 않는다.

사용: python3 autoedit.py 입력.mp4 [-o 출력.mp4] [--noise -35] [--min-silence 0.6]
      python3 autoedit.py 입력.mp4 --speech   # 추임새·반복 발화(NG)도 컷 + 자막(.srt) 저장
      python3 autoedit.py 입력.mp4 --edl      # 프리미어·리졸브용 컷 목록(.edl)도 저장
검수: 결과와 함께 나오는 .json의 "cuts"를 고친 뒤 다시 렌더링한다
      python3 autoedit.py 입력.mp4 --cuts 입력.cut.json -o 최종.mp4
필요: ffmpeg, ffprobe (PATH). --speech는 추가로 pip install faster-whisper
"""
import argparse
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from fractions import Fraction
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


def subtitles(segments, keeps, max_chars=18):
    """잘리지 않은 단어만 모아, 컷된 영상의 시각에 맞춘 자막 (시작, 끝, 글) 목록을 만든다."""
    def remap(t):  # 원본 시각 -> 컷된 영상 시각
        return sum(min(t, b) - a for a, b in keeps if t > a)

    cues = []
    for seg in segments:
        line = []
        kept = [w for w in seg["words"] if any(a <= (w["start"] + w["end"]) / 2 <= b for a, b in keeps)]
        for w in kept + [None]:
            if line and (w is None or len(" ".join(x["text"] for x in line + [w])) > max_chars):
                cues.append((remap(line[0]["start"]), remap(line[-1]["end"]), " ".join(x["text"] for x in line)))
                line = []
            line.append(w)
    return cues


def srt(cues):
    def stamp(t):
        ms = round(t * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))


def probe_fps(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=r_frame_rate", "-of", "csv=p=0", str(path)]).stdout.strip()
    if not out:
        sys.exit("--edl에는 영상 트랙이 필요합니다.")
    return Fraction(out)


def edl(clip_name, keeps, fps):
    """남길 구간을 CMX3600 EDL로 만든다. 편집툴에서 원본 파일에 다시 연결해 쓴다."""
    # shortcut: 원본 타임코드가 00:00:00:00에서 시작한다고 가정, 드롭 프레임 표기 미지원.
    # 카메라 원본처럼 시작 타임코드가 있는 파일을 다룰 때 ffprobe의 timecode 태그를 더한다.
    base = round(fps)

    def tc(f):
        return f"{f // (base * 3600):02}:{f // (base * 60) % 60:02}:{f // base % 60:02}:{f % base:02}"

    lines, rec, n = [f"TITLE: {clip_name}", "FCM: NON-DROP FRAME", ""], 0, 0
    for a, b in keeps:
        fa, fb = round(a * fps), round(b * fps)
        if fb > fa:
            n += 1
            lines += [f"{n:03}  AX       AA/V  C        {tc(fa)} {tc(fb)} {tc(rec)} {tc(rec + fb - fa)}",
                      f"* FROM CLIP NAME: {clip_name}", ""]
            rec += fb - fa
    return "\n".join(lines)


def load_cuts(path, duration):
    """사람이 고친 .json에서 컷 목록과 전사를 읽는다. 손으로 고친 파일이라 형식을 확인한다."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cuts = [(float(c["start"]), float(c["end"]), str(c.get("reason", "manual"))) for c in data["cuts"]]
        segments = data.get("segments", [])
        [(float(w["start"]), float(w["end"]), w["text"]) for s in segments for w in s["words"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.exit(f"컷 파일을 읽을 수 없습니다: {path} ({type(e).__name__}: {e})")
    for a, b, _ in cuts:
        if not 0 <= a < b <= duration + 0.01:
            sys.exit(f"컷 구간이 잘못됐습니다: {a}~{b} (0 <= start < end <= {duration:.2f})")
    return cuts, segments


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
    ap.add_argument("--edl", action="store_true", help="편집툴용 컷 목록(CMX3600 .edl)도 저장")
    ap.add_argument("--cuts", type=Path, help="감지 대신 이 .json의 컷 목록으로 렌더링 (검수 후 재적용)")
    ap.add_argument("--model", default="small", help="Whisper 모델 (기본 small, 정확도 우선이면 large-v3)")
    a = ap.parse_args()

    if not a.input.is_file():
        sys.exit(f"입력 파일이 없습니다: {a.input}")
    duration = probe_duration(a.input)
    loaded = load_cuts(a.cuts, duration) if a.cuts else None
    out = a.output or a.input.with_name(f"{a.input.stem}.cut{a.input.suffix}")
    want_srt = a.speech or bool(loaded and loaded[1])
    sidecars = [out.with_suffix(x) for x, on in ((".json", True), (".srt", want_srt), (".edl", a.edl)) if on]
    for f in [out, *sidecars]:  # 원본·기존 결과물을 덮어쓰지 않는다
        if f.exists():
            sys.exit(f"출력 파일이 이미 있습니다: {f} (-o로 다른 이름을 지정하세요)")

    fps = probe_fps(a.input) if a.edl else None
    if loaded:
        cuts, segments = loaded
    else:
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
    # 무엇을 왜 잘랐는지 남긴다 (시각은 원본 기준). 이 파일을 고쳐 --cuts로 다시 렌더링할 수 있다
    report = {"source": str(a.input), "duration": duration, "segments": segments,
              "cuts": [{"start": c[0], "end": c[1], "reason": c[2]} for c in sorted(cuts)]}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if want_srt:
        out.with_suffix(".srt").write_text(srt(subtitles(segments, keeps)), encoding="utf-8")
    if a.edl:
        out.with_suffix(".edl").write_text(edl(a.input.name, keeps, fps), encoding="utf-8")
    print(f"{out}  {duration:.1f}초 -> {kept:.1f}초 (구간 {len(keeps)}개, {duration - kept:.1f}초 제거)")


if __name__ == "__main__":
    main()
