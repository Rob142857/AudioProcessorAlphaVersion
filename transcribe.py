import sys
import os
import shutil
import subprocess
import tempfile
import argparse
import whisper
import torch
from docx import Document
from moviepy import VideoFileClip, AudioFileClip
import imageio_ffmpeg as iio_ffmpeg
from deepmultilingualpunctuation import PunctuationModel
import webrtcvad
import re
import time
import json

try:
    import torch_directml
except Exception:
    torch_directml = None


def split_into_paragraphs(text, max_length=500):
    paras = []
    current = []
    count = 0
    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                paras.append(" ".join(current).strip())
                current = []
                count = 0
        else:
            current.append(line.strip())
            count += len(line)
            if count > max_length:
                paras.append(" ".join(current).strip())
                current = []
                count = 0
    if current:
        paras.append(" ".join(current).strip())
    return paras


def convert_to_mp3(input_path, bitrate="192k"):
    ext = os.path.splitext(input_path)[1].lower()
    mp3_path = os.path.splitext(input_path)[0] + "_converted.mp3"
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        clip = VideoFileClip(input_path)
        try:
            if clip.audio is None:
                raise ValueError("No audio track found in the video file")
            clip.audio.write_audiofile(mp3_path, codec="libmp3lame", bitrate=bitrate)
        finally:
            clip.close()
        return mp3_path
    elif ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
        clip = AudioFileClip(input_path)
        try:
            clip.write_audiofile(mp3_path, codec="libmp3lame", bitrate=bitrate)
        finally:
            clip.close()
        return mp3_path
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def ensure_ffmpeg_in_path():
    if shutil.which("ffmpeg"):
        return
    ffexe = None
    if hasattr(iio_ffmpeg, "get_ffmpeg_exe"):
        try:
            ffexe = iio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffexe = None
    if not ffexe and hasattr(iio_ffmpeg, "get_exe"):
        try:
            ffexe = iio_ffmpeg.get_exe()
        except Exception:
            ffexe = None
    if ffexe and os.path.isfile(ffexe):
        # Create a small wrapper batch file named ffmpeg.bat in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bat_path = os.path.join(script_dir, "ffmpeg.bat")
        try:
            if not os.path.exists(bat_path):
                with open(bat_path, "w", encoding="utf-8") as bat:
                    # Use quotes around ffexe and forward all args
                    bat.write(f'@"{ffexe}" %*\n')
            # Prepend script dir to PATH so 'ffmpeg' resolves to our batch wrapper
            os.environ["PATH"] = script_dir + os.pathsep + os.environ.get("PATH", "")
            return
        except Exception:
            # Fall back to adding the ffmpeg folder directly
            ffdir = os.path.dirname(ffexe)
            os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("ffmpeg not found. Install ffmpeg or ensure imageio_ffmpeg is available.")


def preprocess_audio(input_path, out_path, bitrate="128k"):
    """Run ffmpeg with simple filters to improve audio quality before transcription.
    Applies highpass, lowpass, spectral denoise and normalization, and resamples to 16k mono.
    """
    ff = shutil.which("ffmpeg") or None
    if not ff:
        # Try imageio_ffmpeg
        ffexe = None
        if hasattr(iio_ffmpeg, "get_ffmpeg_exe"):
            try:
                ffexe = iio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffexe = None
        if not ffexe and hasattr(iio_ffmpeg, "get_exe"):
            try:
                ffexe = iio_ffmpeg.get_exe()
            except Exception:
                ffexe = None
        ff = ffexe
    if not ff:
        raise RuntimeError("ffmpeg not found for preprocessing")

    # Filter chain: remove low rumble, limit highs, denoise, normalize
    filters = "highpass=f=80, lowpass=f=8000, afftdn, dynaudnorm=g=5"
    cmd = [
        ff,
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filters,
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        out_path,
    ]
    subprocess.run(cmd, check=True)


def choose_device(preferred: str = "auto"):
    """Return a torch device string based on preferred choice and availability.
    preferred: 'auto'|'cpu'|'cuda'|'mps'|'dml'
    Behavior: prefer DirectML (dml) when available by default to support Windows/ARM NPUs; fallback to CUDA/MPS/CPU.
    """
    preferred = (preferred or "auto").lower()
    if preferred == "auto":
        # Prefer DirectML on Windows when torch-directml is importable and initializable
        if torch_directml is not None:
            try:
                # attempt to get a DirectML device; this will raise if not supported
                _ = torch_directml.device()
                return "dml"
            except Exception:
                pass
        if torch.cuda.is_available():
            return "cuda"
        # macOS MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred in ("dml", "directml"):
        if torch_directml is not None:
            try:
                _ = torch_directml.device()
                return "dml"
            except Exception:
                print("torch-directml importable but initialization failed; falling back to CPU")
                return "cpu"
        print("torch-directml not installed or not importable; falling back to CPU")
        return "cpu"

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_media_duration(path):
    """Return media duration in seconds. Try moviepy then ffprobe fallback."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            clip = VideoFileClip(path)
            try:
                return float(clip.duration)
            finally:
                clip.close()
        else:
            clip = AudioFileClip(path)
            try:
                return float(clip.duration)
            finally:
                clip.close()
    except Exception:
        # fallback to ffprobe
        ff = shutil.which("ffprobe") or shutil.which("ffmpeg")
        if not ff:
            return None
        # use ffprobe if available
        try:
            cmd = [ff, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            return float(out.decode().strip())
        except Exception:
            return None


def transcribe_file(input_path, model_name="base", preprocess=False, keep_temp=False, bitrate="192k", device_preference="auto", vad=False, punctuate=False):
    ensure_ffmpeg_in_path()

    start_time = time.time()

    temp_files = []
    log_outputs = []
    try:
        ext = os.path.splitext(input_path)[1].lower()
        # If user requested preprocessing, always create a temp file with processed audio
        if preprocess:
            tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tf.close()
            pre_path = tf.name
            print(f"Preprocessing audio -> {pre_path}...")
            preprocess_audio(input_path, pre_path, bitrate=bitrate)
            audio_path = pre_path
            temp_files.append(pre_path)
        else:
            # Convert non-mp3 inputs to mp3
            if ext != ".mp3":
                print(f"Converting {input_path} to MP3 (compressed)...")
                audio_path = convert_to_mp3(input_path, bitrate=bitrate)
                temp_files.append(audio_path)
            else:
                audio_path = input_path

        media_length = get_media_duration(input_path)

        print(f"Loading Whisper model '{model_name}' (this may take a while)...")
        # Select device
        device = choose_device(device_preference)
        print(f"Selected device: {device}")
        # Whisper's load_model will place parameters on the detected device; load then ensure placement
        model = whisper.load_model(model_name)
        try:
            model.to(device)
        except Exception:
            # Some whisper wrappers accept a device argument; ignore if move fails
            pass

        outputs = []
        segment_files_used = []
        if vad:
            print("Running VAD segmentation...")
            segments = vad_segment_times(audio_path, aggressiveness=2, frame_duration_ms=30, padding_ms=200)
            print(f"Raw segments found: {len(segments)}")
            segments = post_process_segments(segments, min_duration=0.6, merge_gap=0.35, max_segments=120)
            print(f"Segments after post-processing: {len(segments)}")
            seg_files = []
            for idx, (s, e) in enumerate(segments):
                seg_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                seg_tmp.close()
                print(f"Extracting segment {idx+1}/{len(segments)}: {s:.2f}-{e:.2f}s")
                extract_segment(audio_path, s, e, seg_tmp.name, bitrate=bitrate)
                seg_files.append(seg_tmp.name)
                temp_files.append(seg_tmp.name)
            # transcribe each segment and join
            for seg in seg_files:
                print(f"Transcribing segment {seg}")
                res = model.transcribe(seg, language=None)
                text = res.get("text", "").strip()
                outputs.append(text)
            segment_files_used = seg_files
        else:
            res = model.transcribe(audio_path, language=None)
            outputs = [res.get("text", "").strip()]

        full_text = "\n\n".join(outputs)

        if punctuate:
            print("Running punctuation model to restore punctuation...")
            pm = PunctuationModel()
            full_text = pm.restore_punctuation(full_text)

        # clean fillers
        full_text = clean_fillers(full_text)

        out_txt_path = os.path.splitext(audio_path)[0] + ".txt"
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Transcription saved to {out_txt_path}")
        log_outputs.append(os.path.abspath(out_txt_path))

        doc = Document()
        doc.add_heading("Transcription", 0)
        for para in split_into_paragraphs(full_text):
            doc.add_paragraph(para)
        out_docx = os.path.splitext(audio_path)[0] + ".docx"
        doc.save(out_docx)
        print(f"Word document saved to {out_docx}")
        log_outputs.append(os.path.abspath(out_docx))

        # include segments if kept
        if keep_temp and segment_files_used:
            log_outputs.extend([os.path.abspath(p) for p in segment_files_used])

        end_time = time.time()
        elapsed = end_time - start_time

        # write log file next to the output txt (or next to input if outputs were temp)
        orig_base = os.path.splitext(os.path.basename(input_path))[0]
        log_name = f"{orig_base}_transcription_log.json"
        # prefer writing next to project folder or same dir as output
        log_dir = os.path.dirname(os.path.abspath(out_txt_path)) if out_txt_path else os.path.dirname(os.path.abspath(input_path))
        log_path = os.path.join(log_dir, log_name)
        log = {
            "input_file": os.path.abspath(input_path),
            "model": model_name,
            "device": device,
            "preprocess": bool(preprocess),
            "vad": bool(vad),
            "punctuate": bool(punctuate),
            "time_taken_seconds": round(elapsed, 2),
            "original_length_seconds": None if media_length is None else round(media_length, 2),
            "outputs": log_outputs,
            "temp_files_removed": not keep_temp,
        }
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                json.dump(log, lf, indent=2)
            print(f"Log written to {log_path}")
        except Exception as e:
            print(f"Failed to write log file: {e}")

    finally:
        if not keep_temp:
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

    return out_txt_path


def frames_from_pcm(pcm_bytes, frame_duration_ms=30, sample_rate=16000):
    bytes_per_frame = int(sample_rate * 2 * (frame_duration_ms / 1000.0))
    for i in range(0, len(pcm_bytes), bytes_per_frame):
        yield pcm_bytes[i:i+bytes_per_frame]


def get_pcm_from_file(input_path):
    # Use ffmpeg to output raw PCM s16le 16k mono
    ff = shutil.which("ffmpeg") or None
    if not ff:
        ffexe = None
        if hasattr(iio_ffmpeg, "get_ffmpeg_exe"):
            try:
                ffexe = iio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffexe = None
        if not ffexe and hasattr(iio_ffmpeg, "get_exe"):
            try:
                ffexe = iio_ffmpeg.get_exe()
            except Exception:
                ffexe = None
        ff = ffexe
    cmd = [ff, "-i", input_path, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = p.stdout.read()
    p.stdout.close()
    return data


def vad_segment_times(input_path, aggressiveness=3, frame_duration_ms=30, padding_ms=300):
    pcm = get_pcm_from_file(input_path)
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frames_from_pcm(pcm, frame_duration_ms=frame_duration_ms))
    sample_rate = 16000
    in_speech = False
    segments = []
    speech_start = 0
    for i, frame in enumerate(frames):
        is_speech = False
        if len(frame) == int(sample_rate * 2 * (frame_duration_ms/1000.0)):
            is_speech = vad.is_speech(frame, sample_rate)
        t = (i * frame_duration_ms) / 1000.0
        if is_speech and not in_speech:
            in_speech = True
            speech_start = t
        elif not is_speech and in_speech:
            in_speech = False
            speech_end = t
            # expand by padding
            start = max(0, speech_start - (padding_ms/1000.0))
            end = speech_end + (padding_ms/1000.0)
            segments.append((start, end))
    # if file ends while in speech
    if in_speech:
        speech_end = (len(frames) * frame_duration_ms) / 1000.0
        start = max(0, speech_start - (padding_ms/1000.0))
        end = speech_end + (padding_ms/1000.0)
        segments.append((start, end))
    return segments


def extract_segment(input_path, start, end, out_path, bitrate="128k"):
    ff = shutil.which("ffmpeg") or None
    if not ff:
        ffexe = None
        if hasattr(iio_ffmpeg, "get_ffmpeg_exe"):
            try:
                ffexe = iio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffexe = None
        if not ffexe and hasattr(iio_ffmpeg, "get_exe"):
            try:
                ffexe = iio_ffmpeg.get_exe()
            except Exception:
                ffexe = None
        ff = ffexe
    cmd = [
        ff,
        "-y",
        "-i",
        input_path,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        out_path,
    ]
    subprocess.run(cmd, check=True)


def clean_fillers(text):
    # simple filler removal
    fillers = [r"\bumm?\b", r"\buh\b", r"\bhello\b", r"\bhi\b", r"\bsorry\b", r"\byou know\b"]
    pattern = re.compile("|".join(fillers), flags=re.IGNORECASE)
    cleaned = pattern.sub("", text)
    # collapse extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def post_process_segments(segments, min_duration=0.5, merge_gap=0.3, max_segments=200):
    """Merge nearby segments, drop very short ones, and cap total segments.
    segments: list of (start,end) tuples sorted by start
    """
    if not segments:
        return []
    # sort
    segs = sorted(segments, key=lambda s: s[0])
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        last = merged[-1]
        # if gap between last end and this start is small, merge
        if s - last[1] <= merge_gap:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    # filter short segments
    filtered = []
    for s, e in merged:
        if (e - s) >= min_duration:
            filtered.append((s, e))
    # cap number of segments
    if len(filtered) > max_segments:
        # keep longest segments
        filtered = sorted(filtered, key=lambda t: t[1]-t[0], reverse=True)[:max_segments]
        filtered = sorted(filtered, key=lambda t: t[0])
    return filtered


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe audio/video to text and docx")
    parser.add_argument("input", help="input audio or video file")
    parser.add_argument("--model", default="base", help="whisper model: tiny, base, small, medium, large")
    parser.add_argument("--preprocess", action="store_true", help="run ffmpeg preprocessing filters")
    parser.add_argument("--keep-temp", action="store_true", help="keep temporary files")
    parser.add_argument("--bitrate", default="192k", help="mp3 bitrate for converted audio")
    parser.add_argument("--device", default="auto", help="device preference: auto/cpu/cuda/mps/dml")
    parser.add_argument("--vad", action="store_true", help="use voice activity detection to segment audio")
    parser.add_argument("--punctuate", action="store_true", help="run punctuation restoration on transcript")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    transcribe_file(args.input, model_name=args.model, preprocess=args.preprocess, keep_temp=args.keep_temp, bitrate=args.bitrate, device_preference=args.device, vad=args.vad, punctuate=args.punctuate)


if __name__ == "__main__":
    main()
