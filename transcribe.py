import sys
import os
import shutil
import subprocess
import tempfile
import argparse
import whisper
from docx import Document
from moviepy import VideoFileClip, AudioFileClip
import imageio_ffmpeg as iio_ffmpeg


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


def transcribe_file(input_path, model_name="base", preprocess=False, keep_temp=False, bitrate="192k"):
    ensure_ffmpeg_in_path()

    temp_files = []
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

        print(f"Loading Whisper model '{model_name}' (this may take a while)...")
        model = whisper.load_model(model_name)

        print(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()

        out_txt = os.path.splitext(audio_path)[0] + ".txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcription saved to {out_txt}")

        doc = Document()
        doc.add_heading("Transcription", 0)
        for para in split_into_paragraphs(text):
            doc.add_paragraph(para)
        out_docx = os.path.splitext(audio_path)[0] + ".docx"
        doc.save(out_docx)
        print(f"Word document saved to {out_docx}")

    finally:
        if not keep_temp:
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass


def main(argv=None):
    p = argparse.ArgumentParser(description="Transcribe audio/video to text and DOCX using Whisper")
    p.add_argument("input", help="Input audio or video file")
    p.add_argument("--model", default="base", help="Whisper model to use: tiny, base, small, medium, large")
    p.add_argument("--preprocess", action="store_true", help="Run ffmpeg audio preprocessing filters before transcription")
    p.add_argument("--keep-temp", action="store_true", help="Keep intermediate converted files")
    p.add_argument("--bitrate", default="192k", help="MP3 bitrate for conversion (e.g., 128k, 192k)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    transcribe_file(args.input, model_name=args.model, preprocess=args.preprocess, keep_temp=args.keep_temp, bitrate=args.bitrate)


if __name__ == "__main__":
    main()
