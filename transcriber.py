"""
transcriber.py - Local speech-to-text using faster-whisper
"""

import os
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="base"):
        print(f"[transcriber] Loading Whisper model: {model_size}")
        # device="auto" uses CUDA if available, falls back to CPU
        self.model = WhisperModel(model_size, device="auto", compute_type="auto")
        print("[transcriber] Model ready.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a WAV file and return the text."""
        if not audio_path or not os.path.exists(audio_path):
            return ""

        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        # Clean up temp file
        try:
            os.unlink(audio_path)
        except Exception:
            pass

        print(f"[transcriber] Transcribed: {text!r}")
        return text
