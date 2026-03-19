"""
transcriber.py - Local speech-to-text using faster-whisper
"""

import os
from faster_whisper import WhisperModel


LANGUAGE_HINTS = {
    "en": {
        "initial_prompt": (
            "This is natural spoken English with technical words like OpenClaw, Solynex, "
            "Coolify, dashstats, and dashstats.net."
        ),
        "hotwords": "OpenClaw, Solynex, Coolify, dashstats, dashstats.net",
    },
    "no": {
        "initial_prompt": (
            "Dette er naturlig muntlig norsk. Vanlige uttrykk kan vaere hallo, hallais, halla, "
            "hei, hvordan gaar det, koss gaar det, det gaar bra, ikke, og, jeg, du, OpenClaw, "
            "Solynex, Coolify, dashstats, og dashstats.net."
        ),
        "hotwords": (
            "halla, hallais, hei, hallo, koss gaar det, hvordan gaar det, OpenClaw, Solynex, "
            "Coolify, dashstats, dashstats.net"
        ),
    },
}


class Transcriber:
    def __init__(self, model_size="base", language="auto"):
        self.language = None if not language or language == "auto" else language
        language_label = self.language or "auto"
        print(f"[transcriber] Loading Whisper model: {model_size} (language: {language_label})")
        # device="auto" uses CUDA if available, falls back to CPU
        self.model = WhisperModel(model_size, device="auto", compute_type="auto")
        print("[transcriber] Model ready.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a WAV file and return the text."""
        if not audio_path or not os.path.exists(audio_path):
            return ""

        kwargs = {"beam_size": 5}
        if self.language:
            kwargs["language"] = self.language
            hints = LANGUAGE_HINTS.get(self.language)
            if hints:
                kwargs["initial_prompt"] = hints["initial_prompt"]
                kwargs["hotwords"] = hints["hotwords"]

        segments, info = self.model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()

        # Clean up temp file
        try:
            os.unlink(audio_path)
        except Exception:
            pass

        print(f"[transcriber] Transcribed: {text!r}")
        return text
