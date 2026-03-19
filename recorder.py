"""
recorder.py - Audio capture using sounddevice
Records audio while a key is held, stops on release.
"""

import sounddevice as sd
import numpy as np
import tempfile
import os
import soundfile as sf


SAMPLE_RATE = 16000  # 16kHz mono — optimal for Whisper


class Recorder:
    def __init__(self, device=None):
        self._frames = []
        self._recording = False
        self._stream = None
        self._device = device  # None = system default

    def start(self):
        """Start recording from the selected microphone."""
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        print("[recorder] Recording started...")

    def stop(self):
        """Stop recording and return path to a temp WAV file."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            print("[recorder] No audio captured.")
            return None

        audio = np.concatenate(self._frames, axis=0)

        # Write to a temp WAV file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        print(f"[recorder] Saved audio to {tmp.name}")
        return tmp.name

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"[recorder] Warning: {status}")
        if self._recording:
            self._frames.append(indata.copy())
