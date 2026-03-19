"""
tts.py - Text-to-speech using edge-tts with pipelined pygame playback
Converts sentences to audio as they arrive and plays them in order.
"""

import asyncio
import edge_tts
import pygame
import tempfile
import os
import re
import threading
from queue import Queue


# Split text into sentences on . ! ?
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_ENDINGS.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class TTSPipeline:
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", output_device=None):
        self.voice = voice
        self.rate = rate
        self._audio_queue = Queue()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

        # Set output device before init if specified
        if output_device is not None:
            devices = sd.query_devices()
            if output_device < len(devices):
                device_name = devices[output_device]["name"]
                pygame.mixer.pre_init(devicename=device_name)

        pygame.mixer.init()
        print(f"[tts] Initialized with voice: {voice}, rate: {rate}")

    def feed_token(self, token: str, buffer: list) -> list:
        """
        Accumulate tokens. When a sentence boundary is detected, 
        generate TTS for the buffered sentence and return the remainder.
        """
        buffer.append(token)
        combined = "".join(buffer)

        # Check if we have a complete sentence
        if re.search(r'[.!?]\s', combined) or combined.endswith(('.', '!', '?')):
            sentences = split_sentences(combined)
            if len(sentences) >= 1:
                # Send all but possibly the last incomplete sentence
                # If combined ends with punctuation, send all
                if combined[-1] in '.!?':
                    for s in sentences:
                        self._enqueue_sentence(s)
                    return []
                else:
                    for s in sentences[:-1]:
                        self._enqueue_sentence(s)
                    return list(sentences[-1])

        return buffer

    def flush(self, buffer: list):
        """Flush any remaining text in the buffer."""
        text = "".join(buffer).strip()
        if text:
            self._enqueue_sentence(text)

    def _enqueue_sentence(self, sentence: str):
        """Generate audio for a sentence and queue it for playback."""
        sentence = sentence.strip()
        if not sentence:
            return
        print(f"[tts] Generating: {sentence!r}")
        # Generate in a background thread so we don't block token streaming
        threading.Thread(target=self._generate_and_queue, args=(sentence,), daemon=True).start()

    def _generate_and_queue(self, sentence: str):
        """Generate TTS audio and put the file path in the playback queue."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()

            asyncio.run(self._generate_audio(sentence, tmp.name))
            self._audio_queue.put(tmp.name)
        except Exception as e:
            print(f"[tts] Error generating audio: {e}")

    async def _generate_audio(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(output_path)

    def _playback_loop(self):
        """Continuously play audio files from the queue in order."""
        while True:
            audio_path = self._audio_queue.get()
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
            except Exception as e:
                print(f"[tts] Playback error: {e}")
            finally:
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
            self._audio_queue.task_done()

    def wait_until_done(self):
        """Block until all queued audio has finished playing."""
        self._audio_queue.join()
