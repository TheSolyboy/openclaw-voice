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
import time
from queue import Empty, Queue
import sounddevice as sd


SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+|\n+')
CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
URL_RE = re.compile(r"https?://([^\s/]+)[^\s]*")
MAX_CHUNK_SENTENCES = 2
MAX_CHUNK_CHARS = 120


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_ENDINGS.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_tts_text(text: str) -> str:
    """Make model output sound more natural when spoken."""
    text = text.replace("\r", "\n")
    text = CODE_BLOCK_RE.sub(" Code omitted. ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = URL_RE.sub(r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s*", "", text)
    text = re.sub(r"(\*\*|__|\*|_)", "", text)
    text = text.replace("`", "")
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_sentences_for_tts(sentences: list[str]) -> list[str]:
    """Group short adjacent sentences to reduce dead air between clips."""
    chunks = []
    current_chunk = []

    for sentence in sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue

        candidate = " ".join(current_chunk + [cleaned]).strip()
        if current_chunk and (
            len(current_chunk) >= MAX_CHUNK_SENTENCES or len(candidate) > MAX_CHUNK_CHARS
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = [cleaned]
        else:
            current_chunk.append(cleaned)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def cleanup_temp_file(path: str, retries: int = 10, delay: float = 0.1) -> bool:
    """Retry temp-file removal to avoid transient Windows file locks."""
    for _ in range(retries):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(delay)
    return False


class TTSPipeline:
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", output_device=None):
        self.voice = voice
        self.rate = rate
        self._audio_queue = Queue()
        self._generation = 0
        self._next_sequence = 0
        self._generation_lock = threading.Lock()
        self._init_mixer(output_device)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        print(f"[tts] Initialized with voice: {voice}, rate: {rate}")

    def _init_mixer(self, output_device):
        """Try the selected output device first, then fall back to SDL's default."""
        if output_device is not None:
            devices = sd.query_devices()
            if 0 <= output_device < len(devices):
                device_name = devices[output_device]["name"]
                try:
                    pygame.mixer.pre_init(devicename=device_name)
                    pygame.mixer.init()
                    print(f"[tts] Using output device: {device_name}")
                    return
                except pygame.error as e:
                    print(f"[tts] Output device '{device_name}' unavailable: {e}. Falling back to default device.")
                    try:
                        pygame.mixer.quit()
                    except Exception:
                        pass
            else:
                print(f"[tts] Saved output device index {output_device} is no longer valid. Falling back to default device.")

        pygame.mixer.pre_init()
        pygame.mixer.init()

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
                    for chunk in chunk_sentences_for_tts(sentences):
                        self._enqueue_sentence(chunk)
                    return []
                else:
                    for chunk in chunk_sentences_for_tts(sentences[:-1]):
                        self._enqueue_sentence(chunk)
                    return [sentences[-1]]

        return buffer

    def flush(self, buffer: list):
        """Flush any remaining text in the buffer."""
        text = "".join(buffer).strip()
        if text:
            self._enqueue_sentence(text)

    def interrupt(self):
        """Stop current playback and drop any queued audio from the interrupted reply."""
        with self._generation_lock:
            self._generation += 1
            self._next_sequence = 0
        self._clear_queue()
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
        except Exception:
            pass

    def _current_generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def _reserve_sequence(self) -> tuple[int, int]:
        with self._generation_lock:
            generation_id = self._generation
            sequence_id = self._next_sequence
            self._next_sequence += 1
        return generation_id, sequence_id

    def _clear_queue(self):
        while True:
            try:
                _, _, audio_path = self._audio_queue.get_nowait()
            except Empty:
                return
            cleanup_temp_file(audio_path)
            self._audio_queue.task_done()

    def _enqueue_sentence(self, sentence: str):
        """Generate audio for a sentence and queue it for playback."""
        sentence = normalize_tts_text(sentence)
        if not sentence:
            return
        print(f"[tts] Generating: {sentence!r}")
        generation_id, sequence_id = self._reserve_sequence()
        # Generate in a background thread so we don't block token streaming
        threading.Thread(
            target=self._generate_and_queue,
            args=(sentence, generation_id, sequence_id),
            daemon=True,
        ).start()

    def _generate_and_queue(self, sentence: str, generation_id: int, sequence_id: int):
        """Generate TTS audio and put the file path in the playback queue."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()

            asyncio.run(self._generate_audio(sentence, tmp.name))
            if generation_id != self._current_generation():
                cleanup_temp_file(tmp.name)
                return
            self._audio_queue.put((generation_id, sequence_id, tmp.name))
        except Exception as e:
            print(f"[tts] Error generating audio: {e}")

    async def _generate_audio(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(output_path)

    def _playback_loop(self):
        """Continuously play audio files from the queue in order."""
        active_generation = None
        expected_sequence = 0
        pending_audio = {}

        while True:
            generation_id, sequence_id, audio_path = self._audio_queue.get()
            try:
                if generation_id != self._current_generation():
                    continue

                if active_generation != generation_id:
                    for stale_path in pending_audio.values():
                        cleanup_temp_file(stale_path)
                    pending_audio = {}
                    active_generation = generation_id
                    expected_sequence = 0

                pending_audio[sequence_id] = audio_path

                while expected_sequence in pending_audio:
                    next_audio_path = pending_audio.pop(expected_sequence)
                    try:
                        pygame.mixer.music.load(next_audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            if generation_id != self._current_generation():
                                pygame.mixer.music.stop()
                                break
                            pygame.time.wait(50)
                    finally:
                        cleanup_temp_file(next_audio_path)
                    expected_sequence += 1
            except Exception as e:
                print(f"[tts] Playback error: {e}")
            finally:
                try:
                    pygame.mixer.music.stop()
                    if hasattr(pygame.mixer.music, "unload"):
                        pygame.mixer.music.unload()
                except Exception:
                    pass
            self._audio_queue.task_done()

    def wait_until_done(self):
        """Block until all queued audio has finished playing."""
        self._audio_queue.join()
