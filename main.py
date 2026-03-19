"""
main.py - OpenClaw Voice Client
Hold a key to record, release to send. Hear the response streamed back as speech.
"""

import json
import os
import sys
import threading

from recorder import Recorder
from transcriber import Transcriber
from client import GatewayClient
from hotkeys import HoldHotkeyBinding, normalize_hotkey_text
from tts import TTSPipeline


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"[error] config.json not found at {CONFIG_PATH}")
        print("Copy config.json.example to config.json and fill in your settings.")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    required = ["gateway_url", "gateway_token", "agent_id", "hotkey"]
    for key in required:
        if not config.get(key):
            print(f"[error] Missing required config key: {key}")
            sys.exit(1)

    return config


def main():
    print("=" * 50)
    print("  OpenClaw Voice Client")
    print("=" * 50)

    config = load_config()

    hotkey = normalize_hotkey_text(config["hotkey"])
    print(f"\nLoading models...")

    recorder = Recorder()
    transcriber = Transcriber(
        model_size=config.get("whisper_model", "base"),
        language=config.get("transcription_language", "auto"),
    )
    gateway = GatewayClient(
        gateway_url=config["gateway_url"],
        token=config["gateway_token"],
        agent_id=config.get("agent_id", "main"),
        session_user=config.get("session_user", "voice-client"),
    )
    tts = TTSPipeline(
        voice=config.get("tts_voice", "en-US-AriaNeural"),
        rate=config.get("tts_rate", "+0%"),
    )

    print(f"\nReady! Hold [{hotkey}] to speak, release to send.")
    print("Press Ctrl+C to quit.\n")

    is_recording = False
    response_cancel_event = None

    def on_hotkey_press(e):
        nonlocal is_recording, response_cancel_event
        if response_cancel_event is not None:
            response_cancel_event.set()
        tts.interrupt()
        if not is_recording:
            is_recording = True
            recorder.start()

    def on_hotkey_release(e):
        nonlocal is_recording
        if is_recording:
            is_recording = False
            threading.Thread(target=process_recording, daemon=True).start()

    def process_recording():
        nonlocal response_cancel_event
        audio_path = recorder.stop()
        if not audio_path:
            print("[main] No audio recorded, skipping.")
            return

        text = transcriber.transcribe(audio_path)
        if not text:
            print("[main] Nothing transcribed, skipping.")
            return

        print(f"\nYou: {text}")
        print("Atlas: ", end="", flush=True)

        # Stream response and pipeline through TTS
        cancel_event = threading.Event()
        response_cancel_event = cancel_event
        buffer = []
        prompt_text = f"voice: {text}"
        for token in gateway.send(prompt_text, stop_event=cancel_event):
            if cancel_event.is_set():
                break
            print(token, end="", flush=True)
            buffer = tts.feed_token(token, buffer)

        # Flush any remaining text
        if not cancel_event.is_set():
            tts.flush(buffer)
        if response_cancel_event is cancel_event:
            response_cancel_event = None
        print("\n")

    # Register hotkey hooks
    hotkey_binding = HoldHotkeyBinding(hotkey, on_hotkey_press, on_hotkey_release)
    hotkey_binding.start()

    try:
        threading.Event().wait()  # Block forever until Ctrl+C
    except KeyboardInterrupt:
        hotkey_binding.stop()
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
