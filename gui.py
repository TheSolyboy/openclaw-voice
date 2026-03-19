"""
gui.py - OpenClaw Voice Client UI
Modern settings + status interface built with customtkinter.
"""

import json
import os
import sys
import threading
import time
import traceback
import tkinter as tk
import customtkinter as ctk
import sounddevice as sd
import edge_tts
import asyncio

from recorder import Recorder
from transcriber import Transcriber
from client import GatewayClient, describe_gateway_status, resolve_chat_completions_url
from hotkeys import HoldHotkeyBinding, format_hotkey, normalize_hotkey_text, normalize_key_name
from tts import TTSPipeline


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "config.json.example")

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

DEFAULT_CONFIG = {
    "gateway_url": "",
    "gateway_token": "",
    "agent_id": "main",
    "hotkey": "F13",
    "whisper_model": "base",
    "transcription_language": "auto",
    "tts_voice": "en-US-AriaNeural",
    "tts_rate": "+0%",
    "session_user": "voice-client",
    "input_device": None,
    "output_device": None,
}

POPULAR_VOICES = [
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-EricNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
    "en-CA-ClaraNeural",
    "nb-NO-PernilleNeural",
    "nb-NO-FinnNeural",
]

TRANSCRIPTION_LANGUAGE_OPTIONS = {
    "Auto detect": "auto",
    "English": "en",
    "Norwegian": "no",
}

TRANSCRIPTION_LANGUAGE_LABELS = list(TRANSCRIPTION_LANGUAGE_OPTIONS.keys())
TRANSCRIPTION_LANGUAGE_BY_CODE = {value: label for label, value in TRANSCRIPTION_LANGUAGE_OPTIONS.items()}


def get_preview_text(voice: str) -> str:
    if voice.startswith("nb-NO"):
        return "Hei, dette er en norsk stemmeprove."
    return "Hello, this is a voice preview."


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        return {**DEFAULT_CONFIG, **data}
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_audio_devices():
    """Return (input_devices, output_devices) as lists of (index, name)."""
    devices = sd.query_devices()
    inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    outputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
    return inputs, outputs


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


class VoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OpenClaw Voice")
        self.geometry("560x700")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config_data = load_config()
        self._voice_running = False
        self._recorder = None
        self._transcriber = None
        self._gateway = None
        self._tts = None
        self._hotkey_binding = None
        self._hotkey_capture_hook = None
        self._captured_hotkey_keys = []
        self._captured_pressed_keys = set()
        self._response_cancel_event = None
        self._is_recording = False
        self._starting = False
        self._record_started_at = 0.0
        self._release_after_id = None

        self._build_ui()
        self._load_devices()
        self._populate_fields()

    # ─── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 0))

        ctk.CTkLabel(header, text="🌐 OpenClaw Voice", font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        self._status_dot = ctk.CTkLabel(header, text="●", font=ctk.CTkFont(size=18), text_color="#555")
        self._status_dot.pack(side="right", padx=(0, 4))
        self._status_label = ctk.CTkLabel(header, text="Stopped", text_color="#888")
        self._status_label.pack(side="right")

        # Tab view
        self._tabs = ctk.CTkTabview(self, height=520)
        self._tabs.pack(fill="both", expand=True, padx=20, pady=12)

        self._tabs.add("Voice")
        self._tabs.add("Audio")
        self._tabs.add("Gateway")

        self._build_voice_tab()
        self._build_audio_tab()
        self._build_gateway_tab()

        # Bottom buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        self._save_btn = ctk.CTkButton(btn_row, text="Save Settings", command=self._save_settings)
        self._save_btn.pack(side="left")

        self._start_btn = ctk.CTkButton(
            btn_row, text="▶  Start", width=120,
            fg_color="#1f6aa5", hover_color="#144d7a",
            command=self._toggle_voice
        )
        self._start_btn.pack(side="right")

    def _build_voice_tab(self):
        tab = self._tabs.tab("Voice")

        # Push-to-talk indicator
        indicator_frame = ctk.CTkFrame(tab, fg_color=("#1a1a2e", "#1a1a2e"), corner_radius=12)
        indicator_frame.pack(fill="x", padx=4, pady=(12, 8))

        self._mic_label = ctk.CTkLabel(
            indicator_frame, text="🎙  Hold your hotkey to speak",
            font=ctk.CTkFont(size=14), text_color="#888"
        )
        self._mic_label.pack(pady=20)

        # Transcript display
        ctk.CTkLabel(tab, text="Last transcript", text_color="#888", anchor="w").pack(fill="x", padx=4)
        self._transcript_box = ctk.CTkTextbox(tab, height=60, state="disabled", text_color="#ccc")
        self._transcript_box.pack(fill="x", padx=4, pady=(2, 10))

        # Response display
        ctk.CTkLabel(tab, text="Last response", text_color="#888", anchor="w").pack(fill="x", padx=4)
        self._response_box = ctk.CTkTextbox(tab, height=180, state="disabled", text_color="#ccc")
        self._response_box.pack(fill="x", padx=4, pady=(2, 4))

        # Hotkey setting
        hk_row = ctk.CTkFrame(tab, fg_color="transparent")
        hk_row.pack(fill="x", padx=4, pady=8)
        ctk.CTkLabel(hk_row, text="Hotkey", width=100, anchor="w").pack(side="left")
        self._hotkey_entry = ctk.CTkEntry(hk_row, placeholder_text="F13")
        self._hotkey_entry.pack(side="left", fill="x", expand=True)
        self._record_hotkey_btn = ctk.CTkButton(hk_row, text="Record", width=90, command=self._toggle_hotkey_capture)
        self._record_hotkey_btn.pack(side="left", padx=(8, 0))

        # Whisper model
        wm_row = ctk.CTkFrame(tab, fg_color="transparent")
        wm_row.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(wm_row, text="Whisper model", width=100, anchor="w").pack(side="left")
        self._whisper_menu = ctk.CTkOptionMenu(wm_row, values=WHISPER_MODELS)
        self._whisper_menu.pack(side="left", fill="x", expand=True)

        lang_row = ctk.CTkFrame(tab, fg_color="transparent")
        lang_row.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(lang_row, text="Transcribe", width=100, anchor="w").pack(side="left")
        self._transcription_language_menu = ctk.CTkOptionMenu(lang_row, values=TRANSCRIPTION_LANGUAGE_LABELS)
        self._transcription_language_menu.pack(side="left", fill="x", expand=True)

    def _build_audio_tab(self):
        tab = self._tabs.tab("Audio")

        # Input device
        ctk.CTkLabel(tab, text="Microphone (input)", anchor="w").pack(fill="x", padx=4, pady=(12, 2))
        self._input_menu = ctk.CTkOptionMenu(tab, values=["Loading..."], dynamic_resizing=False, width=400)
        self._input_menu.pack(fill="x", padx=4, pady=(0, 12))

        # Output device
        ctk.CTkLabel(tab, text="Speakers (output)", anchor="w").pack(fill="x", padx=4, pady=(4, 2))
        self._output_menu = ctk.CTkOptionMenu(tab, values=["Loading..."], dynamic_resizing=False, width=400)
        self._output_menu.pack(fill="x", padx=4, pady=(0, 16))

        ctk.CTkButton(tab, text="↻  Refresh devices", command=self._load_devices, width=140).pack(anchor="w", padx=4)

        # TTS voice
        ctk.CTkLabel(tab, text="TTS Voice", anchor="w").pack(fill="x", padx=4, pady=(20, 2))
        self._voice_menu = ctk.CTkOptionMenu(tab, values=POPULAR_VOICES, dynamic_resizing=False, width=400)
        self._voice_menu.pack(fill="x", padx=4)

        ctk.CTkButton(tab, text="▶  Preview voice", command=self._preview_voice, width=140).pack(anchor="w", padx=4, pady=8)

        # TTS rate
        ctk.CTkLabel(tab, text="Speech speed", anchor="w").pack(fill="x", padx=4, pady=(4, 2))
        rate_row = ctk.CTkFrame(tab, fg_color="transparent")
        rate_row.pack(fill="x", padx=4)

        self._rate_slider = ctk.CTkSlider(rate_row, from_=-50, to=50, number_of_steps=20, command=self._update_rate_label)
        self._rate_slider.set(0)
        self._rate_slider.pack(side="left", fill="x", expand=True)

        self._rate_label = ctk.CTkLabel(rate_row, text="+0%", width=50)
        self._rate_label.pack(side="right")

    def _build_gateway_tab(self):
        tab = self._tabs.tab("Gateway")

        fields = [
            ("Gateway URL", "_gw_url", "https://your-openclaw-gateway.example.com", False),
            ("Gateway auth", "_gw_token", "token or password", True),
            ("Agent ID", "_gw_agent", "main", False),
            ("Session user", "_gw_session", "voice-client", False),
        ]

        for label, attr, placeholder, secret in fields:
            ctk.CTkLabel(tab, text=label, anchor="w").pack(fill="x", padx=4, pady=(12, 2))
            entry = ctk.CTkEntry(
                tab, placeholder_text=placeholder,
                show="●" if secret else "",
                width=400
            )
            entry.pack(fill="x", padx=4)
            setattr(self, attr, entry)

        # Show/hide token button
        ctk.CTkButton(tab, text="Show/hide token", width=130, command=self._toggle_token).pack(anchor="w", padx=4, pady=6)

        ctk.CTkLabel(
            tab,
            text="Use the gateway token by default. If your gateway is set to password mode, enter that password here.",
            text_color="#888",
            justify="left",
            wraplength=480,
        ).pack(anchor="w", padx=4, pady=(0, 4))

        # Connection test button
        ctk.CTkButton(tab, text="🔗  Test connection", command=self._test_connection).pack(anchor="w", padx=4, pady=(16, 4))
        self._conn_label = ctk.CTkLabel(tab, text="", text_color="#888", justify="left", wraplength=480)
        self._conn_label.pack(anchor="w", padx=4)

    # ─── Device Loading ──────────────────────────────────────────────────────

    def _load_devices(self):
        self._input_devices, self._output_devices = get_audio_devices()

        input_names = [f"{i}: {n}" for i, n in self._input_devices]
        output_names = [f"{i}: {n}" for i, n in self._output_devices]

        self._input_menu.configure(values=input_names or ["No input devices found"])
        self._output_menu.configure(values=output_names or ["No output devices found"])

        # Restore saved selections
        saved_in = self.config_data.get("input_device")
        saved_out = self.config_data.get("output_device")

        if saved_in is not None:
            match = next((f"{i}: {n}" for i, n in self._input_devices if i == saved_in), None)
            if match:
                self._input_menu.set(match)
            elif input_names:
                self._input_menu.set(input_names[0])
        elif input_names:
            # Default: system default
            default = sd.query_devices(kind="input")
            idx = next((i for i, (di, _) in enumerate(self._input_devices) if di == sd.default.device[0]), 0)
            self._input_menu.set(input_names[idx] if idx < len(input_names) else input_names[0])

        if saved_out is not None:
            match = next((f"{i}: {n}" for i, n in self._output_devices if i == saved_out), None)
            if match:
                self._output_menu.set(match)
            elif output_names:
                self._output_menu.set(output_names[0])
        elif output_names:
            default = sd.query_devices(kind="output")
            idx = next((i for i, (di, _) in enumerate(self._output_devices) if di == sd.default.device[1]), 0)
            self._output_menu.set(output_names[idx] if idx < len(output_names) else output_names[0])

    # ─── Field Population ────────────────────────────────────────────────────

    def _populate_fields(self):
        c = self.config_data

        self._hotkey_entry.insert(0, normalize_hotkey_text(c.get("hotkey", "F13")))
        self._whisper_menu.set(c.get("whisper_model", "base"))
        language_code = c.get("transcription_language", "auto")
        self._transcription_language_menu.set(TRANSCRIPTION_LANGUAGE_BY_CODE.get(language_code, "Auto detect"))

        voice = c.get("tts_voice", "en-US-AriaNeural")
        if voice not in POPULAR_VOICES:
            POPULAR_VOICES.append(voice)
            self._voice_menu.configure(values=POPULAR_VOICES)
        self._voice_menu.set(voice)

        # Parse rate string like "+20%" → 20
        rate_str = c.get("tts_rate", "+0%").replace("%", "")
        try:
            rate_val = int(rate_str)
        except ValueError:
            rate_val = 0
        self._rate_slider.set(rate_val)
        self._rate_label.configure(text=f"{rate_val:+d}%")

        self._gw_url.insert(0, c.get("gateway_url", ""))
        self._gw_token.insert(0, c.get("gateway_token", ""))
        self._gw_agent.insert(0, c.get("agent_id", "main"))
        self._gw_session.insert(0, c.get("session_user", "voice-client"))

    # ─── Actions ─────────────────────────────────────────────────────────────

    def _update_rate_label(self, val):
        v = int(float(val))
        self._rate_label.configure(text=f"{v:+d}%")

    def _toggle_token(self):
        current = self._gw_token.cget("show")
        self._gw_token.configure(show="" if current == "●" else "●")

    def _toggle_hotkey_capture(self):
        if self._hotkey_capture_hook is not None:
            self._stop_hotkey_capture()
            return

        if self._voice_running or self._starting:
            self._set_status("Stop voice mode before recording a new hotkey", "#ff9800")
            return

        import keyboard as kb

        self._captured_hotkey_keys = []
        self._captured_pressed_keys = set()
        self._hotkey_capture_hook = kb.hook(self._handle_hotkey_capture, suppress=False)
        self._record_hotkey_btn.configure(text="Press keys...")
        self._set_status("Press the desired hotkey combo and release", "#ff9800")

    def _stop_hotkey_capture(self):
        import keyboard as kb

        if self._hotkey_capture_hook is not None:
            kb.unhook(self._hotkey_capture_hook)
            self._hotkey_capture_hook = None
        self._record_hotkey_btn.configure(text="Record")

    def _handle_hotkey_capture(self, event):
        key_name = normalize_key_name(event.name)
        if not key_name:
            return

        if event.event_type == "down":
            self._captured_pressed_keys.add(key_name)
            if key_name not in self._captured_hotkey_keys:
                self._captured_hotkey_keys.append(key_name)
        elif event.event_type == "up":
            self._captured_pressed_keys.discard(key_name)

        if self._captured_hotkey_keys and not self._captured_pressed_keys:
            hotkey = format_hotkey(self._captured_hotkey_keys)
            self.after(0, lambda value=hotkey: self._finish_hotkey_capture(value))

    def _finish_hotkey_capture(self, hotkey):
        self._stop_hotkey_capture()
        self._hotkey_entry.delete(0, "end")
        self._hotkey_entry.insert(0, hotkey)
        self._set_status(f"Hotkey recorded: {hotkey}", "#4caf50")

    def _get_selected_device_index(self, menu, device_list):
        val = menu.get()
        try:
            idx = int(val.split(":")[0])
            return idx
        except (ValueError, IndexError):
            return None

    def _save_settings(self):
        rate_val = int(float(self._rate_slider.get()))
        rate_str = f"{rate_val:+d}%"

        config = {
            "gateway_url": self._gw_url.get().strip(),
            "gateway_token": self._gw_token.get().strip(),
            "agent_id": self._gw_agent.get().strip() or "main",
            "hotkey": normalize_hotkey_text(self._hotkey_entry.get().strip() or "F13"),
            "whisper_model": self._whisper_menu.get(),
            "transcription_language": TRANSCRIPTION_LANGUAGE_OPTIONS[self._transcription_language_menu.get()],
            "tts_voice": self._voice_menu.get(),
            "tts_rate": rate_str,
            "session_user": self._gw_session.get().strip() or "voice-client",
            "input_device": self._get_selected_device_index(self._input_menu, self._input_devices),
            "output_device": self._get_selected_device_index(self._output_menu, self._output_devices),
        }

        save_config(config)
        self.config_data = config
        self._hotkey_entry.delete(0, "end")
        self._hotkey_entry.insert(0, config["hotkey"])
        self._set_status("Settings saved", "#4caf50")

    def _preview_voice(self):
        voice = self._voice_menu.get()
        rate_val = int(float(self._rate_slider.get()))
        rate_str = f"{rate_val:+d}%"

        def run():
            import tempfile, pygame
            tmp_path = None
            started_mixer = False
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp.close()
                tmp_path = tmp.name
                asyncio.run(self._generate_preview(voice, rate_str, tmp_path))
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                    started_mixer = True
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[preview] Error: {e}")
            finally:
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.music.stop()
                        if hasattr(pygame.mixer.music, "unload"):
                            pygame.mixer.music.unload()
                except Exception:
                    pass

                if started_mixer:
                    try:
                        pygame.mixer.quit()
                    except Exception:
                        pass

                if tmp_path and not cleanup_temp_file(tmp_path):
                    print(f"[preview] Warning: could not delete temp file: {tmp_path}")

        threading.Thread(target=run, daemon=True).start()

    async def _generate_preview(self, voice, rate, path):
        comm = edge_tts.Communicate(get_preview_text(voice), voice, rate=rate)
        await comm.save(path)

    def _test_connection(self):
        url = self._gw_url.get().strip()
        token = self._gw_token.get().strip()
        agent = self._gw_agent.get().strip() or "main"

        if not url or not token:
            self._conn_label.configure(text="Fill in the gateway URL and auth field first", text_color="#ff9800")
            return

        endpoint = resolve_chat_completions_url(url)
        self._conn_label.configure(text=f"Testing {endpoint} ...", text_color="#888")

        def run():
            import httpx
            try:
                resp = httpx.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {token}", "x-openclaw-agent-id": agent},
                    json={"model": "openclaw", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    self.after(0, lambda: self._conn_label.configure(text=f"Connected: {endpoint}", text_color="#4caf50"))
                else:
                    message = describe_gateway_status(resp.status_code)
                    self.after(0, lambda: self._conn_label.configure(text=message, text_color="#f44336"))
            except Exception as e:
                self.after(0, lambda: self._conn_label.configure(text=f"Could not reach gateway: {e}", text_color="#f44336"))

        threading.Thread(target=run, daemon=True).start()

    def _toggle_voice(self):
        if self._starting:
            return
        if self._voice_running:
            self._stop_voice()
        else:
            self._start_voice()

    def _start_voice(self):
        self._save_settings()
        c = self.config_data

        if not c.get("gateway_url") or not c.get("gateway_token"):
            self._set_status("Set the gateway URL and auth field first", "#f44336")
            return

        self._starting = True
        self._start_btn.configure(state="disabled", text="Loading...")
        self._set_status("Initializing models and audio... first run may take a few minutes", "#ff9800")
        threading.Thread(target=self._initialize_voice_components, args=(c,), daemon=True).start()

    def _initialize_voice_components(self, config):
        try:
            recorder = Recorder(device=config.get("input_device"))
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
                output_device=config.get("output_device"),
            )
            hotkey = config.get("hotkey", "F13")
            self.after(0, lambda: self._finish_start_voice(recorder, transcriber, gateway, tts, hotkey))
        except Exception as e:
            print("[start] Initialization failed:")
            print(traceback.format_exc())
            error_message = str(e)
            self.after(0, lambda message=error_message: self._handle_start_failure(message))

    def _finish_start_voice(self, recorder, transcriber, gateway, tts, hotkey):
        try:
            self._recorder = recorder
            self._transcriber = transcriber
            self._gateway = gateway
            self._tts = tts
            self._voice_running = True
            self._hotkey_binding = HoldHotkeyBinding(hotkey, self._on_key_press, self._on_key_release)
            self._hotkey_binding.start()
        except Exception as e:
            print("[start] Hotkey hook failed:")
            print(traceback.format_exc())
            self._handle_start_failure(str(e))
            return

        self._starting = False
        self._start_btn.configure(state="normal", text="⏹  Stop", fg_color="#b71c1c", hover_color="#7f1212")
        self._set_status(f"Listening ({hotkey})", "#4caf50")

    def _handle_start_failure(self, message):
        self._starting = False
        self._voice_running = False
        if self._hotkey_binding is not None:
            self._hotkey_binding.stop()
            self._hotkey_binding = None
        self._start_btn.configure(state="normal", text="▶  Start", fg_color="#1f6aa5", hover_color="#144d7a")
        self._set_status(f"Init error: {message}", "#f44336")

    def _stop_voice(self):
        self._starting = False
        self._voice_running = False
        self._interrupt_response_playback()
        if self._release_after_id is not None:
            self.after_cancel(self._release_after_id)
            self._release_after_id = None
        if self._hotkey_binding is not None:
            self._hotkey_binding.stop()
            self._hotkey_binding = None
        self._start_btn.configure(state="normal", text="▶  Start", fg_color="#1f6aa5", hover_color="#144d7a")
        self._set_status("Stopped", "#888")
        self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888")

    def _on_key_press(self, e):
        self._interrupt_response_playback()
        if self._release_after_id is not None:
            self.after_cancel(self._release_after_id)
            self._release_after_id = None
        if not self._is_recording and self._voice_running:
            self._is_recording = True
            self._record_started_at = time.monotonic()
            self.after(0, lambda: self._mic_label.configure(text="🔴  Recording...", text_color="#f44336"))
            self._recorder.start()

    def _on_key_release(self, e):
        if self._is_recording and self._release_after_id is None:
            started_at = self._record_started_at
            self._release_after_id = self.after(60, lambda s=started_at: self._confirm_key_release(s))
            return

    def _confirm_key_release(self, started_at):
        import keyboard as kb

        self._release_after_id = None
        if not self._is_recording or started_at != self._record_started_at:
            return

        hotkey = self.config_data.get("hotkey", "F13")
        try:
            if kb.is_pressed(hotkey):
                return
        except Exception:
            pass

        self._is_recording = False
        self.after(0, lambda: self._mic_label.configure(text="Processing...", text_color="#ff9800"))
        threading.Thread(target=self._process_recording, daemon=True).start()

    def _interrupt_response_playback(self):
        if self._response_cancel_event is not None:
            self._response_cancel_event.set()
        if self._tts is not None:
            self._tts.interrupt()

    def _process_recording(self):
        audio_path = self._recorder.stop()
        if not audio_path:
            self.after(0, lambda: self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888"))
            return

        text = self._transcriber.transcribe(audio_path)
        if not text:
            self.after(0, lambda: self._mic_label.configure(text="🎙  (nothing heard)", text_color="#888"))
            return

        self._set_transcript(text)
        self.after(0, lambda: self._mic_label.configure(text="💬  Waiting for response...", text_color="#2196f3"))

        cancel_event = threading.Event()
        self._response_cancel_event = cancel_event
        response_parts = []
        buffer = []
        prompt_text = f"voice: {text}"
        for token in self._gateway.send(prompt_text, stop_event=cancel_event):
            if cancel_event.is_set():
                break
            response_parts.append(token)
            buffer = self._tts.feed_token(token, buffer)
            self._append_response("".join(response_parts))

        if not cancel_event.is_set():
            self._tts.flush(buffer)
            self.after(0, lambda: self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888"))
        elif not self._is_recording:
            self.after(0, lambda: self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888"))

        if self._response_cancel_event is cancel_event:
            self._response_cancel_event = None

    # ─── UI Helpers ──────────────────────────────────────────────────────────

    def _set_status(self, text, color):
        self.after(0, lambda: self._status_label.configure(text=text, text_color=color))
        self.after(0, lambda: self._status_dot.configure(text_color=color))

    def _set_transcript(self, text):
        def _update():
            self._transcript_box.configure(state="normal")
            self._transcript_box.delete("1.0", "end")
            self._transcript_box.insert("1.0", text)
            self._transcript_box.configure(state="disabled")
        self.after(0, _update)

    def _append_response(self, text):
        def _update():
            self._response_box.configure(state="normal")
            self._response_box.delete("1.0", "end")
            self._response_box.insert("1.0", text)
            self._response_box.see("end")
            self._response_box.configure(state="disabled")
        self.after(0, _update)


def main():
    app = VoiceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
