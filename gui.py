"""
gui.py - OpenClaw Voice Client UI
Modern settings + status interface built with customtkinter.
"""

import json
import os
import sys
import threading
import tkinter as tk
import customtkinter as ctk
import sounddevice as sd
import edge_tts
import asyncio

from recorder import Recorder
from transcriber import Transcriber
from client import GatewayClient
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
]


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
        self._hotkey_thread = None
        self._is_recording = False

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

        # Whisper model
        wm_row = ctk.CTkFrame(tab, fg_color="transparent")
        wm_row.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(wm_row, text="Whisper model", width=100, anchor="w").pack(side="left")
        self._whisper_menu = ctk.CTkOptionMenu(wm_row, values=WHISPER_MODELS)
        self._whisper_menu.pack(side="left", fill="x", expand=True)

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
            ("Gateway token", "_gw_token", "your-token-here", True),
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

        # Connection test button
        ctk.CTkButton(tab, text="🔗  Test connection", command=self._test_connection).pack(anchor="w", padx=4, pady=(16, 4))
        self._conn_label = ctk.CTkLabel(tab, text="", text_color="#888")
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

        self._hotkey_entry.insert(0, c.get("hotkey", "F13"))
        self._whisper_menu.set(c.get("whisper_model", "base"))

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
            "hotkey": self._hotkey_entry.get().strip() or "F13",
            "whisper_model": self._whisper_menu.get(),
            "tts_voice": self._voice_menu.get(),
            "tts_rate": rate_str,
            "session_user": self._gw_session.get().strip() or "voice-client",
            "input_device": self._get_selected_device_index(self._input_menu, self._input_devices),
            "output_device": self._get_selected_device_index(self._output_menu, self._output_devices),
        }

        save_config(config)
        self.config_data = config
        self._set_status("Settings saved", "#4caf50")

    def _preview_voice(self):
        voice = self._voice_menu.get()
        rate_val = int(float(self._rate_slider.get()))
        rate_str = f"{rate_val:+d}%"

        def run():
            import tempfile, pygame
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp.close()
                asyncio.run(self._generate_preview(voice, rate_str, tmp.name))
                pygame.mixer.init()
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    import time; time.sleep(0.05)
                os.unlink(tmp.name)
            except Exception as e:
                print(f"[preview] Error: {e}")

        threading.Thread(target=run, daemon=True).start()

    async def _generate_preview(self, voice, rate, path):
        comm = edge_tts.Communicate("Hello, this is a voice preview.", voice, rate=rate)
        await comm.save(path)

    def _test_connection(self):
        url = self._gw_url.get().strip()
        token = self._gw_token.get().strip()
        agent = self._gw_agent.get().strip() or "main"

        if not url or not token:
            self._conn_label.configure(text="⚠  Fill in gateway URL and token first", text_color="#ff9800")
            return

        self._conn_label.configure(text="Testing...", text_color="#888")

        def run():
            import httpx
            try:
                resp = httpx.post(
                    url.rstrip("/") + "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}", "x-openclaw-agent-id": agent},
                    json={"model": "openclaw", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    self.after(0, lambda: self._conn_label.configure(text="✅  Connected!", text_color="#4caf50"))
                else:
                    self.after(0, lambda: self._conn_label.configure(text=f"❌  HTTP {resp.status_code}", text_color="#f44336"))
            except Exception as e:
                self.after(0, lambda: self._conn_label.configure(text=f"❌  {e}", text_color="#f44336"))

        threading.Thread(target=run, daemon=True).start()

    def _toggle_voice(self):
        if self._voice_running:
            self._stop_voice()
        else:
            self._start_voice()

    def _start_voice(self):
        import keyboard as kb

        self._save_settings()
        c = self.config_data

        if not c.get("gateway_url") or not c.get("gateway_token"):
            self._set_status("Set gateway URL and token first", "#f44336")
            return

        try:
            self._recorder = Recorder(device=c.get("input_device"))
            self._transcriber = Transcriber(model_size=c.get("whisper_model", "base"))
            self._gateway = GatewayClient(
                gateway_url=c["gateway_url"],
                token=c["gateway_token"],
                agent_id=c.get("agent_id", "main"),
                session_user=c.get("session_user", "voice-client"),
            )
            self._tts = TTSPipeline(
                voice=c.get("tts_voice", "en-US-AriaNeural"),
                rate=c.get("tts_rate", "+0%"),
                output_device=c.get("output_device"),
            )
        except Exception as e:
            self._set_status(f"Init error: {e}", "#f44336")
            return

        hotkey = c.get("hotkey", "F13")
        self._voice_running = True

        kb.on_press_key(hotkey, self._on_key_press)
        kb.on_release_key(hotkey, self._on_key_release)

        self._start_btn.configure(text="⏹  Stop", fg_color="#b71c1c", hover_color="#7f1212")
        self._set_status(f"Listening ({hotkey})", "#4caf50")

    def _stop_voice(self):
        import keyboard as kb
        self._voice_running = False
        kb.unhook_all()
        self._start_btn.configure(text="▶  Start", fg_color="#1f6aa5", hover_color="#144d7a")
        self._set_status("Stopped", "#888")
        self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888")

    def _on_key_press(self, e):
        if not self._is_recording and self._voice_running:
            self._is_recording = True
            self.after(0, lambda: self._mic_label.configure(text="🔴  Recording...", text_color="#f44336"))
            self._recorder.start()

    def _on_key_release(self, e):
        if self._is_recording:
            self._is_recording = False
            self.after(0, lambda: self._mic_label.configure(text="⏳  Processing...", text_color="#ff9800"))
            threading.Thread(target=self._process_recording, daemon=True).start()

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

        response_parts = []
        buffer = []
        for token in self._gateway.send(text):
            response_parts.append(token)
            buffer = self._tts.feed_token(token, buffer)
            self._append_response("".join(response_parts))

        self._tts.flush(buffer)
        self.after(0, lambda: self._mic_label.configure(text="🎙  Hold your hotkey to speak", text_color="#888"))

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
