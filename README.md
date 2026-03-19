# OpenClaw Voice Client

A push-to-talk voice interface for [OpenClaw](https://openclaw.ai). Hold a key → speak → get a streamed response read back to you in real-time.

## How it works

1. Hold your configured hotkey (default: `F13`)
2. Speak your prompt
3. Release the key — audio is transcribed locally with Whisper
4. Text is streamed to your OpenClaw gateway
5. Response streams back token-by-token, converted to speech sentence-by-sentence as it arrives

## Running

**With GUI (recommended):**
```bash
python gui.py
```

**Headless (no UI):**
```bash
python main.py
```

## Requirements

- Windows 10/11
- Python 3.10+
- A running [OpenClaw](https://openclaw.ai) gateway with the OpenAI-compatible endpoint enabled
- Microphone

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/openclaw-voice.git
cd openclaw-voice
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** If you have an NVIDIA GPU, faster-whisper can use CUDA for faster transcription. Install the CUDA version of PyTorch first. CPU works fine for the `base` model.

### 4. Configure

Copy the example config and fill in your details:

```bash
copy config.json.example config.json
```

Edit `config.json`:

```json
{
  "gateway_url": "https://your-openclaw-gateway.example.com",
  "gateway_token": "your-token-here",
  "agent_id": "main",
  "hotkey": "F13",
  "whisper_model": "base",
  "tts_voice": "en-US-AriaNeural",
  "tts_rate": "+0%",
  "session_user": "voice-client"
}
```

| Key | Description |
|-----|-------------|
| `gateway_url` | Your OpenClaw gateway URL |
| `gateway_token` | Gateway auth token (from OpenClaw config) |
| `agent_id` | Which agent to talk to (usually `main`) |
| `hotkey` | Key to hold while speaking (e.g. `F13`, `F24`, `caps lock`) |
| `whisper_model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `tts_voice` | Edge TTS voice — see [available voices](https://github.com/rany2/edge-tts#list-of-available-voices) |
| `tts_rate` | Speech speed: `+0%` normal, `+20%` faster, `-10%` slower |
| `session_user` | Stable session identifier (keeps conversation history) |

### 5. Enable the OpenClaw gateway endpoint

In your OpenClaw config, make sure the OpenAI-compatible endpoint is enabled:

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

### 6. Run

```bash
python main.py
```

## Hotkey tips

- `F13`–`F24` are great choices — most keyboards don't use them, so no conflicts
- `caps lock` works well if you never use it
- `right alt`, `right ctrl`, `scroll lock` are other good options
- Run as administrator if key detection doesn't work for your chosen key

## Available TTS voices (examples)

| Voice | Language | Style |
|-------|----------|-------|
| `en-US-AriaNeural` | English (US) | Warm, natural |
| `en-US-GuyNeural` | English (US) | Male |
| `en-GB-SoniaNeural` | English (UK) | British female |
| `en-AU-NatashaNeural` | English (AU) | Australian female |

Run `edge-tts --list-voices` to see all available voices.

## Troubleshooting

**Mic not detected:** Check your Windows default recording device in Sound settings.

**Gateway connection error:** Verify `gateway_url` and `gateway_token` in config.json.

**Key detection not working:** Try running as administrator (right-click → Run as administrator).

**Slow transcription:** Use `"whisper_model": "tiny"` for faster (but less accurate) transcription, or set up CUDA for GPU acceleration.
