"""
client.py - Streaming HTTP client for OpenClaw's OpenAI-compatible gateway
"""

import httpx
import json
import threading
from typing import Generator
from urllib.parse import urlsplit, urlunsplit


CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def resolve_chat_completions_url(gateway_url: str) -> str:
    """Accept either a gateway base URL or the full chat completions endpoint."""
    raw_url = gateway_url.strip()
    parts = urlsplit(raw_url)
    path = parts.path.rstrip("/")
    cleaned = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    url = cleaned.rstrip("/")
    if url.endswith(CHAT_COMPLETIONS_PATH):
        return url
    return url + CHAT_COMPLETIONS_PATH


def describe_gateway_status(status_code: int) -> str:
    """Return a user-facing explanation for common gateway errors."""
    if status_code in (401, 403):
        return (
            f"gateway auth rejected (HTTP {status_code}). "
            "Use the gateway token, or enter the gateway password here if auth mode is password."
        )
    if status_code == 404:
        return (
            "gateway endpoint not found (HTTP 404). "
            "Check that the URL points at the OpenClaw gateway and that chatCompletions is enabled."
        )
    return f"gateway returned HTTP {status_code}"


class GatewayClient:
    def __init__(self, gateway_url: str, token: str, agent_id: str, session_user: str):
        self.url = resolve_chat_completions_url(gateway_url)
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-openclaw-agent-id": agent_id,
        }
        self.session_user = session_user
        self.history = []  # Simple conversation history

    def send(self, text: str, stop_event: threading.Event | None = None) -> Generator[str, None, None]:
        """Send a message and yield tokens as they stream in."""
        self.history.append({"role": "user", "content": text})

        payload = {
            "model": "openclaw",
            "messages": self.history,
            "stream": True,
            "user": self.session_user,
        }

        full_response = ""

        try:
            with httpx.stream(
                "POST",
                self.url,
                headers=self.headers,
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if stop_event is not None and stop_event.is_set():
                        response.close()
                        return
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Strip "data: "
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        token = delta.get("content", "")
                        if token:
                            full_response += token
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        except httpx.ConnectError:
            yield "[Error: Could not connect to gateway. Check the URL and that the gateway host is reachable]"
            return
        except httpx.HTTPStatusError as e:
            yield f"[Error: {describe_gateway_status(e.response.status_code)}]"
            return
        except Exception as e:
            yield f"[Error: {e}]"
            return

        # Add assistant response to history
        if full_response and not (stop_event is not None and stop_event.is_set()):
            self.history.append({"role": "assistant", "content": full_response})
