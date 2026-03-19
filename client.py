"""
client.py - Streaming HTTP client for OpenClaw's OpenAI-compatible gateway
"""

import httpx
import json
from typing import Generator


class GatewayClient:
    def __init__(self, gateway_url: str, token: str, agent_id: str, session_user: str):
        self.url = gateway_url.rstrip("/") + "/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-openclaw-agent-id": agent_id,
        }
        self.session_user = session_user
        self.history = []  # Simple conversation history

    def send(self, text: str) -> Generator[str, None, None]:
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
            yield "[Error: Could not connect to gateway. Check your gateway_url in config.json]"
            return
        except httpx.HTTPStatusError as e:
            yield f"[Error: Gateway returned {e.response.status_code}]"
            return
        except Exception as e:
            yield f"[Error: {e}]"
            return

        # Add assistant response to history
        if full_response:
            self.history.append({"role": "assistant", "content": full_response})
