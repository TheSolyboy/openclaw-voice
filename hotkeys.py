"""
hotkeys.py - Shared helpers for combo hotkeys and capture.
"""

from __future__ import annotations

import re
from typing import Callable

import keyboard


MODIFIER_ORDER = ["ctrl", "shift", "alt", "windows"]
DISPLAY_NAMES = {
    "ctrl": "Ctrl",
    "shift": "Shift",
    "alt": "Alt",
    "windows": "Win",
    "caps lock": "Caps Lock",
    "page up": "Page Up",
    "page down": "Page Down",
}
ALIASES = {
    "control": "ctrl",
    "left control": "ctrl",
    "right control": "ctrl",
    "left ctrl": "ctrl",
    "right ctrl": "ctrl",
    "left shift": "shift",
    "right shift": "shift",
    "left alt": "alt",
    "right alt": "alt",
    "alt gr": "alt",
    "left windows": "windows",
    "right windows": "windows",
    "win": "windows",
    "command": "windows",
    "cmd": "windows",
    "esc": "escape",
    "return": "enter",
    ")": "0",
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    "\"": "'",
    "<": ",",
    ">": ".",
    "?": "/",
}


def normalize_key_name(name: str | None) -> str:
    """Normalize keyboard names so saved hotkeys and live events match."""
    if not name:
        return ""
    key = re.sub(r"\s+", " ", name.strip().lower())
    return ALIASES.get(key, key)


def canonicalize_hotkey_keys(keys: list[str]) -> list[str]:
    """Deduplicate keys and put modifiers first in a stable order."""
    normalized = []
    for key in keys:
        normalized_key = normalize_key_name(key)
        if normalized_key and normalized_key not in normalized:
            normalized.append(normalized_key)

    modifiers = [key for key in MODIFIER_ORDER if key in normalized]
    others = [key for key in normalized if key not in MODIFIER_ORDER]
    return modifiers + others


def parse_hotkey(text: str) -> list[str]:
    parts = re.split(r"\s*\+\s*", text.strip()) if text.strip() else []
    return canonicalize_hotkey_keys(parts)


def format_hotkey(keys: list[str]) -> str:
    formatted = []
    for key in canonicalize_hotkey_keys(keys):
        display = DISPLAY_NAMES.get(key, key.upper() if key.startswith("f") else key.title() if len(key) > 1 else key)
        formatted.append(display)
    return "+".join(formatted)


def normalize_hotkey_text(text: str, default: str = "F13") -> str:
    keys = parse_hotkey(text)
    if not keys:
        return default
    return format_hotkey(keys)


class HoldHotkeyBinding:
    """Trigger press/release callbacks when a key combo becomes active/inactive."""

    def __init__(self, hotkey: str, on_press: Callable, on_release: Callable):
        self.hotkey_keys = set(parse_hotkey(hotkey))
        if not self.hotkey_keys:
            raise ValueError(f"Invalid hotkey: {hotkey!r}")
        self.on_press = on_press
        self.on_release = on_release
        self._pressed_keys: set[str] = set()
        self._active = False
        self._hook = None

    def start(self):
        if self._hook is None:
            self._hook = keyboard.hook(self._handle_event, suppress=False)

    def stop(self):
        if self._hook is not None:
            keyboard.unhook(self._hook)
            self._hook = None
        self._pressed_keys.clear()
        self._active = False

    def _handle_event(self, event):
        key_name = normalize_key_name(event.name)
        if not key_name:
            return

        if event.event_type == "down":
            self._pressed_keys.add(key_name)
        elif event.event_type == "up":
            self._pressed_keys.discard(key_name)

        is_active = self.hotkey_keys.issubset(self._pressed_keys)
        if is_active and not self._active:
            self._active = True
            self.on_press(event)
        elif self._active and not is_active:
            self._active = False
            self.on_release(event)
