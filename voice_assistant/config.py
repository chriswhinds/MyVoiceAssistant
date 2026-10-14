# Copyright 2026 Christopher Hinds, Stratum LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""External JSON config, Ollama URL sanitizing and diagnostics tracing."""

import json
import re
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import tkinter as tk


class ConfigMixin:
    @staticmethod
    def _load_external_config(path):
        """Load optional JSON config; invalid/missing files fall back safely."""
        defaults = {
            "ollama": {
                "host": "127.0.0.1",
                "port": 11434,
                "model": "llama3.2",
                # Number of user/assistant exchanges sent back as context.
                "max_history_turns": 12,
                # How long Ollama keeps the model loaded between requests.
                "keep_alive": "30m",
            },
            "system_prompt": (
                "You are a helpful local voice assistant. "
                "Answer conversationally and keep spoken responses reasonably concise. "
                "Your replies are read aloud, so do not use markdown, bullet points, "
                "tables or code blocks."
            ),
            "standby": {
                "enabled": True,
                "interval_seconds": 10,
                "phrase": "I'm researching your request, please stand by.",
            },
            "speech": {
                "endpoint_silence_seconds": 5.0,
                "max_utterance_seconds": 120,
                "low_latency_mode": True,
                "endpoint_smoothing_alpha": 0.35,
                "endpoint_voice_multiplier": 1.30,
                "endpoint_voice_margin": 14.0,
                "endpoint_confirm_windows": 2,
                # Speak the reply sentence-by-sentence while Ollama streams.
                "stream_tts": True,
                # Delay mouth animation to match audio player start-up latency.
                "avatar_sync_offset_ms": 80,
                # Play inaudible silence so the output (notably HDMI) never
                # suspends; otherwise the first word is lost while it wakes.
                "keep_audio_output_awake": True,
                # Fallback when the keep-alive stream is unavailable: silence
                # prepended to speech that starts after the output went idle.
                "playback_lead_in_ms": 300,
            },
            "diagnostics": {
                "trace_enabled": False
            },
            "trigger_words": {
                "enabled": True,
                "auto_arm_on_launch": True,
                "start": ["hey assistant", "start listening"],
                "stop": ["stop listening", "go to sleep"],
                "start_acknowledgement": "Listening.",
                "stop_acknowledgement": "Standing by.",
                "strip_trigger_from_request": True,
                "active_trigger_scale": 0.72,
                "post_tts_guard_seconds": 0.25,
                "active_voice_floor": 45.0,
                "active_confirm_windows": 1,
            },
        }

        try:
            if not Path(path).exists():
                return defaults
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                return defaults

            # Shallow/nested merge for supported sections.
            merged = defaults
            for key in ("ollama", "standby", "speech", "trigger_words", "diagnostics"):
                if isinstance(loaded.get(key), dict):
                    merged[key].update(loaded[key])
            if isinstance(loaded.get("system_prompt"), str):
                merged["system_prompt"] = loaded["system_prompt"]
            return merged
        except Exception:
            return defaults

    @staticmethod
    def _sanitize_ollama_base_url(value):
        """
        Return canonical Ollama server root, e.g.
        http://192.168.1.165:11434
        """
        text = "" if value is None else str(value)

        # Remove both literal escaped and real CR/LF.
        text = text.replace("\\r", "").replace("\\n", "")
        text = text.replace("\r", "").replace("\n", "")
        text = re.sub(r"\s+", "", text).strip()

        if not text:
            return "http://127.0.0.1:11434"

        low = text.lower()

        # Exact malformed schemes only. Do NOT let "http:/" match "http://".
        if low.startswith("heep://"):
            text = "http://" + text[len("heep://"):]
        elif low.startswith("htpp://"):
            text = "http://" + text[len("htpp://"):]
        elif low.startswith("hhttp://"):
            text = "http://" + text[len("hhttp://"):]
        elif low.startswith("http:/") and not low.startswith("http://"):
            text = "http://" + text[len("http:/"):].lstrip("/")
        elif low.startswith("https:/") and not low.startswith("https://"):
            text = "https://" + text[len("https:/"):].lstrip("/")
        elif "://" in text and not re.match(r"^https?://", text, flags=re.I):
            # Unknown malformed scheme: preserve authority/path but force HTTP.
            text = "http://" + text.split("://", 1)[1].lstrip("/")
        elif not re.match(r"^https?://", text, flags=re.I):
            text = "http://" + text.lstrip("/")

        parsed = urlsplit(text)

        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid Ollama URL scheme after normalization: {scheme!r}"
            )

        if not parsed.netloc:
            raise ValueError(
                f"Invalid Ollama host URL: {value!r}. "
                "Expected http://HOST:PORT"
            )

        # Base URL must be server root. Strip any accidental API path.
        path = parsed.path.rstrip("/")
        lower_path = path.lower()

        for suffix in (
            "/api/chat",
            "/chat",
            "/api/tags",
            "/tags",
            "/api",
        ):
            if lower_path.endswith(suffix):
                path = path[:-len(suffix)]
                break

        # Ignore any remaining path entirely; application owns API paths.
        return urlunsplit((scheme, parsed.netloc, "", "", "")).rstrip("/")

    @staticmethod
    def _force_http_scheme(url):
        """
        Final request gate. Repairs malformed scheme one last time and
        requires a valid HTTP(S) URL before requests is called.
        """
        text = "" if url is None else str(url)
        text = text.replace("\\r", "").replace("\\n", "")
        text = text.replace("\r", "").replace("\n", "")
        text = re.sub(r"\s+", "", text).strip()

        low = text.lower()
        if low.startswith("heep://"):
            text = "http://" + text[len("heep://"):]
        elif low.startswith("htpp://"):
            text = "http://" + text[len("htpp://"):]
        elif low.startswith("hhttp://"):
            text = "http://" + text[len("hhttp://"):]
        elif low.startswith("http:/") and not low.startswith("http://"):
            text = "http://" + text[len("http:/"):].lstrip("/")
        elif low.startswith("https:/") and not low.startswith("https://"):
            text = "https://" + text[len("https:/"):].lstrip("/")
        elif "://" in text and not re.match(r"^https?://", text, flags=re.I):
            text = "http://" + text.split("://", 1)[1].lstrip("/")
        elif not re.match(r"^https?://", text, flags=re.I):
            text = "http://" + text.lstrip("/")

        parsed = urlsplit(text)

        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(
                f"Blocked malformed Ollama URL scheme: {parsed.scheme!r}"
            )
        if not parsed.netloc:
            raise ValueError(f"Blocked malformed Ollama URL host: {text!r}")

        return urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path,
            "",
            "",
        ))

    @staticmethod
    def _validate_ollama_api_url(url, endpoint):
        """Validate the exact final URL before requests sees it."""
        clean = str(url or "").replace("\\r", "").replace("\\n", "")
        clean = clean.replace("\r", "").replace("\n", "").strip()

        parsed = urlsplit(clean)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid Ollama URL scheme: {parsed.scheme!r}. "
                "Only http:// or https:// are allowed."
            )

        if not parsed.netloc:
            raise ValueError(f"Invalid Ollama URL host: {clean!r}")

        expected_path = f"/api/{str(endpoint).strip().lstrip('/')}"
        if parsed.path != expected_path:
            raise ValueError(
                f"Invalid Ollama API path {parsed.path!r}; "
                f"expected {expected_path!r}."
            )

        if parsed.query or parsed.fragment:
            raise ValueError("Ollama API URL must not contain query or fragment data.")

        return clean

    def _trace_enabled(self):
        """Return whether verbose terminal diagnostics are enabled."""
        try:
            diagnostics = self.app_config.get("diagnostics", {})
            return bool(diagnostics.get("trace_enabled", False))
        except Exception:
            return False

    def _trace_print(self, *args, **kwargs):
        """Print only when diagnostics.trace_enabled is true."""
        if not self._trace_enabled():
            return
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    def _trace_exception(self, scope, exc, **context):
        """Print a full terminal traceback plus diagnostic context."""
        if not self._trace_enabled():
            return
        try:
            print("\n" + "!" * 78, flush=True)
            print(f"[EXCEPTION TRACE] scope={scope}", flush=True)
            print(f"[EXCEPTION TRACE] type={type(exc).__name__}", flush=True)
            print(f"[EXCEPTION TRACE] message={exc!r}", flush=True)
            for key, value in context.items():
                try:
                    print(f"[EXCEPTION TRACE] {key}={value!r}", flush=True)
                except Exception:
                    print(f"[EXCEPTION TRACE] {key}=<unprintable>", flush=True)
            print("[EXCEPTION TRACE] traceback:", flush=True)
            print(
                "".join(
                    traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                ),
                flush=True,
            )
            print("!" * 78 + "\n", flush=True)
        except Exception as trace_exc:
            print(
                f"[EXCEPTION TRACE] tracing itself failed: {trace_exc!r}",
                flush=True,
            )

    def _safe_stt_threads(self):
        """Read the CPU threads spinbox without raising on empty/bad input."""
        try:
            return max(1, int(self.stt_threads_var.get()))
        except (tk.TclError, ValueError, TypeError):
            return 4

    def _refresh_runtime_settings(self):
        """UI thread only: mirror live Tk settings for worker threads."""
        try:
            settings = {
                "system_prompt": self.system_prompt.get("1.0", "end-1c").strip(),
                "auto_speak": bool(self.auto_speak_var.get()),
            }
            try:
                settings["ollama_url"] = self._sanitize_ollama_base_url(
                    self.ollama_url_var.get()
                )
            except ValueError:
                pass  # keep the last valid URL while the user is typing
        except Exception:
            return
        with self._runtime_settings_lock:
            self._runtime_settings.update(settings)

    def _get_runtime_settings(self):
        with self._runtime_settings_lock:
            return dict(self._runtime_settings)

    def _configured_ollama_url(self):
        ollama_cfg = self.app_config.get("ollama", {})
        host = str(ollama_cfg.get("host", "127.0.0.1"))
        try:
            port = int(str(ollama_cfg.get("port", 11434)).strip())
        except Exception:
            port = 11434

        clean_host = self._sanitize_ollama_base_url(host)

        # If the host already contains an explicit numeric port, keep it.
        # Otherwise append the configured port.
        authority = clean_host.split("://", 1)[-1]
        if re.search(r":\d+$", authority):
            return clean_host

        return f"{clean_host}:{port}"

    def reload_external_config(self):
        """Reload JSON config and update the visible Ollama/system-prompt fields."""
        self.app_config = self._load_external_config(self.config_path)
        ollama_cfg = self.app_config.get("ollama", {})

        self.ollama_url_var.set(self._sanitize_ollama_base_url(self._configured_ollama_url()))
        self.model_var.set(str(ollama_cfg.get("model", "llama3.2")))
        self._set_runtime_ollama_model(self.model_var.get())

        self.system_prompt.delete("1.0", "end")
        self.system_prompt.insert(
            "1.0",
            str(self.app_config.get("system_prompt", "")).strip()
        )

        trigger_cfg = dict(self.app_config.get("trigger_words", {}))
        self.voice_trigger_config = trigger_cfg
        self.voice_triggers_enabled = bool(trigger_cfg.get("enabled", True))
        if hasattr(self, "trigger_enabled_var"):
            self.trigger_enabled_var.set(self.voice_triggers_enabled)

        if self.voice_triggers_enabled and not self.is_listening:
            self.root.after(100, self.arm_trigger_service)
        elif not self.voice_triggers_enabled and self.is_listening:
            self.command_listening_event.clear()
            self.is_listening = False

        self.status_var.set(f"Reloaded config: {self.config_path}")
