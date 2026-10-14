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

import json
import queue
import threading
import traceback
import time
from urllib.parse import urlsplit, urlunsplit
import tkinter as tk

APP_BUILD = "v7.10.9"
from tkinter import ttk, filedialog, messagebox

import requests
import speech_recognition as sr
import pyaudio
import platform
import shutil
import subprocess
import random
import re
import math
import os
import tempfile
import wave
from array import array
from collections import deque
from pathlib import Path


class OllamaVoiceAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Voice Assistant v7.10.9 - Configurable Trace")
        self.root.geometry("1250x850")
        self.root.minsize(1000, 700)

        # External JSON configuration. By default the app looks beside this
        # Python file for ollama_voice_assistant_config.json. Override with:
        #   export OLLAMA_VOICE_CONFIG=/path/to/config.json
        self.config_path = Path(
            os.environ.get(
                "OLLAMA_VOICE_CONFIG",
                str(Path(__file__).resolve().parent / "ollama_voice_assistant_config.json")
            )
        ).expanduser()
        self.app_config = self._load_external_config(self.config_path)

        self.recognizer = sr.Recognizer()
        # Long-form dictation tuning:
        # - up to 120 seconds in a single captured utterance
        # - allow a few seconds of hesitation before considering speech complete
        self.recognizer.pause_threshold = 3.0
        self.recognizer.non_speaking_duration = 1.0
        self.microphone = None
        self._mic_display_to_index = {}
        self._listener_config = {}

        # Live Ollama request state. Unlike _listener_config, this must NOT be
        # frozen when the microphone listener starts because the user may
        # change the Ollama model while the listener remains active.
        self._ollama_runtime_lock = threading.Lock()
        self._ollama_runtime_model = str(
            self.app_config.get("ollama", {}).get("model", "llama3.2")
        ).strip()

        # Fully local speech-to-text state.
        # "Auto" prefers whisper.cpp/Vulkan when available, otherwise it
        # falls back to Faster-Whisper on the Mac Pro Xeon CPU.
        self.stt_backend_var = tk.StringVar(value="Auto")
        self.stt_model_var = tk.StringVar(value="base.en")
        self.stt_threads_var = tk.IntVar(value=max(2, min(8, (os.cpu_count() or 4) - 1)))
        self._faster_whisper_model = None
        self._faster_whisper_loaded_name = None
        self._stt_lock = threading.Lock()
        self._last_stt_backend = "Not initialized"
        self._whisper_cpp_cli = None
        self._whisper_cpp_model = None
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.shutdown_requested = False
        self.audio_busy = threading.Event()
        self.ui_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.messages = []

        # Tracks a single active Ollama wait/reminder cycle.
        self._response_wait_lock = threading.Lock()
        self._response_wait_event = None

        # Voice trigger state.
        trigger_cfg = dict(self.app_config.get("trigger_words", {}))
        self.voice_triggers_enabled = bool(trigger_cfg.get("enabled", True))
        self.voice_trigger_config = trigger_cfg
        self.command_listening_event = threading.Event()
        self.listener_thread = None

        # Explicit listener state machine.
        self.listener_mode_lock = threading.Lock()
        self.listener_mode = "STANDBY"

        # Avatar animation state.  Tkinter drawing must stay on the UI thread,
        # so the TTS worker only changes flags and the scheduled avatar loop
        # reads those flags.
        self.avatar_speaking = False
        self.avatar_phase = 0.0
        self.avatar_blink = 0
        self.avatar_next_blink = time.time() + random.uniform(2.5, 5.0)
        self.avatar_mouth_level = 0.0

        # Ubuntu/Linux audio state.  TTS is rendered to a WAV file with
        # espeak-ng and then played through PipeWire/PulseAudio/ALSA.
        self.avatar_envelope = []
        self.avatar_envelope_frame_ms = 50
        self.avatar_envelope_started = 0.0
        self.audio_player_name = None
        self.audio_player_cmd = None
        self.tts_synth = None

        # Neural female TTS. Piper is the preferred local voice engine.
        # espeak-ng remains available as a fallback.
        self.piper_voice_name = "en_US-hfc_female-medium"
        self.piper_voice = None
        self.piper_voice_path = None
        self.piper_available = False
        self._piper_lock = threading.Lock()

        self._detect_linux_audio_stack()
        self._detect_piper_voice()
        self._detect_local_stt_stack()

        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        self._build_ui()
        self._load_microphones()
        self._validate_ubuntu_runtime()
        self.root.after(80, self._process_ui_queue)
        self.root.after(60, self._animate_avatar)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if (
            self.voice_triggers_enabled
            and bool(self.voice_trigger_config.get("auto_arm_on_launch", True))
        ):
            self.root.after(900, self.arm_trigger_service)

    @staticmethod
    def _load_external_config(path):
        """Load optional JSON config; invalid/missing files fall back safely."""
        defaults = {
            "ollama": {
                "host": "127.0.0.1",
                "port": 11434,
                "model": "llama3.2",
            },
            "system_prompt": (
                "You are a helpful local voice assistant. "
                "Answer conversationally and keep spoken responses reasonably concise."
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
            },
            "diagnostics": {
                "trace_enabled": True
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
            for key in ("ollama", "standby", "speech", "trigger_words"):
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

    def _prewarm_stt(self, config):
        """Pre-load persistent Faster-Whisper before the user starts speaking."""
        try:
            requested = config.get("stt_backend", "Auto")
            low_latency = bool(
                self.app_config.get("speech", {}).get("low_latency_mode", True)
            )

            if requested == "Faster-Whisper CPU":
                self._load_faster_whisper_cpu(config)
            elif (
                requested == "Auto"
                and low_latency
                and self._python_module_available("faster_whisper")
            ):
                self._load_faster_whisper_cpu(config)
        except Exception as exc:
            self.ui_queue.put((
                "status",
                f"STT pre-warm warning: {exc}"
            ))

    def _validate_ubuntu_runtime(self):
        """Non-fatal startup checks for the Ubuntu 26.04 build."""
        problems = []

        if platform.system().lower() != "linux":
            problems.append("Operating system is not Linux.")

        machine = platform.machine().lower()
        if machine not in ("x86_64", "amd64"):
            problems.append(
                f"Architecture is {platform.machine()}, not the expected x86_64."
            )

        if not self.piper_available and not self.tts_synth:
            problems.append(
                "Neither Piper neural TTS nor espeak-ng fallback is available."
            )

        if not self.piper_available:
            problems.append(
                "Piper female voice is not ready; speech will fall back to espeak-ng. "
                "Run the updated installer to download en_US-hfc_female-medium."
            )

        if not self.audio_player_cmd:
            problems.append(
                "No pw-play, paplay, or aplay speaker backend was found."
            )

        # Do not require Vulkan: Faster-Whisper CPU is a supported local
        # fallback.  Warn only when neither local backend is available.
        faster_whisper_available = self._python_module_available("faster_whisper")
        vulkan_ready = bool(self._whisper_cpp_cli and self._whisper_cpp_model)

        if not faster_whisper_available and not vulkan_ready:
            problems.append(
                "No local STT backend is ready. Install faster-whisper and/or "
                "build whisper.cpp with Vulkan."
            )

        if problems:
            self.status_var.set("Ubuntu prerequisites need attention")
            self.root.after(
                500,
                lambda: messagebox.showwarning(
                    "Ubuntu Runtime Check",
                    "The application can start, but these components need attention:\\n\\n"
                    + "\\n".join(f"• {item}" for item in problems)
                    + "\\n\\nUse the included install_ubuntu_26_04.sh script."
                )
            )

    def _build_ui(self):
        settings = ttk.LabelFrame(self.root, text="Voice and Ollama Settings", padding=10)
        settings.pack(fill="x", padx=12, pady=(12, 8))

        ttk.Label(settings, text="Microphone:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(settings, textvariable=self.mic_var, state="readonly", width=52)
        self.mic_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=4)
        mic_button_frame = ttk.Frame(settings)
        mic_button_frame.grid(row=0, column=2, sticky="w", padx=(0, 12), pady=4)

        ttk.Button(
            mic_button_frame,
            text="Refresh",
            command=self._load_microphones
        ).pack(side="left")

        ttk.Button(
            mic_button_frame,
            text="Test Mic",
            command=self.test_microphone
        ).pack(side="left", padx=(6, 0))

        ttk.Label(settings, text="Local STT backend:").grid(row=0, column=3, sticky="w", padx=(0, 6), pady=4)
        self.stt_backend_combo = ttk.Combobox(
            settings,
            textvariable=self.stt_backend_var,
            values=("Auto", "Vulkan (whisper.cpp)", "Faster-Whisper CPU"),
            state="readonly",
            width=24,
        )
        self.stt_backend_combo.grid(row=0, column=4, sticky="ew", pady=4)

        ttk.Label(settings, text="STT model:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        self.stt_model_combo = ttk.Combobox(
            settings,
            textvariable=self.stt_model_var,
            values=("tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium"),
            width=20,
        )
        self.stt_model_combo.grid(row=1, column=1, sticky="w", padx=(0, 6), pady=4)

        ttk.Label(settings, text="CPU threads:").grid(row=1, column=2, sticky="e", padx=(0, 6), pady=4)
        ttk.Spinbox(settings, from_=1, to=max(2, os.cpu_count() or 16), textvariable=self.stt_threads_var, width=6).grid(
            row=1, column=3, sticky="w", padx=(0, 6), pady=4
        )

        ttk.Button(settings, text="STT Diagnostics", command=self.stt_diagnostics).grid(
            row=1, column=4, sticky="w", pady=4
        )

        ttk.Label(settings, text="Ollama URL:").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        self.ollama_url_var = tk.StringVar(value=self._sanitize_ollama_base_url(self._configured_ollama_url()))
        ttk.Entry(settings, textvariable=self.ollama_url_var).grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)

        config_button_frame = ttk.Frame(settings)
        config_button_frame.grid(row=2, column=2, padx=(0, 12), pady=4)
        ttk.Button(
            config_button_frame,
            text="Load Models",
            command=self.load_ollama_models
        ).pack(side="left")
        ttk.Button(
            config_button_frame,
            text="Reload Config",
            command=self.reload_external_config
        ).pack(side="left", padx=(6, 0))

        ttk.Label(settings, text="Model:").grid(row=2, column=3, sticky="w", padx=(0, 6), pady=4)
        self.model_var = tk.StringVar(value=str(self.app_config.get("ollama", {}).get("model", "llama3.2")))
        self.model_combo = ttk.Combobox(
            settings,
            textvariable=self.model_var,
            width=25,
        )
        self.model_combo.grid(row=2, column=4, sticky="ew", pady=4)
        self.model_combo.bind(
            "<<ComboboxSelected>>",
            self._on_ollama_model_selected,
        )
        self._set_runtime_ollama_model(self.model_var.get())

        ttk.Label(settings, text="System prompt:").grid(row=3, column=0, sticky="nw", padx=(0, 6), pady=4)
        self.system_prompt = tk.Text(settings, height=3, wrap="word")
        self.system_prompt.grid(row=3, column=1, columnspan=4, sticky="ew", pady=4)
        self.system_prompt.insert(
            "1.0",
            str(self.app_config.get("system_prompt", "")).strip()
        )

        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(4, weight=1)

        controls = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        controls.pack(fill="x")

        self.start_button = ttk.Button(controls, text="Start Listening", command=self.start_listening)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop Listening", command=self.stop_listening, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Test Ollama", command=self.test_ollama).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Test Speaker", command=self.test_speaker).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Audio Diagnostics", command=self.audio_diagnostics).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Speak Last Response", command=self.speak_last_response).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Clear Conversation", command=self.clear_conversation).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Save Conversation", command=self.save_conversation).pack(side="left")

        status_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        status_frame.pack(fill="x")
        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.status_var = tk.StringVar(value=f"Ready — {APP_BUILD}")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=(6, 18))
        self.auto_speak_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            status_frame,
            text="Automatically speak Ollama responses",
            variable=self.auto_speak_var
        ).pack(side="left")

        self.trigger_enabled_var = tk.BooleanVar(value=self.voice_triggers_enabled)
        ttk.Checkbutton(
            status_frame,
            text="Voice trigger words armed",
            variable=self.trigger_enabled_var,
            command=self._on_trigger_toggle
        ).pack(side="left", padx=(18, 0))

        panes = ttk.Panedwindow(self.root, orient="vertical")
        panes.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        top_pane = ttk.Panedwindow(panes, orient="horizontal")
        panes.add(top_pane, weight=2)

        avatar_frame = ttk.LabelFrame(top_pane, text="Female Assistant Avatar", padding=8)
        top_pane.add(avatar_frame, weight=1)

        self.avatar_canvas = tk.Canvas(
            avatar_frame,
            width=330,
            height=330,
            highlightthickness=0,
            background="#eef2f5"
        )
        self.avatar_canvas.pack(fill="both", expand=True)

        self.avatar_status_var = tk.StringVar(value="Idle")
        ttk.Label(
            avatar_frame,
            textvariable=self.avatar_status_var,
            anchor="center"
        ).pack(fill="x", pady=(6, 0))

        heard_frame = ttk.LabelFrame(top_pane, text="Speech Heard / Transcribed", padding=8)
        top_pane.add(heard_frame, weight=2)
        self.heard_text = tk.Text(heard_frame, wrap="word", font=("Arial", 12), padx=8, pady=8)
        heard_scroll = ttk.Scrollbar(heard_frame, orient="vertical", command=self.heard_text.yview)
        self.heard_text.configure(yscrollcommand=heard_scroll.set)
        self.heard_text.pack(side="left", fill="both", expand=True)
        heard_scroll.pack(side="right", fill="y")

        response_frame = ttk.LabelFrame(panes, text="Streaming Ollama Response", padding=8)
        panes.add(response_frame, weight=2)
        self.response_text = tk.Text(response_frame, wrap="word", font=("Arial", 12), padx=8, pady=8)
        response_scroll = ttk.Scrollbar(response_frame, orient="vertical", command=self.response_text.yview)
        self.response_text.configure(yscrollcommand=response_scroll.set)
        self.response_text.pack(side="left", fill="both", expand=True)
        response_scroll.pack(side="right", fill="y")

        self._draw_avatar(0.0, False, 0.0)

        footer = ttk.Label(self.root, text="Fully local pipeline: raw PyAudio VAD + 2-minute microphone capture -> whisper.cpp Vulkan or Faster-Whisper CPU -> streaming Ollama -> Piper neural female TTS -> PipeWire/PulseAudio/ALSA -> speakers + amplitude-driven female avatar.", padding=(12, 0, 12, 10), wraplength=1100)
        footer.pack(fill="x")

    def _load_microphones(self):
        """Enumerate only real PortAudio input-capable devices."""
        self._mic_display_to_index = {}
        pa = None
        try:
            pa = pyaudio.PyAudio()
            default_index = None
            try:
                default_index = pa.get_default_input_device_info().get("index")
            except Exception:
                pass

            display = []
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                except Exception:
                    continue
                if int(info.get("maxInputChannels", 0) or 0) <= 0:
                    continue
                name = str(info.get("name", f"Input device {i}"))
                rate = int(float(info.get("defaultSampleRate", 0) or 0))
                label = f"{i}: {name} [{rate} Hz]"
                display.append(label)
                self._mic_display_to_index[label] = i

            self.mic_combo["values"] = display
            if not display:
                self.mic_combo["values"] = ["No input-capable microphones detected"]
                self.mic_combo.current(0)
                return

            selected_label = None
            if default_index is not None:
                for label, idx in self._mic_display_to_index.items():
                    if idx == int(default_index):
                        selected_label = label
                        break

            if selected_label is None:
                for label in display:
                    lower = label.lower()
                    if "monitor" in lower or "hdmi" in lower:
                        continue
                    if any(k in lower for k in ("usb", "mic", "microphone", "input", "pulse", "pipewire", "default")):
                        selected_label = label
                        break

            self.mic_var.set(selected_label or display[0])
        except Exception as exc:
            messagebox.showerror(
                "Microphone Error",
                f"Unable to enumerate real PortAudio input devices.\n\n{exc}\n\n"
                "Check that PyAudio and PortAudio are installed and that your user can access audio devices."
            )
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def _selected_microphone_index(self):
        value = self.mic_var.get()
        if not value or value.startswith("No input-capable"):
            return None
        if value in self._mic_display_to_index:
            return self._mic_display_to_index[value]
        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    def test_microphone(self):
        """Capture a short sample and report measured microphone energy."""
        mic_index = self._selected_microphone_index()
        if mic_index is None:
            messagebox.showwarning("Microphone", "Select a valid input microphone first.")
            return

        self.status_var.set("Testing microphone for 3 seconds...")
        threading.Thread(target=self._test_microphone_worker, args=(mic_index,), daemon=True).start()

    def _test_microphone_worker(self, mic_index):
        try:
            mic = sr.Microphone(device_index=mic_index)
            local_rec = sr.Recognizer()
            local_rec.dynamic_energy_threshold = True
            with mic as source:
                self.ui_queue.put(("status", "Mic test: calibrating..."))
                local_rec.adjust_for_ambient_noise(source, duration=0.5)
                threshold = local_rec.energy_threshold
                self.ui_queue.put(("status", "Mic test: speak now..."))
                audio = local_rec.record(source, duration=3.0)

            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            samples = array("h")
            samples.frombytes(raw)
            if not samples:
                rms = 0.0
            else:
                total = sum(float(s) * float(s) for s in samples)
                rms = math.sqrt(total / len(samples))

            if rms < 80:
                verdict = "Very little/no microphone signal detected. Check the selected input device and Ubuntu sound input settings."
            elif rms < 250:
                verdict = "Microphone signal is low but present. Increase input gain if recognition is unreliable."
            else:
                verdict = "Microphone signal is present and should be usable."

            self.ui_queue.put(("info_mic", f"Device index: {mic_index}\nAmbient threshold: {threshold:.0f}\nMeasured RMS: {rms:.0f}\n\n{verdict}"))
            self.ui_queue.put(("status", "Ready"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Microphone test failed:\n{exc}"))

    def _set_runtime_ollama_model(self, model_name):
        model_name = str(model_name or "").strip()
        if not model_name:
            return
        with self._ollama_runtime_lock:
            self._ollama_runtime_model = model_name
        self._trace_print(
            f"[OLLAMA MODEL] runtime selection updated -> {model_name!r}"
        )

    def _get_runtime_ollama_model(self):
        with self._ollama_runtime_lock:
            return str(self._ollama_runtime_model or "").strip()

    def _on_ollama_model_selected(self, event=None):
        # This callback executes on the Tk/UI thread, so reading model_var here
        # is safe. Worker threads use _get_runtime_ollama_model() instead of
        # touching Tk variables directly.
        selected = self.model_var.get().strip()
        if selected:
            self._set_runtime_ollama_model(selected)
            self.status_var.set(
                f"{APP_BUILD} — selected Ollama model: {selected}"
            )

    def _snapshot_listener_config(self, mic_index):
        return {
            "mic_index": mic_index,
            "stt_backend": self.stt_backend_var.get().strip() or "Auto",
            "stt_model": self.stt_model_var.get().strip() or "base.en",
            "stt_threads": max(1, int(self.stt_threads_var.get() or 4)),
            "ollama_url": self._sanitize_ollama_base_url(self.ollama_url_var.get()),
            "ollama_model": self.model_var.get().strip(),
            "system_prompt": self.system_prompt.get("1.0", "end-1c").strip(),
            "auto_speak": bool(self.auto_speak_var.get()),
        }

    @staticmethod
    def _normalize_trigger_text(text):
        words = re.findall(r"[a-z0-9']+", (text or "").lower())
        return " ".join(words).strip()

    def _match_trigger_phrase(self, text, phrases):
        normalized = self._normalize_trigger_text(text)
        if not normalized:
            return None
        padded = f" {normalized} "
        for phrase in phrases or []:
            p = self._normalize_trigger_text(str(phrase))
            if p and f" {p} " in padded:
                return p
        return None

    def _remove_trigger_phrase(self, text, matched_phrase):
        if not text or not matched_phrase:
            return (text or "").strip()
        words = matched_phrase.split()
        pattern = r"\b" + r"[\s,;:!?.-]+".join(
            re.escape(word) for word in words
        ) + r"\b"
        cleaned = re.sub(pattern, " ", text, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:!?.-")
        return cleaned.strip()

    def _trigger_standby_status(self):
        starts = self.voice_trigger_config.get("start", ["hey assistant"])
        phrase = starts[0] if starts else "hey assistant"
        return f'Trigger standby — say "{phrase}"'

    def _set_command_mode(self, active, speak_ack=False):
        with self.listener_mode_lock:
            self.listener_mode = "ACTIVE" if active else "STANDBY"

        if active:
            self.command_listening_event.set()
            self.ui_queue.put(("trigger_state", True))
            if speak_ack:
                phrase = str(
                    self.voice_trigger_config.get(
                        "start_acknowledgement", "Listening."
                    )
                ).strip()
                if phrase:
                    self._queue_tts(phrase, label="Listening")
        else:
            self.command_listening_event.clear()
            self.ui_queue.put(("trigger_state", False))
            if speak_ack:
                phrase = str(
                    self.voice_trigger_config.get(
                        "stop_acknowledgement", "Standing by."
                    )
                ).strip()
                if phrase:
                    self._queue_tts(phrase, label="Trigger standby")

    def _validate_listener_microphone(self, mic_index):
        pa = None
        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(mic_index)
            if int(info.get("maxInputChannels", 0) or 0) < 1:
                raise RuntimeError("Selected PortAudio device has no input channels.")
            return True
        except Exception as exc:
            messagebox.showerror("Microphone Error", str(exc))
            return False
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def _ensure_listener_service(self, command_active):
        mic_index = self._selected_microphone_index()
        if mic_index is None:
            messagebox.showwarning(
                "Microphone",
                "Please select a valid input-capable microphone."
            )
            return

        if self.is_listening:
            self._set_command_mode(command_active, speak_ack=False)
            return

        if not self._validate_listener_microphone(mic_index):
            return

        self._listener_config = self._snapshot_listener_config(mic_index)

        if self.tts_queue.empty() and not self.is_speaking:
            self.audio_busy.clear()

        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 3.0
        self.recognizer.non_speaking_duration = 1.0

        self.is_listening = True
        if command_active:
            self.command_listening_event.set()
            with self.listener_mode_lock:
                self.listener_mode = "ACTIVE"
        else:
            self.command_listening_event.clear()
            with self.listener_mode_lock:
                self.listener_mode = "STANDBY"

        self.mic_combo.config(state="disabled")
        self.listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.listener_thread.start()
        self.ui_queue.put(("trigger_state", command_active))

    def arm_trigger_service(self):
        if not self.voice_triggers_enabled or self.shutdown_requested:
            return
        self._ensure_listener_service(command_active=False)

    def start_listening(self):
        self._ensure_listener_service(command_active=True)

    def stop_listening(self):
        if self.voice_triggers_enabled:
            self._set_command_mode(False, speak_ack=False)
        else:
            self.command_listening_event.clear()
            self.is_listening = False
            self.avatar_status_var.set("Idle")
            self.status_var.set("Stopping listener...")

    def _on_trigger_toggle(self):
        enabled = bool(self.trigger_enabled_var.get())
        self.voice_triggers_enabled = enabled
        self.voice_trigger_config["enabled"] = enabled

        if enabled:
            if not self.is_listening:
                self.arm_trigger_service()
            elif not self.command_listening_event.is_set():
                self.ui_queue.put(("trigger_state", False))
        else:
            self.command_listening_event.clear()
            self.is_listening = False
            self.status_var.set("Voice triggers disabled — microphone stopping.")

    @staticmethod
    def _pcm16_rms(raw_bytes):
        """Return RMS amplitude for little-endian signed 16-bit PCM."""
        if not raw_bytes:
            return 0.0
        samples = array("h")
        samples.frombytes(raw_bytes)
        if not samples:
            return 0.0
        total = 0.0
        for sample in samples:
            value = float(sample)
            total += value * value
        return math.sqrt(total / len(samples))

    @staticmethod
    def _pcm16_rms(raw_bytes):
        if not raw_bytes:
            return 0.0
        samples = array("h")
        samples.frombytes(raw_bytes)
        if not samples:
            return 0.0
        total = 0.0
        for sample in samples:
            value = float(sample)
            total += value * value
        return math.sqrt(total / len(samples))

    @staticmethod
    def _noise_floor_from_levels(levels):
        values = sorted(float(v) for v in levels if v >= 0.0)
        if not values:
            return 35.0
        count = max(1, int(len(values) * 0.60))
        quiet = values[:count]
        return max(20.0, quiet[len(quiet) // 2])

    @staticmethod
    def _speech_thresholds(noise_floor):
        start_level = max(70.0, min(900.0, noise_floor * 1.65 + 30.0))
        end_level = max(45.0, min(start_level * 0.60, noise_floor * 1.20 + 18.0))
        return start_level, end_level

    def _listen_loop(self):
        """
        Uses the same sr.Microphone + Recognizer.record capture mechanism
        as Test Mic. The source remains open for the lifetime of the listener.
        """
        config = dict(self._listener_config)
        mic_index = int(config.get("mic_index"))

        sample_window = 0.25
        pre_roll_windows = 4
        speech_confirm_windows = 1

        speech_cfg = self.app_config.get("speech", {})
        # Once speech begins, the utterance remains open until there are
        # endpoint_silence continuous seconds with no detected voice.
        endpoint_silence = max(
            1.0,
            min(10.0, float(speech_cfg.get("endpoint_silence_seconds", 5.0)))
        )
        max_utterance = max(
            10.0,
            min(120.0, float(speech_cfg.get("max_utterance_seconds", 120)))
        )

        # Endpointing is timestamp-based rather than "N silent windows".
        # This is much more stable when there are occasional noise spikes.
        endpoint_smoothing_alpha = max(
            0.05,
            min(
                0.95,
                float(speech_cfg.get("endpoint_smoothing_alpha", 0.35))
            ),
        )
        endpoint_voice_multiplier = max(
            1.05,
            min(
                3.0,
                float(speech_cfg.get("endpoint_voice_multiplier", 1.30))
            ),
        )
        endpoint_voice_margin = max(
            2.0,
            min(
                100.0,
                float(speech_cfg.get("endpoint_voice_margin", 14.0))
            ),
        )
        endpoint_confirm_windows = max(
            1,
            min(
                6,
                int(speech_cfg.get("endpoint_confirm_windows", 2))
            ),
        )

        max_windows = int(max_utterance / sample_window)

        microphone = sr.Microphone(device_index=mic_index)
        local_rec = sr.Recognizer()
        local_rec.dynamic_energy_threshold = False

        # Load the persistent STT model before active listening so the first
        # spoken request does not pay model initialization cost afterwards.
        self.ui_queue.put(("status", "Pre-warming local speech model..."))
        self._prewarm_stt(config)

        try:
            with microphone as source:
                self.ui_queue.put((
                    "status",
                    f"Calibrating microphone {mic_index}; remain quiet for 1 second..."
                ))

                calibration_levels = []
                for _ in range(4):
                    if not self.is_listening or self.shutdown_requested:
                        return
                    sample = local_rec.record(source, duration=sample_window)
                    raw = sample.get_raw_data(convert_width=2)
                    calibration_levels.append(self._pcm16_rms(raw))

                noise_floor = self._noise_floor_from_levels(calibration_levels)
                pre_roll = deque(maxlen=pre_roll_windows)
                candidate_count = 0
                was_busy = False
                cooldown_until = 0.0
                last_status = 0.0

                while self.is_listening and not self.shutdown_requested:
                    # Only suppress recognition while speaker audio is actually
                    # playing. audio_busy can remain set for queued TTS and must
                    # not make the microphone listener deaf.
                    if self.is_speaking:
                        was_busy = True
                        pre_roll.clear()
                        candidate_count = 0
                        local_rec.record(source, duration=sample_window)
                        continue

                    now = time.monotonic()

                    if was_busy:
                        was_busy = False
                        guard = max(
                            0.15,
                            min(
                                1.5,
                                float(
                                    self.voice_trigger_config.get(
                                        "post_tts_guard_seconds", 0.35
                                    )
                                ),
                            ),
                        )
                        cooldown_until = now + guard
                        pre_roll.clear()
                        candidate_count = 0

                    chunk_audio = local_rec.record(source, duration=sample_window)
                    raw = chunk_audio.get_raw_data(convert_width=2)
                    level = self._pcm16_rms(raw)

                    if now < cooldown_until:
                        continue

                    trigger, release = self._speech_thresholds(noise_floor)

                    with self.listener_mode_lock:
                        current_mode = self.listener_mode

                    if current_mode == "ACTIVE":
                        # Once awake, use a much more permissive ambient-relative
                        # detector. Wake-word standby remains conservative.
                        configured_floor = float(
                            self.voice_trigger_config.get(
                                "active_voice_floor", 30.0
                            )
                        )
                        active_floor = max(
                            20.0, min(120.0, configured_floor)
                        )
                        active_noise_margin = float(
                            self.voice_trigger_config.get(
                                "active_noise_margin", 8.0
                            )
                        )
                        active_noise_multiplier = float(
                            self.voice_trigger_config.get(
                                "active_noise_multiplier", 1.10
                            )
                        )
                        trigger = max(
                            active_floor,
                            noise_floor * active_noise_multiplier
                            + active_noise_margin,
                        )
                        release = max(
                            20.0,
                            noise_floor * 1.04 + 4.0,
                        )

                    if level < trigger * 0.65:
                        noise_floor = noise_floor * 0.97 + max(15.0, level) * 0.03

                    pre_roll.append(raw)

                    if current_mode == "ACTIVE":
                        active_detect = max(
                            20.0,
                            noise_floor * 1.06 + 5.0,
                        )
                        if level >= trigger or level >= active_detect:
                            candidate_count = 1
                        else:
                            candidate_count = 0
                    else:
                        if level >= trigger:
                            candidate_count += 1
                        elif level >= max(55.0, noise_floor * 1.35 + 18.0):
                            candidate_count = max(1, candidate_count)
                        else:
                            candidate_count = 0

                    if now - last_status >= 0.25:
                        if current_mode == "ACTIVE":
                            status_text = (
                                f"ACTIVE mic {mic_index} — level {level:.0f} | "
                                f"ambient {noise_floor:.0f} | trigger {trigger:.0f}"
                            )
                            self._trace_print(
                                f"[LISTENER TRACE] mode=ACTIVE mic={mic_index} "
                                f"level={level:.1f} ambient={noise_floor:.1f} "
                                f"trigger={trigger:.1f} release={release:.1f} "
                                f"speaking={self.is_speaking} "
                                f"processing={self.is_processing}"
                            )
                        else:
                            starts = self.voice_trigger_config.get(
                                "start", ["hey assistant"]
                            )
                            wake = starts[0] if starts else "hey assistant"
                            status_text = (
                                f'Trigger standby — say "{wake}" | '
                                f"mic level {level:.0f}"
                            )
                        self.ui_queue.put(("status", status_text))
                        last_status = now

                    required_confirm = (
                        max(
                            1,
                            int(
                                self.voice_trigger_config.get(
                                    "active_confirm_windows", 1
                                )
                            ),
                        )
                        if current_mode == "ACTIVE"
                        else speech_confirm_windows
                    )

                    if candidate_count < required_confirm:
                        continue

                    if current_mode == "ACTIVE":
                        detect_status = (
                            f"Speech detected — recording "
                            f"(level {level:.0f}, trigger {trigger:.0f})..."
                        )
                    else:
                        detect_status = "Possible trigger phrase detected..."
                    self.ui_queue.put(("status", detect_status))
                    self._trace_print(
                        f"[LISTENER TRACE] speech candidate accepted "
                        f"mode={current_mode} level={level:.1f} "
                        f"trigger={trigger:.1f}"
                    )

                    frames = list(pre_roll)
                    peak_rms = level
                    windows_recorded = len(frames)

                    # Stable endpoint detector state.
                    # last_voice_time is only refreshed by CONFIRMED voice
                    # activity, not by isolated noise spikes.
                    last_voice_time = time.monotonic()
                    smoothed_level = float(level)
                    endpoint_voice_hits = endpoint_confirm_windows
                    endpoint_silent_for = 0.0
                    endpoint_silence_frames = 0

                    while (
                        self.is_listening
                        and not self.shutdown_requested
                        and windows_recorded < max_windows
                    ):
                        if self.is_speaking:
                            frames = []
                            break

                        part = local_rec.record(source, duration=sample_window)
                        part_raw = part.get_raw_data(convert_width=2)
                        part_level = self._pcm16_rms(part_raw)

                        frames.append(part_raw)
                        windows_recorded += 1
                        peak_rms = max(peak_rms, part_level)

                        # Smooth RMS so one noisy frame cannot dominate endpointing.
                        smoothed_level = (
                            endpoint_smoothing_alpha * float(part_level)
                            + (1.0 - endpoint_smoothing_alpha) * smoothed_level
                        )

                        # Separate endpoint voice threshold from the start trigger.
                        # It tracks the calibrated ambient floor, but stays above
                        # the lower release threshold used by the VAD.
                        endpoint_voice_threshold = max(
                            release,
                            noise_floor * endpoint_voice_multiplier
                            + endpoint_voice_margin,
                            35.0,
                        )

                        # Confirm continuing voice across multiple windows. A single
                        # transient spike is intentionally NOT enough to reset the
                        # five-second silence clock.
                        strong_voice = (
                            part_level >= trigger
                            or smoothed_level >= endpoint_voice_threshold
                        )

                        if strong_voice:
                            endpoint_voice_hits += 1
                        else:
                            endpoint_voice_hits = 0

                        now_capture = time.monotonic()

                        if endpoint_voice_hits >= endpoint_confirm_windows:
                            last_voice_time = now_capture
                            endpoint_silent_for = 0.0
                            endpoint_silence_frames = 0
                        else:
                            endpoint_silent_for = max(
                                0.0,
                                now_capture - last_voice_time
                            )
                            endpoint_silence_frames += 1

                        elapsed = windows_recorded * sample_window

                        if windows_recorded % 2 == 0:
                            if self.command_listening_event.is_set():
                                if endpoint_silent_for > 0.0:
                                    remaining = max(
                                        0.0,
                                        endpoint_silence - endpoint_silent_for
                                    )
                                    status_text = (
                                        f"Recording voice: {elapsed:.1f}s — "
                                        f"silence {endpoint_silent_for:.1f}s / "
                                        f"{endpoint_silence:.1f}s "
                                        f"({remaining:.1f}s remaining) | "
                                        f"level {part_level:.0f} / "
                                        f"hold {endpoint_voice_threshold:.0f}"
                                    )
                                else:
                                    status_text = (
                                        f"Recording voice: {elapsed:.1f}s — "
                                        f"level {part_level:.0f} | "
                                        f"hold {endpoint_voice_threshold:.0f}"
                                    )
                                self.ui_queue.put(("status", status_text))

                        if endpoint_silent_for >= endpoint_silence:
                            self.ui_queue.put((
                                "status",
                                f"{endpoint_silence:.1f} seconds since last "
                                "confirmed voice — finalizing captured audio..."
                            ))
                            break

                    pre_roll.clear()
                    candidate_count = 0

                    if not frames:
                        continue

                    if endpoint_silent_for > 0.75 and len(frames) > 4:
                        # Remove most endpoint silence before Whisper, retaining
                        # ~0.5 second. This is only output trimming and does not
                        # influence when the endpoint is declared.
                        silent_windows_estimate = int(
                            round(endpoint_silent_for / sample_window)
                        )
                        keep_silence_windows = max(
                            1,
                            int(round(0.5 / sample_window))
                        )
                        trim = max(
                            0,
                            silent_windows_estimate - keep_silence_windows
                        )
                        if 0 < trim < len(frames):
                            frames = frames[:-trim]

                    raw_pcm = b"".join(frames)
                    if not raw_pcm:
                        continue

                    audio = sr.AudioData(
                        raw_pcm,
                        source.SAMPLE_RATE,
                        source.SAMPLE_WIDTH,
                    )

                    self.ui_queue.put((
                        "status",
                        f"Voice captured (peak RMS {peak_rms:.0f}). "
                        "Transcribing locally..."
                    ))

                    text, backend_used = self._transcribe_local_audio(audio, config)
                    text = (text or "").strip()

                    if not text:
                        self.ui_queue.put((
                            "status",
                            "Audio was captured, but Whisper returned no text. "
                            "Listening again..."
                        ))
                        continue

                    self._last_stt_backend = backend_used
                    trigger_cfg = dict(self.voice_trigger_config)
                    start_phrases = trigger_cfg.get(
                        "start", ["hey assistant", "start listening"]
                    )
                    stop_phrases = trigger_cfg.get(
                        "stop", ["stop listening", "go to sleep"]
                    )

                    start_match = self._match_trigger_phrase(
                        text, start_phrases
                    )
                    stop_match = self._match_trigger_phrase(
                        text, stop_phrases
                    )

                    with self.listener_mode_lock:
                        decision_mode = self.listener_mode

                    # STOP ALWAYS HAS PRIORITY while ACTIVE.
                    if stop_match and decision_mode == "ACTIVE":
                        self._set_command_mode(False, speak_ack=True)
                        self.ui_queue.put((
                            "heard",
                            f"[trigger] Stop phrase detected: {text}\n"
                        ))
                        continue

                    if decision_mode != "ACTIVE":
                        if not start_match:
                            self.ui_queue.put((
                                "status",
                                self._trigger_standby_status()
                            ))
                            continue

                        self._set_command_mode(True, speak_ack=True)
                        print(
                            f"[LISTENER TRACE] START trigger accepted; "
                            f"listener_mode=ACTIVE text={text!r}",
                            flush=True,
                        )
                        request_text = text

                        if bool(
                            trigger_cfg.get(
                                "strip_trigger_from_request", True
                            )
                        ):
                            request_text = self._remove_trigger_phrase(
                                text, start_match
                            )

                        if not request_text:
                            continue

                        text = request_text

                    else:
                        # While an Ollama request is in flight, keep listening
                        # for the stop trigger but ignore additional normal prompts.
                        if self.is_processing:
                            self.ui_queue.put((
                                "status",
                                "Ollama is processing — listening only for stop trigger..."
                            ))
                            continue

                        if start_match and bool(
                            trigger_cfg.get(
                                "strip_trigger_from_request", True
                            )
                        ):
                            text = self._remove_trigger_phrase(
                                text, start_match
                            )
                            if not text:
                                continue

                    stamp = time.strftime("%H:%M:%S")
                    self.ui_queue.put((
                        "heard",
                        f"[{stamp}] You ({backend_used}): {text}\n"
                    ))

                    self.is_processing = True
                    request_config = dict(config)

                    # The listener's config is intentionally long-lived for
                    # microphone/STT settings. Ollama model selection is live.
                    live_model = self._get_runtime_ollama_model()
                    if live_model:
                        request_config["ollama_model"] = live_model

                    self._trace_print(
                        f"[OLLAMA MODEL] request will use -> "
                        f"{request_config.get('ollama_model')!r}"
                    )

                    threading.Thread(
                        target=self._stream_ollama_response,
                        args=(text, request_config),
                        daemon=True,
                    ).start()

        except Exception as exc:
            self.ui_queue.put(("error", f"Microphone/listening error:\n{exc}"))
        finally:
            self.ui_queue.put(("listener_stopped", None))

    def _python_module_available(self, module_name):
        try:
            import importlib.util
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _detect_local_stt_stack(self, config=None):
        """
        Locate whisper.cpp Vulkan CLI/model and Faster-Whisper availability.

        The application does not claim that Faster-Whisper uses Vulkan:
        CTranslate2's standard GPU backend is CUDA/NVIDIA. AMD/Vulkan local
        acceleration is provided through whisper.cpp; Faster-Whisper is the
        CPU fallback behind the same transcription interface.
        """
        candidates = []

        env_root = os.environ.get("WHISPER_CPP_ROOT")
        if env_root:
            candidates.append(Path(env_root).expanduser())

        candidates.extend([
            Path.home() / ".local" / "share" / "ollama-voice-assistant" / "whisper.cpp",
            Path.cwd() / "third_party" / "whisper.cpp",
            Path(__file__).resolve().parent / "third_party" / "whisper.cpp",
        ])

        cli_candidates = []
        if config is not None:
            model_name = config.get("stt_model", "base.en")
        else:
            model_name = self.stt_model_var.get().strip() if hasattr(self, "stt_model_var") else "base.en"

        for root in candidates:
            cli_candidates.extend([
                root / "build" / "bin" / "whisper-cli",
                root / "build-vulkan" / "bin" / "whisper-cli",
                root / "bin" / "whisper-cli",
            ])

        system_cli = shutil.which("whisper-cli")
        if system_cli:
            cli_candidates.insert(0, Path(system_cli))

        self._whisper_cpp_cli = next(
            (str(p) for p in cli_candidates if p and Path(p).is_file() and os.access(p, os.X_OK)),
            None,
        )

        model_candidates = []
        env_model = os.environ.get("WHISPER_CPP_MODEL")
        if env_model:
            model_candidates.append(Path(env_model).expanduser())

        for root in candidates:
            model_candidates.extend([
                root / "models" / f"ggml-{model_name}.bin",
                root / "models" / "ggml-base.en.bin",
            ])

        self._whisper_cpp_model = next(
            (str(p) for p in model_candidates if p and Path(p).is_file()),
            None,
        )

    def _vulkan_summary(self):
        vulkaninfo = shutil.which("vulkaninfo")
        if not vulkaninfo:
            return False, "vulkaninfo not installed"

        try:
            proc = subprocess.run(
                [vulkaninfo, "--summary"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=12,
            )
            text = proc.stdout or ""
            ok = proc.returncode == 0 and (
                "GPU" in text or "deviceName" in text or "Vulkan Instance Version" in text
            )
            return ok, text.strip()
        except Exception as exc:
            return False, str(exc)

    def _selected_stt_backend(self, config=None):
        if config is not None:
            requested = config.get("stt_backend", "Auto")
        else:
            requested = self.stt_backend_var.get().strip() or "Auto"
        self._detect_local_stt_stack(config)

        if requested == "Vulkan (whisper.cpp)":
            if not self._whisper_cpp_cli:
                raise RuntimeError(
                    "whisper.cpp Vulkan CLI was not found. Run build_whisper_vulkan.sh."
                )
            if not self._whisper_cpp_model:
                raise RuntimeError(
                    "A whisper.cpp GGML model was not found. Run build_whisper_vulkan.sh."
                )
            ok, detail = self._vulkan_summary()
            if not ok:
                raise RuntimeError(
                    "Vulkan is not usable on this Ubuntu installation.\n\n" + detail[:2000]
                )
            return "vulkan"

        if requested == "Faster-Whisper CPU":
            if not self._python_module_available("faster_whisper"):
                raise RuntimeError(
                    "faster-whisper is not installed in this Python environment."
                )
            return "faster-whisper-cpu"

        # Auto low-latency mode prefers the persistent Faster-Whisper model.
        # Explicit "Vulkan (whisper.cpp)" still forces AMD/Vulkan.
        low_latency = bool(
            self.app_config.get("speech", {}).get("low_latency_mode", True)
        )

        if low_latency and self._python_module_available("faster_whisper"):
            return "faster-whisper-cpu"

        if self._whisper_cpp_cli and self._whisper_cpp_model:
            ok, _ = self._vulkan_summary()
            if ok:
                return "vulkan"

        if self._python_module_available("faster_whisper"):
            return "faster-whisper-cpu"

        raise RuntimeError(
            "No fully local STT backend is available. Run install_ubuntu_26_04_local_stt.sh."
        )

    def _audio_data_to_wav(self, audio):
        """
        Convert SpeechRecognition AudioData to a 16 kHz, mono, 16-bit WAV.

        whisper.cpp's CLI expects 16-bit WAV input. SpeechRecognition's
        microphone source is mono, and get_wav_data handles resampling and
        width conversion for us.
        """
        fd, wav_path = tempfile.mkstemp(prefix="ollama_stt_", suffix=".wav")
        os.close(fd)

        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        with open(wav_path, "wb") as handle:
            handle.write(wav_bytes)

        return wav_path

    def _transcribe_local_audio(self, audio, config=None):
        with self._stt_lock:
            backend = self._selected_stt_backend(config)
            wav_path = self._audio_data_to_wav(audio)

            try:
                if backend == "vulkan":
                    text = self._transcribe_whisper_cpp_vulkan(wav_path, config)
                    return text, "whisper.cpp/Vulkan"

                text = self._transcribe_faster_whisper_cpu(wav_path, config)
                return text, "Faster-Whisper/CPU"
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _transcribe_whisper_cpp_vulkan(self, wav_path, config=None):
        self._detect_local_stt_stack(config)

        if not self._whisper_cpp_cli or not self._whisper_cpp_model:
            raise RuntimeError("whisper.cpp Vulkan backend is not installed.")

        threads = max(1, int((config or {}).get("stt_threads", 4))) if config is not None else max(1, int(self.stt_threads_var.get() or 4))

        # English-only *.en models should use English. Multilingual models can
        # autodetect. The no-timestamps/no-prints flags produce clean stdout.
        selected_model = ((config or {}).get("stt_model", "base.en") if config is not None else (self.stt_model_var.get().strip() or "base.en")).lower()
        language = "en" if selected_model.endswith(".en") else "auto"

        command = [
            self._whisper_cpp_cli,
            "-m", self._whisper_cpp_model,
            "-f", wav_path,
            "-t", str(threads),
            "-l", language,
            "-nt",
            "-np",
        ]

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                "whisper.cpp/Vulkan transcription failed.\n\n"
                + (stderr[-4000:] if stderr else "No diagnostic output.")
            )

        return (proc.stdout or "").strip()

    def _faster_whisper_model_name(self, config=None):
        selected = (config or {}).get("stt_model", "base.en") if config is not None else (self.stt_model_var.get().strip() or "base.en")
        # Faster-Whisper accepts OpenAI model names such as base.en/small.en.
        return selected

    def _load_faster_whisper_cpu(self, config=None):
        model_name = self._faster_whisper_model_name(config)

        if (
            self._faster_whisper_model is not None
            and self._faster_whisper_loaded_name == model_name
        ):
            return self._faster_whisper_model

        from faster_whisper import WhisperModel

        threads = max(1, int((config or {}).get("stt_threads", 4))) if config is not None else max(1, int(self.stt_threads_var.get() or 4))

        self.ui_queue.put((
            "status",
            f"Loading local Faster-Whisper model {model_name} on CPU..."
        ))

        # int8 is a good default for the Ivy Bridge-era Xeon in Mac Pro 6,1:
        # lower memory use and broad CTranslate2 CPU compatibility.
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=1,
        )

        self._faster_whisper_model = model
        self._faster_whisper_loaded_name = model_name
        return model

    def _transcribe_faster_whisper_cpu(self, wav_path, config=None):
        model = self._load_faster_whisper_cpu(config)

        selected_model = self._faster_whisper_model_name(config).lower()
        language = "en" if selected_model.endswith(".en") else None

        segments, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        parts = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if segment_text:
                parts.append(segment_text)

        return " ".join(parts).strip()

    def stt_diagnostics(self):
        self._detect_local_stt_stack()
        vulkan_ok, vulkan_text = self._vulkan_summary()

        faster = self._python_module_available("faster_whisper")

        auto_backend = "Unavailable"
        try:
            auto_backend = self._selected_stt_backend()
        except Exception as exc:
            auto_backend = f"ERROR: {exc}"

        summary_lines = []
        if vulkan_text:
            # Keep the dialog readable.
            for line in vulkan_text.splitlines():
                if (
                    "Vulkan Instance Version" in line
                    or "deviceName" in line
                    or "deviceType" in line
                    or "GPU" in line
                    or "driverName" in line
                ):
                    summary_lines.append(line.strip())

        report = (
            "Fully Local Speech-to-Text Diagnostics\n"
            "--------------------------------------\n"
            f"Requested backend: {self.stt_backend_var.get()}\n"
            f"Auto-selected backend: {auto_backend}\n"
            f"STT model: {self.stt_model_var.get()}\n"
            f"CPU threads: {self.stt_threads_var.get()}\n"
            f"Last backend used: {self._last_stt_backend}\n\n"
            f"Faster-Whisper installed: {'YES' if faster else 'NO'}\n"
            f"whisper.cpp CLI: {self._whisper_cpp_cli or 'NOT FOUND'}\n"
            f"whisper.cpp model: {self._whisper_cpp_model or 'NOT FOUND'}\n"
            f"Vulkan usable: {'YES' if vulkan_ok else 'NO'}\n\n"
            "Vulkan summary:\n"
            + ("\n".join(summary_lines[:20]) if summary_lines else vulkan_text[:2000] or "No Vulkan details")
        )

        messagebox.showinfo("Local STT Diagnostics", report)

    def _api_url(self, endpoint, base_url=None):
        if base_url is None:
            base_url = self.ollama_url_var.get()

        base = self._sanitize_ollama_base_url(base_url)

        endpoint = str(endpoint or "")
        endpoint = endpoint.replace("\\r", "").replace("\\n", "")
        endpoint = endpoint.replace("\r", "").replace("\n", "")
        endpoint = endpoint.strip().strip("/")

        if endpoint.startswith("api/"):
            endpoint = endpoint[4:]

        if endpoint not in ("chat", "tags"):
            raise ValueError(
                f"Unsupported Ollama API endpoint: {endpoint!r}. "
                "Expected 'chat' or 'tags'."
            )

        final_url = f"{base}/api/{endpoint}"
        final_url = self._force_http_scheme(final_url)
        return self._validate_ollama_api_url(final_url, endpoint)

    def _openai_api_url(self, endpoint, base_url=None):
        if base_url is None:
            base_url = self.ollama_url_var.get()

        base = self._sanitize_ollama_base_url(base_url)

        endpoint = str(endpoint or "").strip().strip("/")
        if endpoint not in ("chat/completions", "models"):
            raise ValueError(
                f"Unsupported OpenAI-compatible Ollama endpoint: {endpoint!r}"
            )

        final_url = self._force_http_scheme(f"{base}/v1/{endpoint}")
        parsed = urlsplit(final_url)

        expected = f"/v1/{endpoint}"
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme in Ollama compatibility URL: {final_url}")
        if parsed.path != expected:
            raise ValueError(
                f"Invalid Ollama compatibility path {parsed.path!r}; "
                f"expected {expected!r}"
            )

        return final_url

    @staticmethod
    def _extract_openai_stream_token(chunk):
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content", "")
        return content if isinstance(content, str) else ""

    def load_ollama_models(self):
        threading.Thread(target=self._load_ollama_models_worker, daemon=True).start()

    def _load_ollama_models_worker(self):
        try:
            self.ui_queue.put(("status", "Loading Ollama models..."))
            base_url = self._sanitize_ollama_base_url(self.ollama_url_var.get())

            native_url = self._api_url("tags", base_url=base_url)
            response = requests.get(
                native_url,
                timeout=10,
                allow_redirects=False,
            )

            if response.status_code == 200:
                models = [
                    m.get("name", "")
                    for m in response.json().get("models", [])
                    if m.get("name")
                ]
                self.ui_queue.put(("status", f"{APP_BUILD} using native Ollama API"))
            elif response.status_code in (404, 405):
                compat_url = self._openai_api_url("models", base_url=base_url)
                response = requests.get(
                    compat_url,
                    timeout=10,
                    allow_redirects=False,
                )
                response.raise_for_status()
                models = [
                    m.get("id", "")
                    for m in response.json().get("data", [])
                    if m.get("id")
                ]
                self.ui_queue.put((
                    "status",
                    f"{APP_BUILD} using OpenAI-compatible Ollama API"
                ))
            else:
                response.raise_for_status()
                models = []

            self.ui_queue.put(("models", models))

        except Exception as exc:
            self._trace_exception(
                "OLLAMA_MODEL_LIST",
                exc,
                base_url=locals().get("base_url"),
                native_url=locals().get("native_url"),
                requested_url=locals().get("requested_url"),
            )
            self.ui_queue.put((
                "error",
                "Could not retrieve Ollama models.\n\n"
                f"{exc}\n\n"
                "Tried the native /api/tags endpoint and, when appropriate, "
                "the OpenAI-compatible /v1/models endpoint."
            ))

    def test_ollama(self):
        threading.Thread(target=self._test_ollama_worker, daemon=True).start()

    def _test_ollama_worker(self):
        try:
            self.ui_queue.put(("status", "Testing Ollama connection..."))
            base_url = self._sanitize_ollama_base_url(self.ollama_url_var.get())

            native_url = self._api_url("tags", base_url=base_url)
            response = requests.get(
                native_url,
                timeout=10,
                allow_redirects=False,
            )

            if response.status_code == 200:
                response.json()
                mode = "Native Ollama API"
                verified_url = native_url
            elif response.status_code in (404, 405):
                compat_url = self._openai_api_url("models", base_url=base_url)
                response = requests.get(
                    compat_url,
                    timeout=10,
                    allow_redirects=False,
                )
                response.raise_for_status()
                response.json()
                mode = "OpenAI-compatible Ollama API"
                verified_url = compat_url
            else:
                response.raise_for_status()
                mode = "Unknown"
                verified_url = native_url

            self.ui_queue.put((
                "info",
                f"Ollama connection successful.\n\n"
                f"API mode: {mode}\n"
                f"Verified endpoint: {verified_url}"
            ))
            self.ui_queue.put(("status", f"Ready — {APP_BUILD} — {mode}"))

        except Exception as exc:
            self._trace_exception(
                "OLLAMA_TEST",
                exc,
                base_url=locals().get("base_url"),
                verified_url=locals().get("verified_url"),
            )
            self.ui_queue.put((
                "error",
                "Ollama connection failed.\n\n"
                f"{exc}\n\n"
                "The application tested both the native Ollama API and "
                "the OpenAI-compatible API."
            ))

    def _standby_reminder_worker(self, done_event):
        """
        Repeat the configured standby phrase every N seconds until the first
        Ollama response token arrives (or the request terminates).
        """
        standby_cfg = self.app_config.get("standby", {})
        if not bool(standby_cfg.get("enabled", True)):
            return

        phrase = str(
            standby_cfg.get(
                "phrase",
                "I'm researching your request, please stand by."
            )
        ).strip()
        interval = max(
            2.0,
            float(standby_cfg.get("interval_seconds", 10))
        )

        if not phrase:
            return

        # Immediate acknowledgement.
        self._queue_tts(
            phrase,
            label="Researching",
            cancel_event=done_event
        )

        # Repeat every interval until response begins.
        while not self.shutdown_requested:
            if done_event.wait(interval):
                return
            self._queue_tts(
                phrase,
                label="Researching",
                cancel_event=done_event
            )

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

    def _trace_ollama_request(self, stage, url, payload):
        """Diagnostics must never be able to abort an Ollama request."""
        if not self._trace_enabled():
            return
        try:
            print("\n" + "=" * 78, flush=True)
            print(f"[OLLAMA TRACE] {stage}", flush=True)
            print(f"[OLLAMA TRACE] URL repr: {url!r}", flush=True)
            print(f"[OLLAMA TRACE] URL text: {url}", flush=True)
            print(f"[OLLAMA TRACE] payload type: {type(payload).__name__}", flush=True)
            print("[OLLAMA TRACE] payload JSON:", flush=True)
            try:
                print(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[OLLAMA TRACE] payload serialization failed: {exc!r}",
                    flush=True,
                )
                print(f"[OLLAMA TRACE] payload repr: {payload!r}", flush=True)

            if isinstance(payload, dict):
                print(
                    f"[OLLAMA TRACE] model repr: "
                    f"{payload.get('model')!r}",
                    flush=True,
                )
                print(
                    f"[OLLAMA TRACE] stream repr: "
                    f"{payload.get('stream')!r}",
                    flush=True,
                )
                trace_messages = payload.get("messages")
                print(
                    f"[OLLAMA TRACE] messages repr: {trace_messages!r}",
                    flush=True,
                )
                if isinstance(trace_messages, list):
                    for i, msg in enumerate(trace_messages):
                        print(
                            f"[OLLAMA TRACE] message[{i}] repr: {msg!r}",
                            flush=True,
                        )
            print("=" * 78 + "\n", flush=True)
        except Exception as exc:
            print(
                f"[OLLAMA TRACE] tracer failure ignored: {exc!r}",
                flush=True,
            )

    def _stream_ollama_response(self, user_text, request_config=None):
        assistant_text = ""
        request_config = dict(request_config or {})
        first_token = threading.Event()

        with self._response_wait_lock:
            # Cancel any stale reminder from an older request.
            if self._response_wait_event is not None:
                self._response_wait_event.set()
            self._response_wait_event = first_token

        try:
            # Use the current UI-selected runtime model as the source of truth.
            # request_config may originate from a listener snapshot created
            # minutes earlier.
            model = self._get_runtime_ollama_model()
            if not model:
                model = request_config.get("ollama_model", "").strip()
            if not model:
                model = str(
                    self.app_config.get("ollama", {}).get("model", "llama3.2")
                ).strip()
            if not model:
                raise ValueError("Please enter or select an Ollama model.")

            system = request_config.get("system_prompt", "").strip()
            base_url = self._sanitize_ollama_base_url(
                request_config.get(
                    "ollama_url",
                    self._configured_ollama_url()
                )
            )
            auto_speak = bool(request_config.get("auto_speak", True))

            request_messages = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(self.messages)
            request_messages.append({"role": "user", "content": user_text})

            payload = {
                "model": model,
                "messages": request_messages,
                "stream": True,
            }

            self.ui_queue.put((
                "response_start",
                f"\nYou: {user_text}\nAssistant: "
            ))
            self.ui_queue.put((
                "status",
                f"Sending prompt to Ollama at {base_url}..."
            ))

            reminder_thread = threading.Thread(
                target=self._standby_reminder_worker,
                args=(first_token,),
                daemon=True,
            )
            reminder_thread.start()

            chat_url = self._api_url("chat", base_url=base_url)
            chat_url = self._force_http_scheme(chat_url)

            parsed_chat = urlsplit(chat_url)
            if parsed_chat.scheme not in ("http", "https"):
                raise ValueError(
                    f"Blocked malformed Ollama scheme before POST: "
                    f"{parsed_chat.scheme!r}"
                )
            if parsed_chat.path != "/api/chat":
                raise ValueError(
                    f"Blocked incorrect Ollama path before POST: "
                    f"{parsed_chat.path!r}"
                )

            # Show the canonical server root back in the UI so malformed input
            # cannot remain visually misleading after it has been repaired.
            self.ui_queue.put(("ollama_url", self._sanitize_ollama_base_url(base_url)))
            self.ui_queue.put(("status", f"{APP_BUILD} POST {chat_url}"))

            api_mode = "native"
            requested_url = chat_url

            self._trace_print(f"[OLLAMA TRACE] request_config repr: {request_config!r}")
            self._trace_print(
                f"[OLLAMA TRACE] request_config model repr: "
                f"{request_config.get('ollama_model')!r}"
            )
            self._trace_print(
                f"[OLLAMA TRACE] live runtime model repr: "
                f"{self._get_runtime_ollama_model()!r}"
            )
            self._trace_print(f"[OLLAMA TRACE] selected model repr: {model!r}")
            self._trace_print(f"[OLLAMA TRACE] base_url repr: {base_url!r}")
            self._trace_print(
                f"[OLLAMA TRACE] request_messages repr: {request_messages!r}"
            )
            self._trace_ollama_request("NATIVE POST /api/chat", chat_url, payload)

            response = requests.post(
                chat_url,
                json=payload,
                stream=True,
                timeout=(10, 300),
                allow_redirects=False,
            )

            self._trace_print(
                f"[OLLAMA TRACE] native HTTP status: {response.status_code}"
            )
            self._trace_print(
                f"[OLLAMA TRACE] native response.url repr: {response.url!r}"
            )
            self._trace_print(
                f"[OLLAMA TRACE] native headers: {dict(response.headers)!r}"
            )
            if response.status_code >= 400:
                try:
                    self._trace_print(
                        f"[OLLAMA TRACE] native response body: {response.text[:4000]!r}"
                    )
                except Exception as trace_exc:
                    self._trace_print(
                        f"[OLLAMA TRACE] could not read response body: {trace_exc!r}"
                    )

            # A correct /api/chat URL returning 404 means this server/port
            # does not expose the native chat endpoint. Ollama also exposes
            # an OpenAI-compatible endpoint, so try that automatically.
            if response.status_code in (404, 405):
                response.close()
                api_mode = "openai"
                requested_url = self._openai_api_url(
                    "chat/completions",
                    base_url=base_url,
                )
                self.ui_queue.put((
                    "status",
                    f"{APP_BUILD}: native /api/chat returned 404; "
                    f"trying {requested_url}"
                ))
                self._trace_ollama_request(
                    "OPENAI-COMPAT POST /v1/chat/completions",
                    requested_url,
                    payload,
                )
                response = requests.post(
                    requested_url,
                    json=payload,
                    stream=True,
                    timeout=(10, 300),
                    allow_redirects=False,
                )

            if api_mode == "openai":
                self._trace_print(
                    f"[OLLAMA TRACE] fallback HTTP status: {response.status_code}"
                )
                self._trace_print(
                    f"[OLLAMA TRACE] fallback response.url repr: {response.url!r}"
                )
                self._trace_print(
                    f"[OLLAMA TRACE] fallback headers: {dict(response.headers)!r}"
                )
                if response.status_code >= 400:
                    try:
                        self._trace_print(
                            f"[OLLAMA TRACE] fallback response body: {response.text[:4000]!r}"
                        )
                    except Exception as trace_exc:
                        self._trace_print(
                            f"[OLLAMA TRACE] could not read fallback body: {trace_exc!r}"
                        )

            try:
                response.raise_for_status()

                self.ui_queue.put((
                    "status",
                    f"{APP_BUILD} receiving via "
                    f"{'native Ollama' if api_mode == 'native' else 'OpenAI-compatible'} API"
                ))

                for raw_line in response.iter_lines(decode_unicode=True):
                    if self.shutdown_requested:
                        first_token.set()
                        return
                    if not raw_line:
                        continue

                    if api_mode == "native":
                        chunk = json.loads(raw_line)
                        token = (chunk.get("message") or {}).get("content", "")
                        done = bool(chunk.get("done", False))
                    else:
                        line = raw_line.strip()
                        if line.startswith("data:"):
                            line = line[5:].strip()

                        if line == "[DONE]":
                            first_token.set()
                            break

                        if not line:
                            continue

                        chunk = json.loads(line)
                        token = self._extract_openai_stream_token(chunk)
                        choices = chunk.get("choices") or []
                        done = bool(
                            choices
                            and choices[0].get("finish_reason") is not None
                        )

                    if token:
                        if not first_token.is_set():
                            first_token.set()

                        assistant_text += token
                        self.ui_queue.put(("response_chunk", token))

                    if done:
                        first_token.set()
                        break

            except requests.exceptions.HTTPError as http_exc:
                body = ""
                try:
                    body = response.text[:2000]
                except Exception:
                    pass
                raise RuntimeError(
                    "Ollama rejected both API styles.\n\n"
                    f"Last URL: {requested_url}\n"
                    f"Status: {response.status_code}\n"
                    f"Response: {body or '(empty)'}"
                ) from http_exc
            finally:
                response.close()

            first_token.set()
            assistant_text = assistant_text.strip()

            if assistant_text:
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                })
                self.ui_queue.put(("response_chunk", "\n"))

                if auto_speak:
                    self.ui_queue.put((
                        "status",
                        "Preparing spoken response..."
                    ))
                    self._queue_tts(
                        assistant_text,
                        label="Speaking response"
                    )
                else:
                    self.ui_queue.put((
                        "status",
                        "Listening..." if self.is_listening else "Ready"
                    ))
            else:
                self.ui_queue.put((
                    "status",
                    "Ollama returned an empty response."
                ))

        except requests.exceptions.ConnectionError as exc:
            first_token.set()
            self._trace_exception(
                "OLLAMA_CONNECTION",
                exc,
                request_config=request_config,
                model=locals().get("model"),
                base_url=locals().get("base_url"),
                requested_url=locals().get("requested_url"),
                payload=locals().get("payload"),
                request_messages=locals().get("request_messages"),
            )
            self.ui_queue.put((
                "error",
                "Cannot connect to Ollama.\\n\\n"
                f"{exc}\\n\\n"
                "Verify the configured host/IP and port."
            ))
        except requests.exceptions.HTTPError as exc:
            first_token.set()
            detail = ""
            try:
                detail = exc.response.text
            except Exception:
                pass
            self._trace_exception(
                "OLLAMA_HTTP",
                exc,
                request_config=request_config,
                model=locals().get("model"),
                base_url=locals().get("base_url"),
                requested_url=locals().get("requested_url"),
                payload=locals().get("payload"),
                request_messages=locals().get("request_messages"),
                response_status=getattr(exc.response, "status_code", None),
                response_url=getattr(exc.response, "url", None),
                response_body=detail[:4000] if detail else "",
            )
            self.ui_queue.put((
                "error",
                f"Ollama HTTP error:\\n{exc}\\n\\n{detail}"
            ))
        except Exception as exc:
            first_token.set()
            self._trace_exception(
                "OLLAMA_GENERAL",
                exc,
                request_config=request_config,
                model=locals().get("model"),
                base_url=locals().get("base_url"),
                requested_url=locals().get("requested_url"),
                payload=locals().get("payload"),
                request_messages=locals().get("request_messages"),
            )
            self.ui_queue.put((
                "error",
                f"Ollama request error:\\n"
                f"{type(exc).__name__}: {exc}"
            ))
        finally:
            first_token.set()
            with self._response_wait_lock:
                if self._response_wait_event is first_token:
                    self._response_wait_event = None
            self.is_processing = False

    def _queue_tts(self, text, label="Speaking", cancel_event=None):
        """Queue speech, optionally skipping it if cancel_event becomes set."""
        text = (text or "").strip()
        if not text:
            return
        self.audio_busy.set()
        self.tts_queue.put((text, label, cancel_event))

    def _tts_worker(self):
        """Process queued TTS jobs serially using a reliable OS-native backend."""
        while not self.shutdown_requested:
            try:
                text = self.tts_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if text is None:
                self.tts_queue.task_done()
                break

            cancel_event = None
            if isinstance(text, tuple):
                if len(text) == 3:
                    text, speech_label, cancel_event = text
                else:
                    text, speech_label = text
            else:
                speech_label = "Speaking"

            # A standby reminder can become obsolete while waiting behind
            # another TTS job. Skip it rather than delaying the real response.
            if cancel_event is not None and cancel_event.is_set():
                self.tts_queue.task_done()
                if self.tts_queue.empty():
                    self.audio_busy.clear()
                continue

            try:
                self.is_speaking = True
                self.avatar_speaking = True
                self.audio_busy.set()
                self.ui_queue.put(("status", f"{speech_label}..."))
                self.ui_queue.put(("avatar_status", speech_label))
                self._speak_text(text)

            except Exception as exc:
                self.ui_queue.put(("error", f"Text-to-speech error:\n{exc}"))

            finally:
                self.is_speaking = False
                self.avatar_speaking = False
                self.avatar_mouth_level = 0.0
                self.ui_queue.put(("avatar_status", "Listening" if self.is_listening else "Idle"))
                self.tts_queue.task_done()

                # Keep listening paused if another speech item is waiting.
                if self.tts_queue.empty():
                    self.audio_busy.clear()

                if self.is_processing:
                    self.ui_queue.put(("status", "Waiting for Ollama response..."))
                    self.ui_queue.put(("avatar_status", "Researching"))
                else:
                    if (
                        self.is_listening
                        and hasattr(self, "command_listening_event")
                        and not self.command_listening_event.is_set()
                        and self.voice_triggers_enabled
                    ):
                        self.ui_queue.put(("status", self._trigger_standby_status()))
                        self.ui_queue.put(("avatar_status", "Trigger Standby"))
                    else:
                        self.ui_queue.put(
                            ("status", "Listening..." if self.is_listening else "Ready")
                        )

    def _detect_piper_voice(self):
        """Locate the Piper package and the preferred local female voice."""
        try:
            import importlib.util
            self.piper_available = importlib.util.find_spec("piper") is not None
        except Exception:
            self.piper_available = False

        candidates = []

        env_voice = os.environ.get("PIPER_VOICE_MODEL")
        if env_voice:
            candidates.append(Path(env_voice).expanduser())

        candidates.extend([
            Path.home() / ".local" / "share" / "ollama-voice-assistant" / "piper" / f"{self.piper_voice_name}.onnx",
            Path.cwd() / "voices" / f"{self.piper_voice_name}.onnx",
            Path.cwd() / f"{self.piper_voice_name}.onnx",
        ])

        self.piper_voice_path = None

        for candidate in candidates:
            if candidate.exists() and Path(str(candidate) + ".json").exists():
                self.piper_voice_path = candidate
                break

        if self.piper_voice_path is None:
            self.piper_available = False

    def _load_piper_voice(self):
        """Lazy-load Piper once and reuse the ONNX voice for low-latency replies."""
        if self.piper_voice is not None:
            return self.piper_voice

        self._detect_piper_voice()

        if not self.piper_available or not self.piper_voice_path:
            return None

        with self._piper_lock:
            if self.piper_voice is not None:
                return self.piper_voice

            try:
                from piper import PiperVoice
                self.piper_voice = PiperVoice.load(str(self.piper_voice_path))
            except Exception:
                self.piper_voice = None

        return self.piper_voice

    def _synthesize_piper_wav(self, text, wav_path):
        """Render natural female speech locally with Piper neural TTS."""
        voice = self._load_piper_voice()
        if voice is None:
            return False

        try:
            from piper import SynthesisConfig

            # Slightly slower than default for a calm assistant-style voice.
            syn_config = SynthesisConfig(
                volume=1.0,
                length_scale=1.06,
                noise_scale=0.62,
                noise_w_scale=0.72,
                normalize_audio=True,
            )

            with wave.open(wav_path, "wb") as wav_file:
                voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=syn_config,
                )

            return True
        except Exception:
            return False

    def _detect_linux_audio_stack(self):
        """
        Detect the Ubuntu audio playback stack.

        Preferred order:
          1. pw-play  -> native PipeWire playback
          2. paplay   -> PulseAudio compatibility layer
          3. aplay    -> ALSA fallback
        """
        if platform.system().lower() != "linux":
            self.audio_player_name = "unsupported"
            self.audio_player_cmd = None
            self.tts_synth = None
            return

        self.tts_synth = shutil.which("espeak-ng") or shutil.which("espeak")

        players = [
            ("PipeWire (pw-play)", shutil.which("pw-play")),
            ("PulseAudio (paplay)", shutil.which("paplay")),
            ("ALSA (aplay)", shutil.which("aplay")),
        ]

        self.audio_player_name = None
        self.audio_player_cmd = None

        for name, command in players:
            if command:
                self.audio_player_name = name
                self.audio_player_cmd = command
                break

    def _make_wav_envelope(self, wav_path, frame_ms=50):
        """
        Build a normalized RMS envelope from the synthesized WAV.

        This is used by the avatar so mouth movement follows the actual
        generated speech amplitude rather than a random animation.
        """
        envelope = []

        try:
            with wave.open(wav_path, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()

                if sample_width != 2:
                    return []

                frames_per_bucket = max(1, int(sample_rate * frame_ms / 1000.0))

                while True:
                    raw = wav_file.readframes(frames_per_bucket)
                    if not raw:
                        break

                    samples = array("h")
                    samples.frombytes(raw)

                    if not samples:
                        envelope.append(0.0)
                        continue

                    # Mac Pro 6,1 is x86_64/little-endian; WAV PCM is little-endian.
                    total = 0.0
                    count = len(samples)

                    for sample in samples:
                        total += float(sample) * float(sample)

                    rms = math.sqrt(total / max(1, count))
                    normalized = min(1.0, rms / 9000.0)

                    # Expand quiet speech while leaving silence near zero.
                    if normalized < 0.025:
                        normalized = 0.0
                    else:
                        normalized = min(1.0, normalized * 1.65)

                    envelope.append(normalized)

        except Exception:
            return []

        return envelope

    def _speak_text(self, text):
        """
        Ubuntu 26.04 local TTS path.

        Preferred:
            Piper neural female voice -> WAV -> speaker

        Fallback:
            espeak-ng female voice -> WAV -> speaker

        The generated WAV also drives the avatar's real amplitude envelope.
        """
        text = (text or "").strip()
        if not text:
            return

        if platform.system().lower() != "linux":
            raise RuntimeError(
                "This build is specifically configured for Ubuntu/Linux."
            )

        self._detect_linux_audio_stack()
        self._detect_piper_voice()

        fd, wav_path = tempfile.mkstemp(prefix="ollama_voice_", suffix=".wav")
        os.close(fd)

        try:
            synthesized = self._synthesize_piper_wav(text, wav_path)

            if not synthesized:
                if not self.tts_synth:
                    raise RuntimeError(
                        "Piper neural TTS is unavailable and espeak-ng fallback "
                        "was not found. Run the v7 Ubuntu installer."
                    )

                # Female espeak fallback if Piper cannot load.
                synth_command = [
                    self.tts_synth,
                    "-v", "en-us+f3",
                    "-s", "165",
                    "-a", "180",
                    "-w", wav_path,
                    text,
                ]

                synth = subprocess.run(
                    synth_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if synth.returncode != 0:
                    raise RuntimeError(
                        "Fallback speech synthesis failed:\\n"
                        + (synth.stderr.strip() or "unknown synthesis error")
                    )

            envelope = self._make_wav_envelope(
                wav_path,
                frame_ms=self.avatar_envelope_frame_ms
            )
            self.avatar_envelope = envelope

            if not self.audio_player_cmd:
                raise RuntimeError(
                    "No Linux WAV playback backend was found. Install "
                    "pipewire-bin, pulseaudio-utils, or alsa-utils."
                )

            player_name = os.path.basename(self.audio_player_cmd)

            if "pw-play" in player_name:
                command = [self.audio_player_cmd, wav_path]
            elif "paplay" in player_name:
                command = [self.audio_player_cmd, wav_path]
            else:
                command = [self.audio_player_cmd, "-q", wav_path]

            self.avatar_envelope_started = time.monotonic()

            playback = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            if playback.returncode != 0:
                raise RuntimeError(
                    f"{self.audio_player_name} playback failed:\\n"
                    + (playback.stderr.strip() or "unknown playback error")
                )

        finally:
            self.avatar_envelope = []
            self.avatar_envelope_started = 0.0
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def audio_diagnostics(self):
        """Show the Linux audio components detected by this application."""
        self._detect_linux_audio_stack()

        try:
            microphones = sr.Microphone.list_microphone_names()
        except Exception as exc:
            microphones = [f"ERROR: {exc}"]

        mic_lines = "\\n".join(
            f"  {i}: {name}" for i, name in enumerate(microphones[:20])
        )
        if len(microphones) > 20:
            mic_lines += f"\\n  ... {len(microphones) - 20} more"

        report = (
            "Ubuntu / Linux Audio Diagnostics\\n"
            "--------------------------------\\n"
            f"Platform: {platform.platform()}\\n"
            f"Architecture: {platform.machine()}\\n"
            f"TTS synthesizer: {self.tts_synth or 'NOT FOUND'}\\n"
            f"Playback backend: {self.audio_player_name or 'NOT FOUND'}\\n"
            f"Playback command: {self.audio_player_cmd or 'NOT FOUND'}\\n"
            f"Ollama URL: {self.ollama_url_var.get().strip()}\\n\\n"
            "PortAudio microphone devices:\\n"
            f"{mic_lines or '  None'}"
        )

        messagebox.showinfo("Audio Diagnostics", report)

    def _animate_avatar(self):
        """
        UI-thread animation loop.

        Ubuntu WAV-based TTS exposes an amplitude envelope, so the animation
        follows the real playback window. While TTS is active, mouth opening,
        jaw position and head motion vary naturally; when playback returns,
        the face immediately settles back to idle.
        """
        if self.shutdown_requested:
            return

        now = time.time()
        self.avatar_phase += 0.30

        # Natural blink every few seconds.
        if self.avatar_blink > 0:
            self.avatar_blink -= 1
        elif now >= self.avatar_next_blink:
            self.avatar_blink = 3
            self.avatar_next_blink = now + random.uniform(2.5, 5.0)

        speaking = bool(self.avatar_speaking)

        if speaking:
            # Prefer the actual amplitude envelope from the espeak-ng WAV.
            mouth = None

            if self.avatar_envelope and self.avatar_envelope_started:
                elapsed_ms = (
                    time.monotonic() - self.avatar_envelope_started
                ) * 1000.0

                index = int(
                    elapsed_ms / float(self.avatar_envelope_frame_ms)
                )

                if 0 <= index < len(self.avatar_envelope):
                    mouth = self.avatar_envelope[index]

            # Fallback animation during the tiny interval before playback starts.
            if mouth is None:
                target = 0.18 + 0.75 * random.random()
                self.avatar_mouth_level = (
                    self.avatar_mouth_level * 0.45 + target * 0.55
                )
                mouth = self.avatar_mouth_level

            # Smooth abrupt amplitude changes for more natural jaw motion.
            self.avatar_mouth_level = (
                self.avatar_mouth_level * 0.35 + mouth * 0.65
            )
            mouth = max(0.06, min(1.0, self.avatar_mouth_level))
        else:
            self.avatar_mouth_level *= 0.50
            mouth = self.avatar_mouth_level

        blink = self.avatar_blink > 0
        self._draw_avatar(mouth, blink, self.avatar_phase if speaking else 0.0)

        self.root.after(70, self._animate_avatar)

    def _draw_avatar(self, mouth_level, blink, phase):
        """Draw a more human-looking female assistant avatar."""
        c = self.avatar_canvas
        c.delete("all")

        try:
            w = max(c.winfo_width(), 330)
            h = max(c.winfo_height(), 330)
        except Exception:
            w, h = 330, 330

        cx = w / 2
        cy = h / 2 + 8

        if self.avatar_speaking:
            cx += 2.0 * math.sin(phase * 0.42)
            cy += 1.5 * math.sin(phase * 0.66)

        # Soft background.
        c.create_oval(
            cx - 145, cy - 155, cx + 145, cy + 170,
            fill="#dde8f0", outline=""
        )

        # Shoulders and blouse.
        c.create_arc(
            cx - 130, cy + 90, cx + 130, cy + 235,
            start=0, extent=180,
            fill="#44576f", outline="#34475f", width=2
        )
        c.create_polygon(
            cx - 38, cy + 94,
            cx, cy + 132,
            cx + 38, cy + 94,
            fill="#f6f7fb", outline=""
        )

        # Long hair behind the face.
        hair = "#4a3028"
        hair_highlight = "#65483c"
        c.create_oval(
            cx - 111, cy - 143, cx + 111, cy + 145,
            fill=hair, outline="#34221d", width=2
        )
        c.create_polygon(
            cx - 107, cy - 55,
            cx - 126, cy + 145,
            cx - 70, cy + 160,
            cx - 55, cy + 38,
            fill=hair, outline=""
        )
        c.create_polygon(
            cx + 107, cy - 55,
            cx + 126, cy + 145,
            cx + 70, cy + 160,
            cx + 55, cy + 38,
            fill=hair, outline=""
        )

        # Neck.
        skin = "#e7b58f"
        skin_edge = "#a8755d"
        c.create_rectangle(
            cx - 34, cy + 68, cx + 34, cy + 118,
            fill=skin, outline=""
        )

        # Face: slightly tapered jaw rather than a simple oval.
        face_points = [
            cx - 82, cy - 92,
            cx - 91, cy - 28,
            cx - 78, cy + 38,
            cx - 53, cy + 80,
            cx, cy + 105,
            cx + 53, cy + 80,
            cx + 78, cy + 38,
            cx + 91, cy - 28,
            cx + 82, cy - 92,
        ]
        c.create_polygon(
            face_points,
            fill=skin, outline=skin_edge, width=2, smooth=True
        )

        # Hairline and side framing.
        c.create_arc(
            cx - 92, cy - 137, cx + 92, cy - 35,
            start=3, extent=174, style="arc",
            outline=hair_highlight, width=22
        )
        c.create_arc(
            cx - 104, cy - 105, cx - 42, cy + 62,
            start=88, extent=122, style="arc",
            outline=hair, width=17
        )
        c.create_arc(
            cx + 42, cy - 105, cx + 104, cy + 62,
            start=-30, extent=122, style="arc",
            outline=hair, width=17
        )

        # Eyebrows.
        brow_y = cy - 36
        c.create_arc(
            cx - 69, brow_y - 9, cx - 24, brow_y + 11,
            start=15, extent=145, style="arc",
            outline="#5b4034", width=3
        )
        c.create_arc(
            cx + 24, brow_y - 9, cx + 69, brow_y + 11,
            start=20, extent=145, style="arc",
            outline="#5b4034", width=3
        )

        # Eyes and lashes.
        eye_y = cy - 13
        if blink:
            c.create_arc(
                cx - 70, eye_y - 4, cx - 24, eye_y + 10,
                start=200, extent=140, style="arc",
                outline="#2f2928", width=3
            )
            c.create_arc(
                cx + 24, eye_y - 4, cx + 70, eye_y + 10,
                start=200, extent=140, style="arc",
                outline="#2f2928", width=3
            )
        else:
            for ex in (cx - 47, cx + 47):
                c.create_oval(
                    ex - 21, eye_y - 12, ex + 21, eye_y + 12,
                    fill="#fafcff", outline="#6f6460", width=2
                )
                pupil_shift = 1.5 * math.sin(phase * 0.22) if self.avatar_speaking else 0
                c.create_oval(
                    ex - 7 + pupil_shift, eye_y - 8,
                    ex + 7 + pupil_shift, eye_y + 8,
                    fill="#6b583e", outline=""
                )
                c.create_oval(
                    ex - 3 + pupil_shift, eye_y - 4,
                    ex + 3 + pupil_shift, eye_y + 4,
                    fill="#1d2227", outline=""
                )
                c.create_oval(
                    ex - 1 + pupil_shift, eye_y - 5,
                    ex + 2 + pupil_shift, eye_y - 2,
                    fill="white", outline=""
                )

            # Upper lashes.
            c.create_arc(
                cx - 70, eye_y - 15, cx - 24, eye_y + 7,
                start=18, extent=145, style="arc",
                outline="#2f2928", width=3
            )
            c.create_arc(
                cx + 24, eye_y - 15, cx + 70, eye_y + 7,
                start=18, extent=145, style="arc",
                outline="#2f2928", width=3
            )

        # Nose with subtle shading.
        c.create_line(
            cx - 2, cy - 1,
            cx - 7, cy + 30,
            cx + 2, cy + 34,
            smooth=True, width=2, fill="#b47f65"
        )
        c.create_arc(
            cx - 12, cy + 24, cx + 13, cy + 41,
            start=205, extent=130, style="arc",
            outline="#b47f65", width=2
        )

        # Cheek highlights.
        c.create_oval(cx - 74, cy + 30, cx - 48, cy + 42, fill="#e9a59d", outline="")
        c.create_oval(cx + 48, cy + 30, cx + 74, cy + 42, fill="#e9a59d", outline="")

        # Mouth driven by the actual speech amplitude.
        mouth_y = cy + 61

        if mouth_level < 0.08:
            # Resting smile.
            c.create_arc(
                cx - 39, mouth_y - 11,
                cx + 39, mouth_y + 20,
                start=205, extent=130,
                style="arc", outline="#a23f55", width=4
            )
            c.create_arc(
                cx - 31, mouth_y - 4,
                cx + 31, mouth_y + 14,
                start=20, extent=140,
                style="arc", outline="#d77588", width=2
            )
        else:
            open_h = 8 + mouth_level * 31
            open_w = 50 + mouth_level * 22

            # Lip outline.
            c.create_oval(
                cx - open_w / 2 - 3,
                mouth_y - open_h / 2 - 2,
                cx + open_w / 2 + 3,
                mouth_y + open_h / 2 + 2,
                fill="#b14b62", outline="#96394f", width=2
            )

            # Mouth interior.
            c.create_oval(
                cx - open_w / 2,
                mouth_y - open_h / 2,
                cx + open_w / 2,
                mouth_y + open_h / 2,
                fill="#5c2730", outline=""
            )

            if mouth_level > 0.36:
                c.create_arc(
                    cx - open_w / 2 + 5,
                    mouth_y - open_h / 2 + 2,
                    cx + open_w / 2 - 5,
                    mouth_y - open_h / 2 + max(12, open_h * 0.45),
                    start=0, extent=180,
                    fill="#fffaf7", outline="#fffaf7", style="pieslice"
                )

            if mouth_level > 0.58:
                c.create_arc(
                    cx - open_w * 0.27,
                    mouth_y + 3,
                    cx + open_w * 0.27,
                    mouth_y + open_h / 2 - 1,
                    start=180, extent=180,
                    fill="#cf7180", outline="#cf7180", style="pieslice"
                )

        # Earrings.
        c.create_oval(cx - 94, cy + 28, cx - 86, cy + 38, fill="#d7b450", outline="")
        c.create_oval(cx + 86, cy + 28, cx + 94, cy + 38, fill="#d7b450", outline="")

        # Speech halo.
        if self.avatar_speaking:
            pulse = 5 + 3 * abs(math.sin(phase * 0.62))
            c.create_oval(
                cx - 119 - pulse, cy - 150 - pulse,
                cx + 119 + pulse, cy + 158 + pulse,
                outline="#7ba5c3", width=2
            )

    def test_speaker(self):
        self._queue_tts("Hello. I am your local voice assistant. My neural female voice is working correctly.", label="Testing female voice")

    def speak_last_response(self):
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and message.get("content", "").strip():
                self._queue_tts(message["content"].strip())
                return
        messagebox.showinfo("Speak Response", "There is no Ollama response to speak yet.")

    def clear_conversation(self):
        self.messages.clear()
        self.heard_text.delete("1.0", "end")
        self.response_text.delete("1.0", "end")
        self.status_var.set("Conversation cleared.")

    def save_conversation(self):
        heard = self.heard_text.get("1.0", "end-1c").strip()
        responses = self.response_text.get("1.0", "end-1c").strip()
        if not heard and not responses:
            messagebox.showinfo("Save Conversation", "There is no conversation to save.")
            return
        path = filedialog.asksaveasfilename(title="Save Conversation", defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("SPEECH HEARD / TRANSCRIBED\n" + "=" * 60 + "\n" + heard + "\n\n")
            f.write("OLLAMA CONVERSATION\n" + "=" * 60 + "\n" + responses + "\n")
        self.status_var.set(f"Conversation saved: {path}")

    @staticmethod
    def _normalize_ui_text(value):
        if value is None:
            return ""
        text = str(value)
        return (
            text
            .replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
        )

    def _process_ui_queue(self):
        try:
            while True:
                action, value = self.ui_queue.get_nowait()
                if action == "status":
                    self.status_var.set(value)
                elif action == "heard":
                    display_value = self._normalize_ui_text(value)
                    self.heard_text.insert("end", display_value)
                    self.heard_text.see("end")
                elif action in ("response_start", "response_chunk"):
                    display_value = self._normalize_ui_text(value)
                    self.response_text.insert("end", display_value)
                    self.response_text.see("end")
                elif action == "avatar_status":
                    self.avatar_status_var.set(value)
                elif action == "ollama_url":
                    self.ollama_url_var.set(value)
                elif action == "models":
                    self.model_combo["values"] = value
                    if value and self.model_var.get() not in value:
                        self.model_var.set(value[0])
                    if self.model_var.get().strip():
                        self._set_runtime_ollama_model(
                            self.model_var.get().strip()
                        )
                    self.status_var.set(
                        f"Loaded {len(value)} Ollama model(s). "
                        f"Selected: {self.model_var.get().strip()}"
                    )
                elif action == "trigger_state":
                    active = bool(value)
                    if active:
                        self.start_button.config(state="disabled")
                        self.stop_button.config(state="normal")
                        self.avatar_status_var.set("Listening")
                        if not self.audio_busy.is_set() and not self.is_processing:
                            self.status_var.set("Listening for requests...")
                    else:
                        self.start_button.config(state="normal")
                        self.stop_button.config(state="disabled")
                        self.avatar_status_var.set("Trigger Standby")
                        if not self.audio_busy.is_set() and not self.is_processing:
                            self.status_var.set(self._trigger_standby_status())

                elif action == "listener_stopped":
                    self.is_listening = False
                    self.command_listening_event.clear()
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.mic_combo.config(state="readonly")
                    self.avatar_status_var.set("Idle")
                    if not self.audio_busy.is_set() and not self.is_processing:
                        self.status_var.set("Ready")
                elif action == "info":
                    messagebox.showinfo("Ollama", value)
                elif action == "info_mic":
                    messagebox.showinfo("Microphone Test", value)
                elif action == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Error", value)
        except queue.Empty:
            pass
        if not self.shutdown_requested:
            self.root.after(80, self._process_ui_queue)

    def _on_close(self):
        self.shutdown_requested = True
        self.command_listening_event.clear()
        self.is_listening = False
        self.audio_busy.clear()
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    OllamaVoiceAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
