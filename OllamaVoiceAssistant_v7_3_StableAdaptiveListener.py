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
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import requests
import speech_recognition as sr
import pyaudio
import platform
import shutil
import subprocess
import random
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
        self.root.title("Ollama Voice Assistant v7.3 - Stable Adaptive Listener / Female / Local STT / AMD Vulkan")
        self.root.geometry("1250x850")
        self.root.minsize(1000, 700)

        self.recognizer = sr.Recognizer()
        # Long-form dictation tuning:
        # - up to 120 seconds in a single captured utterance
        # - allow a few seconds of hesitation before considering speech complete
        self.recognizer.pause_threshold = 3.0
        self.recognizer.non_speaking_duration = 1.0
        self.microphone = None
        self._mic_display_to_index = {}
        self._listener_config = {}

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
        ttk.Button(settings, text="Refresh", command=self._load_microphones).grid(row=0, column=2, padx=(0, 4), pady=4)
        ttk.Button(settings, text="Test Mic", command=self.test_microphone).grid(row=0, column=2, sticky="e", padx=(0, 12), pady=4)

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
        self.ollama_url_var = tk.StringVar(value="http://192.168.1.165:11434")
        ttk.Entry(settings, textvariable=self.ollama_url_var).grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)

        ttk.Button(settings, text="Load Models", command=self.load_ollama_models).grid(row=2, column=2, padx=(0, 12), pady=4)

        ttk.Label(settings, text="Model:").grid(row=2, column=3, sticky="w", padx=(0, 6), pady=4)
        self.model_var = tk.StringVar(value="llama3.2")
        self.model_combo = ttk.Combobox(settings, textvariable=self.model_var, width=25)
        self.model_combo.grid(row=2, column=4, sticky="ew", pady=4)

        ttk.Label(settings, text="System prompt:").grid(row=3, column=0, sticky="nw", padx=(0, 6), pady=4)
        self.system_prompt = tk.Text(settings, height=3, wrap="word")
        self.system_prompt.grid(row=3, column=1, columnspan=4, sticky="ew", pady=4)
        self.system_prompt.insert("1.0", "Your name is Alice and you are a helpful local voice assistant. Answer conversationally and keep spoken responses reasonably concise.")

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
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left", padx=(6, 18))
        self.auto_speak_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(status_frame, text="Automatically speak Ollama responses", variable=self.auto_speak_var).pack(side="left")

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

    def start_listening(self):
        if self.is_listening:
            return

        mic_index = self._selected_microphone_index()
        if mic_index is None:
            messagebox.showwarning("Microphone", "Please select a valid input-capable microphone.")
            return

        # Snapshot every Tkinter-owned setting before starting worker threads.
        # Background audio/STT workers must not read tk.StringVar/IntVar.
        self._listener_config = {
            "mic_index": mic_index,
            "stt_backend": self.stt_backend_var.get().strip() or "Auto",
            "stt_model": self.stt_model_var.get().strip() or "base.en",
            "stt_threads": max(1, int(self.stt_threads_var.get() or 4)),
        }

        # Validate the exact PortAudio input device that the persistent
        # listener will use. Do not open a second SpeechRecognition microphone
        # object here; that creates a different capture path than the live loop.
        pa = None
        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(mic_index)
            if int(info.get("maxInputChannels", 0) or 0) < 1:
                raise RuntimeError("Selected PortAudio device has no input channels.")
        except Exception as exc:
            messagebox.showerror("Microphone Error", str(exc))
            return
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

        # Recover from a stale busy flag left by an interrupted/failed TTS job.
        if self.tts_queue.empty() and not self.is_speaking:
            self.audio_busy.clear()

        # Live capture no longer depends on SpeechRecognition.listen().
        # We use raw PyAudio RMS voice activity detection instead.
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 3.0
        self.recognizer.non_speaking_duration = 1.0

        self.is_listening = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.mic_combo.config(state="disabled")
        self.status_var.set(f"Starting stable adaptive listener on device {mic_index}...")
        self.avatar_status_var.set("Listening")
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False
        self.avatar_status_var.set("Idle")
        self.status_var.set("Stopping listener...")

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
    def _robust_noise_floor(levels):
        """
        Estimate ambient noise without letting a brief voice/transient dominate.

        Use the quieter 60% of calibration samples, then take their median.
        This is much more stable than treating the whole calibration window as
        ambient noise.
        """
        values = sorted(float(v) for v in levels if v >= 0.0)
        if not values:
            return 35.0

        quiet_count = max(1, int(len(values) * 0.60))
        quiet = values[:quiet_count]
        return max(20.0, quiet[len(quiet) // 2])

    @staticmethod
    def _adaptive_trigger(noise_floor):
        """
        Convert the current ambient floor into speech-on and speech-off levels.

        Hysteresis (different ON/OFF thresholds) prevents rapid toggling when
        speech sits near the threshold.
        """
        trigger = max(75.0, min(1100.0, noise_floor * 1.85 + 38.0))
        release = max(55.0, min(trigger * 0.62, noise_floor * 1.30 + 22.0))
        return trigger, release

    def _listen_loop(self):
        """
        Persistent PortAudio listener with adaptive RMS voice activity detection.

        Key stability improvements over v7.2:
          * opens the microphone only once
          * calibrates only once at listener startup
          * continuously adapts the ambient noise floor only while idle
          * freezes noise-floor learning around possible speech
          * uses hysteresis for speech start/end
          * uses pre-roll so first syllables are not clipped
          * drains audio while Ollama/TTS is active, preventing stale speaker
            audio from being processed after the assistant finishes
          * uses a short post-TTS cooldown before accepting new speech
        """
        config = dict(self._listener_config)
        mic_index = int(config.get("mic_index"))

        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(mic_index)

            sample_rate = int(float(info.get("defaultSampleRate", 48000) or 48000))
            sample_rate = max(8000, sample_rate)
            chunk = 1024
            chunk_seconds = chunk / float(sample_rate)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=mic_index,
                frames_per_buffer=chunk,
                start=True,
            )

            # ---------- One-time robust calibration ----------
            self.ui_queue.put((
                "status",
                f"Calibrating microphone {mic_index}; remain quiet briefly..."
            ))

            calibration_levels = []
            calibration_chunks = max(12, int(1.2 / chunk_seconds))

            for _ in range(calibration_chunks):
                if not self.is_listening or self.shutdown_requested:
                    return
                raw = stream.read(chunk, exception_on_overflow=False)
                calibration_levels.append(self._pcm16_rms(raw))

            noise_floor = self._robust_noise_floor(calibration_levels)
            trigger, release = self._adaptive_trigger(noise_floor)

            # Voice-state constants.
            pre_roll = deque(maxlen=max(2, int(0.65 / chunk_seconds)))
            speech_confirm_chunks = max(2, int(0.08 / chunk_seconds))
            likely_speech_chunks = max(2, int(0.14 / chunk_seconds))
            silence_needed = max(1, int(3.0 / chunk_seconds))
            max_record_chunks = max(1, int(120.0 / chunk_seconds))
            trailing_keep = max(1, int(0.35 / chunk_seconds))

            # Adaptive floor controls.
            noise_alpha = min(0.08, max(0.015, chunk_seconds * 0.9))
            status_interval = 0.20

            was_busy = False
            cooldown_until = 0.0
            last_status = 0.0

            while self.is_listening and not self.shutdown_requested:
                raw = stream.read(chunk, exception_on_overflow=False)
                level = self._pcm16_rms(raw)
                now = time.monotonic()

                busy = self.audio_busy.is_set() or self.is_processing

                # While the assistant is processing/speaking, continuously drain
                # the input stream but do not learn speaker output as "ambient".
                if busy:
                    was_busy = True
                    pre_roll.clear()
                    if now - last_status >= 0.5:
                        self.ui_queue.put((
                            "status",
                            "Microphone paused while assistant is processing/speaking..."
                        ))
                        last_status = now
                    continue

                # After assistant audio stops, discard a brief tail of room echo.
                if was_busy:
                    was_busy = False
                    cooldown_until = now + 0.55
                    pre_roll.clear()

                if now < cooldown_until:
                    continue

                # Recompute thresholds from the current learned ambient floor.
                trigger, release = self._adaptive_trigger(noise_floor)

                # Learn ambient only when safely below the trigger. This lets the
                # app follow fan/HVAC changes without learning the user's voice.
                if level < trigger * 0.68:
                    noise_floor = (
                        (1.0 - noise_alpha) * noise_floor
                        + noise_alpha * max(15.0, level)
                    )

                pre_roll.append(raw)

                # Two trigger paths:
                # 1) clear/strong speech
                # 2) sustained moderate speech relative to ambient
                strong_threshold = trigger
                moderate_threshold = max(
                    65.0,
                    noise_floor * 1.42 + 24.0
                )

                strong_count = 0
                moderate_count = 0
                candidate_frames = []
                candidate_peak = 0.0

                # The current frame may already be the beginning of speech.
                if level >= strong_threshold:
                    strong_count = 1
                if level >= moderate_threshold:
                    moderate_count = 1

                if strong_count or moderate_count:
                    candidate_frames = list(pre_roll)
                    candidate_peak = level

                    # Confirm over a small multi-frame window. This catches soft
                    # consonants while rejecting one-frame clicks/transients.
                    confirmation_limit = max(
                        speech_confirm_chunks + 2,
                        likely_speech_chunks + 2
                    )

                    for _ in range(confirmation_limit):
                        nxt = stream.read(chunk, exception_on_overflow=False)
                        nxt_level = self._pcm16_rms(nxt)
                        candidate_frames.append(nxt)
                        candidate_peak = max(candidate_peak, nxt_level)

                        if nxt_level >= strong_threshold:
                            strong_count += 1
                        else:
                            strong_count = max(0, strong_count - 1)

                        if nxt_level >= moderate_threshold:
                            moderate_count += 1
                        else:
                            moderate_count = max(0, moderate_count - 1)

                        if (
                            strong_count >= speech_confirm_chunks
                            or moderate_count >= likely_speech_chunks
                        ):
                            break

                    speech_confirmed = (
                        strong_count >= speech_confirm_chunks
                        or moderate_count >= likely_speech_chunks
                    )

                    if speech_confirmed:
                        self.ui_queue.put((
                            "status",
                            f"Speech detected — recording "
                            f"(level {candidate_peak:.0f}, trigger {trigger:.0f})..."
                        ))

                        frames = candidate_frames
                        peak_rms = candidate_peak
                        silence_chunks = 0
                        total_chunks = len(frames)
                        last_record_status = 0.0

                        while (
                            self.is_listening
                            and not self.shutdown_requested
                            and total_chunks < max_record_chunks
                        ):
                            # Assistant output should not normally begin during
                            # user capture, but abort safely if it does.
                            if self.audio_busy.is_set() or self.is_processing:
                                frames = []
                                break

                            data = stream.read(chunk, exception_on_overflow=False)
                            voice_level = self._pcm16_rms(data)
                            frames.append(data)
                            total_chunks += 1
                            peak_rms = max(peak_rms, voice_level)

                            # Hysteresis: ending speech requires sustained audio
                            # below the lower release threshold.
                            if voice_level < release:
                                silence_chunks += 1
                            else:
                                silence_chunks = 0

                            t = time.monotonic()
                            if t - last_record_status >= 0.25:
                                elapsed = total_chunks * chunk_seconds
                                self.ui_queue.put((
                                    "status",
                                    f"Recording voice: {elapsed:.1f}s "
                                    f"— level {voice_level:.0f}"
                                ))
                                last_record_status = t

                            if silence_chunks >= silence_needed:
                                break

                        if not frames:
                            pre_roll.clear()
                            continue

                        # Keep a little natural trailing silence but remove most
                        # of the 3-second endpoint delay from the Whisper input.
                        if silence_chunks > trailing_keep and len(frames) > silence_chunks:
                            trim = silence_chunks - trailing_keep
                            if trim > 0:
                                frames = frames[:-trim]

                        raw_pcm = b"".join(frames)
                        pre_roll.clear()

                        if not raw_pcm:
                            continue

                        audio = sr.AudioData(raw_pcm, sample_rate, 2)

                        self.ui_queue.put((
                            "status",
                            f"Voice captured (peak RMS {peak_rms:.0f}, "
                            f"ambient {noise_floor:.0f}). Transcribing locally..."
                        ))

                        text, backend_used = self._transcribe_local_audio(
                            audio,
                            config
                        )
                        text = (text or "").strip()

                        if not text:
                            self.ui_queue.put((
                                "status",
                                "Audio captured, but local Whisper returned no "
                                "text. Listening again..."
                            ))
                            continue

                        self._last_stt_backend = backend_used
                        stamp = time.strftime("%H:%M:%S")
                        self.ui_queue.put((
                            "heard",
                            f"[{stamp}] You ({backend_used}): {text}\n"
                        ))

                        self.is_processing = True
                        threading.Thread(
                            target=self._stream_ollama_response,
                            args=(text,),
                            daemon=True,
                        ).start()
                        continue

                    # Candidate was noise, not speech. Keep only the latest part
                    # as pre-roll and continue listening.
                    pre_roll.clear()
                    for item in candidate_frames[-pre_roll.maxlen:]:
                        pre_roll.append(item)

                if now - last_status >= status_interval:
                    self.ui_queue.put((
                        "status",
                        f"Listening mic {mic_index} — level {level:.0f} | "
                        f"ambient {noise_floor:.0f} | trigger {trigger:.0f}"
                    ))
                    last_status = now

        except OSError as exc:
            self.ui_queue.put((
                "error",
                "Persistent PortAudio capture failed.\n\n"
                f"Microphone device: {mic_index}\n{exc}"
            ))
        except Exception as exc:
            self.ui_queue.put((
                "error",
                f"Microphone/listening error:\n{exc}"
            ))
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

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

        # Auto: prefer Vulkan only when all required pieces exist and Vulkan
        # itself responds successfully; otherwise use Faster-Whisper CPU.
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
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
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

    def _api_url(self, endpoint):
        return f"{self.ollama_url_var.get().strip().rstrip('/')}/api/{endpoint.lstrip('/')}"

    def load_ollama_models(self):
        threading.Thread(target=self._load_ollama_models_worker, daemon=True).start()

    def _load_ollama_models_worker(self):
        try:
            self.ui_queue.put(("status", "Loading Ollama models..."))
            response = requests.get(self._api_url("tags"), timeout=10)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
            self.ui_queue.put(("models", models))
        except Exception as exc:
            self.ui_queue.put(("error", f"Could not retrieve Ollama models.\n\n{exc}\n\nConfirm that Ollama is running and the URL is correct."))

    def test_ollama(self):
        threading.Thread(target=self._test_ollama_worker, daemon=True).start()

    def _test_ollama_worker(self):
        try:
            self.ui_queue.put(("status", "Testing Ollama connection..."))
            response = requests.get(self._api_url("tags"), timeout=10)
            response.raise_for_status()
            self.ui_queue.put(("info", "Ollama connection successful."))
            self.ui_queue.put(("status", "Ready"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Ollama connection failed.\n\n{exc}\n\nDefault local address: http://localhost:11434"))

    def _stream_ollama_response(self, user_text):
        assistant_text = ""
        try:
            model = self.model_var.get().strip()
            if not model:
                raise ValueError("Please enter or select an Ollama model.")

            system = self.system_prompt.get("1.0", "end-1c").strip()
            request_messages = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(self.messages)
            request_messages.append({"role": "user", "content": user_text})

            payload = {"model": model, "messages": request_messages, "stream": True}
            self.ui_queue.put(("response_start", f"\nYou: {user_text}\nAssistant: "))
            self.ui_queue.put(("status", f"Requesting response from {model}..."))

            # Spoken acknowledgement while Ollama begins processing.
            # audio_busy is set before playback starts, so the microphone cannot
            # capture the assistant's own announcement.
            self._queue_tts(
                "I'm researching your request, please stand by.",
                label="Researching"
            )

            with requests.post(self._api_url("chat"), json=payload, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self.shutdown_requested:
                        return
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    token = (chunk.get("message") or {}).get("content", "")
                    if token:
                        assistant_text += token
                        self.ui_queue.put(("response_chunk", token))
                    if chunk.get("done", False):
                        break

            assistant_text = assistant_text.strip()
            if assistant_text:
                self.messages.append({"role": "user", "content": user_text})
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.ui_queue.put(("response_chunk", "\n"))
                if self.auto_speak_var.get():
                    self.ui_queue.put(("status", "Preparing spoken response..."))
                    self._queue_tts(assistant_text, label="Speaking response")
                else:
                    self.ui_queue.put(("status", "Listening..." if self.is_listening else "Ready"))
            else:
                self.ui_queue.put(("status", "Ollama returned an empty response."))
        except requests.exceptions.ConnectionError:
            self.ui_queue.put(("error", "Cannot connect to Ollama.\n\nMake sure Ollama is running and verify the server URL."))
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.text
            except Exception:
                pass
            self.ui_queue.put(("error", f"Ollama HTTP error:\n{exc}\n\n{detail}"))
        except Exception as exc:
            self.ui_queue.put(("error", f"Ollama request error:\n{exc}"))
        finally:
            self.is_processing = False

    def _queue_tts(self, text, label="Speaking"):
        """Queue speech and mark audio busy before the worker can run."""
        text = (text or "").strip()
        if not text:
            return
        self.audio_busy.set()
        self.tts_queue.put((text, label))

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

            if isinstance(text, tuple):
                text, speech_label = text
            else:
                speech_label = "Speaking"

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

    def _process_ui_queue(self):
        try:
            while True:
                action, value = self.ui_queue.get_nowait()
                if action == "status":
                    self.status_var.set(value)
                elif action == "heard":
                    self.heard_text.insert("end", value)
                    self.heard_text.see("end")
                elif action in ("response_start", "response_chunk"):
                    self.response_text.insert("end", value)
                    self.response_text.see("end")
                elif action == "avatar_status":
                    self.avatar_status_var.set(value)
                elif action == "models":
                    self.model_combo["values"] = value
                    if value and self.model_var.get() not in value:
                        self.model_var.set(value[0])
                    self.status_var.set(f"Loaded {len(value)} Ollama model(s).")
                elif action == "listener_stopped":
                    self.is_listening = False
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.mic_combo.config(state="readonly")
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
