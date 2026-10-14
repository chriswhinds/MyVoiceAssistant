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

"""Tk window layout, conversation pane actions and the UI event queue."""

import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .constants import APP_BUILD


class UIMixin:
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
        ttk.Button(controls, text="Stop Speaking", command=self.stop_speaking).pack(side="left", padx=(0, 8))
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

        footer = ttk.Label(self.root, text="Fully local pipeline: raw PyAudio VAD + 2-minute microphone capture -> whisper-server (Vulkan GPU) or Faster-Whisper CPU -> streaming Ollama -> Piper neural female TTS -> PipeWire/PulseAudio/ALSA -> speakers + amplitude-driven female avatar.", padding=(12, 0, 12, 10), wraplength=1100)
        footer.pack(fill="x")

    def clear_conversation(self):
        with self._messages_lock:
            self.messages.clear()
            self._conversation_generation += 1
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

    def _show_error(self, value):
        """Show an error dialog, but not the same one repeatedly."""
        now = time.monotonic()
        self.status_var.set("Error")
        if value == self._last_error_text and now - self._last_error_time < 30.0:
            summary = " ".join(str(value or "Error").split())[:140]
            self.status_var.set(f"Error (repeated): {summary}")
            return
        self._last_error_text = value
        self._last_error_time = now
        messagebox.showerror("Error", value)

    def _process_ui_queue(self):
        self._refresh_runtime_settings()
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
                    self._show_error(value)
        except queue.Empty:
            pass
        if not self.shutdown_requested:
            self.root.after(80, self._process_ui_queue)

    def _on_close(self):
        self.shutdown_requested = True
        self.command_listening_event.clear()
        self.is_listening = False
        self.audio_busy.clear()
        # Stop any reply that is still streaming or playing.
        self._cancel_response_and_speech()
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
        # Release the GPU-resident whisper-server process.
        self._whisper_server.stop()
        self.root.destroy()
