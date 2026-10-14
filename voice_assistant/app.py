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

"""Application class: composes the mixins and owns shared state."""

import os
import platform
import queue
import random
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import speech_recognition as sr

from .constants import APP_BUILD, PROJECT_ROOT
from .config import ConfigMixin
from .text_utils import TextUtilsMixin
from .listener import ListenerMixin
from .stt import SpeechToTextMixin
from .ollama_client import OllamaMixin
from .tts import TextToSpeechMixin
from .avatar import AvatarMixin
from .ui import UIMixin
from .whisper_server import WhisperServer


class OllamaVoiceAssistant(
    ConfigMixin,
    TextUtilsMixin,
    ListenerMixin,
    SpeechToTextMixin,
    OllamaMixin,
    TextToSpeechMixin,
    AvatarMixin,
    UIMixin,
):
    def __init__(self, root):
        self.root = root
        self.root.title(f"MyVoiceAssistant {APP_BUILD}")
        self.root.geometry("1250x850")
        self.root.minsize(1000, 700)

        # External JSON configuration. By default the app looks beside this
        # Python file for ollama_voice_assistant_config.json. Override with:
        #   export OLLAMA_VOICE_CONFIG=/path/to/config.json
        self.config_path = Path(
            os.environ.get(
                "OLLAMA_VOICE_CONFIG",
                str(PROJECT_ROOT / "ollama_voice_assistant_config.json")
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
        self._whisper_server = WhisperServer()
        self._whisper_server_error = None
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.shutdown_requested = False
        self.audio_busy = threading.Event()
        self.ui_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.messages = []
        # Conversation history is appended from the Ollama worker thread and
        # cleared from the UI thread. The generation counter lets a reply
        # that was in flight during "Clear Conversation" be discarded.
        self._messages_lock = threading.Lock()
        self._conversation_generation = 0

        # Tracks a single active Ollama wait/reminder cycle.
        self._response_wait_lock = threading.Lock()
        self._response_wait_event = None
        self._active_request_cancel = None

        # Live UI settings mirrored on the UI thread so worker threads never
        # read Tk variables directly.
        self._runtime_settings_lock = threading.Lock()
        self._runtime_settings = {}

        # Interruptible playback. Bumping _speech_generation invalidates
        # every TTS item queued before the bump.
        self._playback_lock = threading.Lock()
        self._playback_proc = None
        self._speech_generation = 0

        # Cached Vulkan probe; vulkaninfo can take seconds to run.
        self._vulkan_cache = None
        self._vulkan_cache_lock = threading.Lock()

        # Error dialog de-duplication (prevents modal dialog storms).
        self._last_error_text = None
        self._last_error_time = 0.0
        self._piper_load_error = None

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
        self._refresh_runtime_settings()
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
                    "The application can start, but these components need attention:\n\n"
                    + "\n".join(f"• {item}" for item in problems)
                    + "\n\nUse the included install_ubuntu_26_04_v7.sh script."
                )
            )


def main():
    root = tk.Tk()
    OllamaVoiceAssistant(root)
    root.mainloop()
