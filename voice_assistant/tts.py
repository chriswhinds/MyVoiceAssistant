"""Piper / espeak-ng speech synthesis, interruptible playback and the TTS queue."""

import math
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import time
import wave
from array import array
from pathlib import Path
from tkinter import messagebox

import speech_recognition as sr

try:
    import numpy as np
except Exception:  # numpy ships with faster-whisper; pure-Python fallback otherwise
    np = None


class TextToSpeechMixin:
    def _queue_tts(self, text, label="Speaking", cancel_event=None):
        """Queue speech, optionally skipping it if cancel_event becomes set."""
        text = self._clean_text_for_speech(text)
        if not text:
            return
        self.audio_busy.set()
        self.tts_queue.put((text, label, cancel_event, self._speech_generation))

    def _cancel_response_and_speech(self):
        """Abort the in-flight Ollama reply and silence queued/playing speech."""
        with self._response_wait_lock:
            if self._active_request_cancel is not None:
                self._active_request_cancel.set()
            if self._response_wait_event is not None:
                self._response_wait_event.set()  # stops standby reminders
        self._speech_generation += 1
        with self._playback_lock:
            proc = self._playback_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop_speaking(self):
        self._cancel_response_and_speech()
        self.status_var.set("Speech stopped.")

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
            generation = self._speech_generation
            if isinstance(text, tuple):
                if len(text) == 4:
                    text, speech_label, cancel_event, generation = text
                elif len(text) == 3:
                    text, speech_label, cancel_event = text
                else:
                    text, speech_label = text
            else:
                speech_label = "Speaking"

            # A standby reminder can become obsolete while waiting behind
            # another TTS job, and "Stop Speaking" invalidates everything
            # queued before it. Skip rather than delaying newer speech.
            if (
                (cancel_event is not None and cancel_event.is_set())
                or generation != self._speech_generation
            ):
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
                self._piper_load_error = None
            except Exception as exc:
                self.piper_voice = None
                # Report once instead of silently degrading to espeak.
                if self._piper_load_error is None:
                    self._piper_load_error = exc
                    self._trace_exception("PIPER_LOAD", exc, voice=self.piper_voice_path)
                    self.ui_queue.put((
                        "status",
                        f"Piper voice failed to load ({exc}); using espeak-ng fallback."
                    ))

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
        except Exception as exc:
            self._trace_exception("PIPER_SYNTH", exc, text=text[:200])
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

                if np is not None:
                    # Vectorised path: ~100x faster than the per-sample loop
                    # on long replies, so playback starts sooner.
                    pcm = np.frombuffer(
                        wav_file.readframes(wav_file.getnframes()), dtype="<i2"
                    ).astype(np.float64)
                    if channels > 1:
                        pcm = pcm[: len(pcm) - len(pcm) % channels]
                    bucket = frames_per_bucket * channels
                    if pcm.size == 0:
                        return []
                    full = (pcm.size // bucket) * bucket
                    rms = np.sqrt(np.mean(pcm[:full].reshape(-1, bucket) ** 2, axis=1))
                    if full < pcm.size:  # final partial bucket, own length
                        tail = pcm[full:]
                        rms = np.append(rms, np.sqrt(np.mean(tail * tail)))
                    normalized = np.minimum(1.0, rms / 9000.0)
                    normalized = np.where(
                        normalized < 0.025, 0.0, np.minimum(1.0, normalized * 1.65)
                    )
                    return normalized.tolist()

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
        if self.piper_voice is None:
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
                        "Fallback speech synthesis failed:\n"
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

            try:
                sync_offset_ms = float(
                    self.app_config.get("speech", {}).get("avatar_sync_offset_ms", 80)
                )
            except (TypeError, ValueError):
                sync_offset_ms = 80.0

            # Popen (not run) so "Stop Speaking" / stop trigger can kill it.
            playback = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.avatar_envelope_started = (
                time.monotonic() + max(0.0, sync_offset_ms) / 1000.0
            )
            with self._playback_lock:
                self._playback_proc = playback
            try:
                _, stderr = playback.communicate()
            finally:
                with self._playback_lock:
                    self._playback_proc = None

            # Negative return codes mean we terminated it deliberately.
            if playback.returncode > 0:
                raise RuntimeError(
                    f"{self.audio_player_name} playback failed:\n"
                    + ((stderr or "").strip() or "unknown playback error")
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

        mic_lines = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(microphones[:20])
        )
        if len(microphones) > 20:
            mic_lines += f"\n  ... {len(microphones) - 20} more"

        report = (
            "Ubuntu / Linux Audio Diagnostics\n"
            "--------------------------------\n"
            f"Platform: {platform.platform()}\n"
            f"Architecture: {platform.machine()}\n"
            f"TTS synthesizer: {self.tts_synth or 'NOT FOUND'}\n"
            f"Playback backend: {self.audio_player_name or 'NOT FOUND'}\n"
            f"Playback command: {self.audio_player_cmd or 'NOT FOUND'}\n"
            f"Ollama URL: {self.ollama_url_var.get().strip()}\n\n"
            "PortAudio microphone devices:\n"
            f"{mic_lines or '  None'}"
        )

        messagebox.showinfo("Audio Diagnostics", report)

    def test_speaker(self):
        self._queue_tts("Hello. I am your local voice assistant. My neural female voice is working correctly.", label="Testing female voice")

    def speak_last_response(self):
        with self._messages_lock:
            history = list(self.messages)
        for message in reversed(history):
            if message.get("role") == "assistant" and message.get("content", "").strip():
                self._queue_tts(message["content"].strip())
                return
        messagebox.showinfo("Speak Response", "There is no Ollama response to speak yet.")
