# Copyright 2026 Christopher Hinds, Stratum Technologies LLC
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

"""Microphone selection, voice-trigger state machine and the VAD listen loop."""

import re
import threading
import time
from collections import deque
from tkinter import messagebox

import pyaudio
import speech_recognition as sr


class ListenerMixin:
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
            rms = self._pcm16_rms(raw)

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

    def _snapshot_listener_config(self, mic_index):
        return {
            "mic_index": mic_index,
            "stt_backend": self.stt_backend_var.get().strip() or "Auto",
            "stt_model": self.stt_model_var.get().strip() or "base.en",
            "stt_threads": self._safe_stt_threads(),
            "ollama_url": self._sanitize_ollama_base_url(self.ollama_url_var.get()),
            "ollama_model": self.model_var.get().strip(),
            "system_prompt": self.system_prompt.get("1.0", "end-1c").strip(),
            "auto_speak": bool(self.auto_speak_var.get()),
        }

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

                    # A transcription failure must not kill the whole listener
                    # thread; report it and keep listening.
                    try:
                        text, backend_used = self._transcribe_local_audio(audio, config)
                    except Exception as stt_exc:
                        self._trace_exception("STT", stt_exc, config=config)
                        self.ui_queue.put((
                            "error",
                            f"Local speech-to-text failed:\n{stt_exc}"
                        ))
                        continue

                    raw_text = (text or "").strip()
                    text = self._filter_stt_noise(raw_text)
                    if raw_text and not text:
                        self._trace_print(
                            f"[LISTENER TRACE] discarded STT noise: {raw_text!r}"
                        )

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
                        # Abandon any in-flight answer before acknowledging.
                        self._cancel_response_and_speech()
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
                        self._trace_print(
                            f"[LISTENER TRACE] START trigger accepted; "
                            f"listener_mode=ACTIVE text={text!r}"
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

                        # Never start a second concurrent Ollama request.
                        if self.is_processing:
                            self.ui_queue.put((
                                "status",
                                "Ollama is still processing the previous request."
                            ))
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
