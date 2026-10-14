"""Local speech-to-text: whisper.cpp (Vulkan) and Faster-Whisper (CPU)."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from tkinter import messagebox

from .constants import PROJECT_ROOT
from .whisper_server import find_whisper_server


class SpeechToTextMixin:
    def _prewarm_stt(self, config):
        """Load the persistent STT backend before the user starts speaking."""
        try:
            backend = self._selected_stt_backend(config)
            if backend == "vulkan-server":
                self.ui_queue.put(("status", "Starting whisper.cpp server on the GPU..."))
                try:
                    self._ensure_whisper_server(config)
                except Exception as exc:
                    self._mark_whisper_server_failed(exc)
                    if config.get("stt_backend", "Auto") == "Auto" and self._python_module_available("faster_whisper"):
                        self._load_faster_whisper_cpu(config)
            elif backend == "faster-whisper-cpu":
                self._load_faster_whisper_cpu(config)
        except Exception as exc:
            self.ui_queue.put((
                "status",
                f"STT pre-warm warning: {exc}"
            ))

    def _whisper_server_binary(self):
        if self._whisper_server_error is not None:
            return None  # failed this session; STT Diagnostics re-enables it
        return find_whisper_server(self._whisper_cpp_cli)

    def _mark_whisper_server_failed(self, exc):
        self._whisper_server_error = str(exc)
        self._trace_exception("WHISPER_SERVER", exc)
        self.ui_queue.put((
            "status",
            "whisper-server unavailable; falling back to another STT backend."
        ))

    def _whisper_cpp_params(self, config=None):
        """Return (threads, language) for whisper.cpp based on the model file found."""
        threads = max(1, int((config or {}).get("stt_threads", 4))) if config is not None else self._safe_stt_threads()

        # English-only *.en models should use English. Multilingual models can
        # autodetect.
        selected_model = ((config or {}).get("stt_model", "base.en") if config is not None else (self.stt_model_var.get().strip() or "base.en")).lower()

        # Base the language flag on the GGML file actually found, since the
        # detector falls back to ggml-base.en.bin when the chosen model is
        # missing.
        actual_model = Path(self._whisper_cpp_model).name.lower()
        if actual_model != f"ggml-{selected_model}.bin":
            self._trace_print(
                f"[STT TRACE] requested whisper.cpp model {selected_model!r} "
                f"not found; using {actual_model!r}"
            )
        language = "en" if actual_model.endswith(".en.bin") else "auto"
        return threads, language

    def _ensure_whisper_server(self, config=None):
        binary = self._whisper_server_binary()
        if not binary or not self._whisper_cpp_model:
            raise RuntimeError("whisper-server binary or GGML model not found.")
        threads, language = self._whisper_cpp_params(config)
        self._whisper_server.ensure(binary, self._whisper_cpp_model, threads, language)

    def _transcribe_whisper_server(self, wav_path, config=None):
        self._ensure_whisper_server(config)
        return self._whisper_server.transcribe(wav_path)

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
            PROJECT_ROOT / "third_party" / "whisper.cpp",
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

    def _vulkan_summary(self, refresh=False):
        """Cached Vulkan probe; pass refresh=True to re-run vulkaninfo."""
        with self._vulkan_cache_lock:
            if self._vulkan_cache is not None and not refresh:
                return self._vulkan_cache
            result = self._probe_vulkan()
            self._vulkan_cache = result
            return result

    def _probe_vulkan(self):
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
            return "vulkan-server" if self._whisper_server_binary() else "vulkan"

        if requested == "Faster-Whisper CPU":
            if not self._python_module_available("faster_whisper"):
                raise RuntimeError(
                    "faster-whisper is not installed in this Python environment."
                )
            return "faster-whisper-cpu"

        # Auto prefers persistent backends: whisper-server keeps the model
        # resident on the GPU (~0.8 s for 11 s of audio on the FirePro D-series
        # versus ~2.9 s for Faster-Whisper on the Xeon). The one-shot
        # whisper-cli reloads the model per request, so low-latency mode only
        # uses it as a last resort.
        low_latency = bool(
            self.app_config.get("speech", {}).get("low_latency_mode", True)
        )
        vulkan_ready = bool(self._whisper_cpp_cli and self._whisper_cpp_model) and self._vulkan_summary()[0]
        faster_ready = self._python_module_available("faster_whisper")

        if vulkan_ready and self._whisper_server_binary():
            return "vulkan-server"

        if vulkan_ready and not low_latency:
            return "vulkan"

        if faster_ready:
            return "faster-whisper-cpu"

        if vulkan_ready:
            return "vulkan"

        raise RuntimeError(
            "No fully local STT backend is available. Run install_ubuntu_26_04_v7.sh."
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
                if backend == "vulkan-server":
                    try:
                        text = self._transcribe_whisper_server(wav_path, config)
                        return text, "whisper-server/Vulkan"
                    except Exception as exc:
                        self._mark_whisper_server_failed(exc)
                        self._whisper_server.stop()
                        requested = (config or {}).get("stt_backend", "Auto")
                        if requested == "Auto" and self._python_module_available("faster_whisper"):
                            backend = "faster-whisper-cpu"
                        else:
                            backend = "vulkan"

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

        threads, language = self._whisper_cpp_params(config)

        # The no-timestamps/no-prints flags produce clean stdout.
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

        threads = max(1, int((config or {}).get("stt_threads", 4))) if config is not None else self._safe_stt_threads()

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
        # Diagnostics re-probe everything and give a failed server another try.
        previous_server_error = self._whisper_server_error
        self._whisper_server_error = None
        self._detect_local_stt_stack()
        vulkan_ok, vulkan_text = self._vulkan_summary(refresh=True)
        server_binary = find_whisper_server(self._whisper_cpp_cli)

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
            f"whisper-server: {server_binary or 'NOT FOUND'}\n"
            f"whisper-server status: {self._whisper_server.describe()}\n"
            + (f"whisper-server last error: {previous_server_error[:300]}\n" if previous_server_error else "")
            +
            f"Vulkan usable: {'YES' if vulkan_ok else 'NO'}\n\n"
            "Vulkan summary:\n"
            + ("\n".join(summary_lines[:20]) if summary_lines else vulkan_text[:2000] or "No Vulkan details")
        )

        messagebox.showinfo("Local STT Diagnostics", report)
