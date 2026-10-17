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

"""Persistent whisper.cpp server process.

Launching whisper-cli for every utterance re-initialises Vulkan and reloads
the GGML model each time (~1-2 s on the Mac Pro FirePro GPUs). whisper-server
keeps the model resident on the GPU, so each request only pays for inference.
"""

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests


def find_whisper_server(whisper_cli_path):
    """Return the whisper-server binary built alongside whisper-cli, if any."""
    if whisper_cli_path:
        sibling = Path(whisper_cli_path).with_name("whisper-server")
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)
    return shutil.which("whisper-server")


# Resolved once in the parent: loading libraries inside a child forked from a
# multithreaded process is unreliable, so preexec_fn must only make the call.
try:
    import ctypes
    import signal

    _prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    _PR_SET_PDEATHSIG = 1
    # SIGKILL, not SIGTERM: whisper-server's SIGTERM handler calls svr->stop()
    # inside the signal handler and can deadlock. It holds no state to save.
    _SIGKILL = int(signal.SIGKILL)
except Exception:
    _prctl = None


def die_with_parent():
    """Linux: kill the server if the app crashes or is killed."""
    if _prctl is not None:
        _prctl(_PR_SET_PDEATHSIG, _SIGKILL)


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class WhisperServer:
    """Start, health-check, query and stop one whisper-server process."""

    def __init__(self, startup_timeout=90.0, request_timeout=180.0):
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._proc = None
        self._key = None
        self._base_url = None
        self._log_path = None
        self._log_handle = None

    @property
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def describe(self):
        if not self.running:
            return "not running"
        binary, model, threads, language = self._key
        return f"{self._base_url} model={Path(model).name} threads={threads} lang={language}"

    def ensure(self, binary, model, threads, language):
        """Start the server, or restart it if the settings changed or it died."""
        key = (binary, model, int(threads), language)
        with self._lock:
            if self.running and self._key == key:
                return
            self._stop_locked()
            self._start_locked(key)

    def _start_locked(self, key):
        binary, model, threads, language = key
        port = _free_local_port()
        fd, self._log_path = tempfile.mkstemp(prefix="whisper_server_", suffix=".log")
        self._log_handle = os.fdopen(fd, "w")
        # Output goes to a log file rather than a pipe so a chatty server can
        # never block on a full pipe buffer.
        self._proc = subprocess.Popen(
            [
                binary,
                "-m", model,
                "--host", "127.0.0.1",
                "--port", str(port),
                "-t", str(threads),
                "-l", language,
                "-nt",
            ],
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=die_with_parent,
        )
        self._key = key
        self._base_url = f"http://127.0.0.1:{port}"

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                break
            try:
                # /health returns 503 while the model is still loading.
                if requests.get(f"{self._base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)

        log_tail = self._read_log_tail()
        self._stop_locked()
        raise RuntimeError(
            "whisper-server did not become ready.\n\n" + (log_tail or "No server output.")
        )

    def transcribe(self, wav_path):
        with self._lock:
            if not self.running:
                raise RuntimeError("whisper-server is not running.")
            url = f"{self._base_url}/inference"
        with open(wav_path, "rb") as handle:
            response = requests.post(
                url,
                files={"file": (os.path.basename(wav_path), handle, "audio/wav")},
                data={"response_format": "json", "temperature": "0.0"},
                timeout=self.request_timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"whisper-server error: {payload['error']}")
        return str(payload.get("text", "")).strip()

    def _read_log_tail(self, limit=4000):
        try:
            if self._log_handle:
                self._log_handle.flush()
            return Path(self._log_path).read_text(errors="replace")[-limit:].strip()
        except Exception:
            return ""

    def stop(self):
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Graceful shutdown can deadlock in the server's signal handler.
                proc.kill()
                proc.wait(timeout=5)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        if self._log_path:
            try:
                os.unlink(self._log_path)
            except OSError:
                pass
            self._log_path = None
        self._key = None
        self._base_url = None
