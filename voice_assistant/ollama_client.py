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

"""Ollama model selection, connection tests and streaming chat."""

import json
import threading
from urllib.parse import urlsplit

import requests

from .constants import APP_BUILD


class OllamaMixin:
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
        # Read the Tk variable here, on the UI thread.
        raw_url = self.ollama_url_var.get()
        threading.Thread(
            target=self._load_ollama_models_worker, args=(raw_url,), daemon=True
        ).start()

    def _load_ollama_models_worker(self, raw_url):
        try:
            self.ui_queue.put(("status", "Loading Ollama models..."))
            base_url = self._sanitize_ollama_base_url(raw_url)

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
        raw_url = self.ollama_url_var.get()
        threading.Thread(
            target=self._test_ollama_worker, args=(raw_url,), daemon=True
        ).start()

    def _test_ollama_worker(self, raw_url):
        try:
            self.ui_queue.put(("status", "Testing Ollama connection..."))
            base_url = self._sanitize_ollama_base_url(raw_url)

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
        cancel_event = threading.Event()
        speech_buffer = ""
        spoke_streamed = False

        with self._response_wait_lock:
            # Cancel any stale reminder from an older request.
            if self._response_wait_event is not None:
                self._response_wait_event.set()
            self._response_wait_event = first_token
            self._active_request_cancel = cancel_event

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

            # Live UI values (mirrored on the UI thread) win over the
            # listener snapshot, which may be minutes old.
            live = self._get_runtime_settings()
            system = str(
                live.get("system_prompt", request_config.get("system_prompt", ""))
            ).strip()
            base_url = self._sanitize_ollama_base_url(
                live.get(
                    "ollama_url",
                    request_config.get("ollama_url", self._configured_ollama_url()),
                )
            )
            auto_speak = bool(
                live.get("auto_speak", request_config.get("auto_speak", True))
            )
            ollama_cfg = self.app_config.get("ollama", {})
            speech_cfg = self.app_config.get("speech", {})
            stream_tts = auto_speak and bool(speech_cfg.get("stream_tts", True))

            try:
                max_turns = max(0, int(ollama_cfg.get("max_history_turns", 12)))
            except (TypeError, ValueError):
                max_turns = 12
            with self._messages_lock:
                generation = self._conversation_generation
                history = list(self.messages[-max_turns * 2:]) if max_turns else []

            request_messages = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(history)
            request_messages.append({"role": "user", "content": user_text})

            payload = {
                "model": model,
                "messages": request_messages,
                "stream": True,
            }
            native_payload = dict(payload)
            keep_alive = ollama_cfg.get("keep_alive")
            if keep_alive not in (None, ""):
                native_payload["keep_alive"] = keep_alive

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
            self._trace_ollama_request("NATIVE POST /api/chat", chat_url, native_payload)

            response = requests.post(
                chat_url,
                json=native_payload,
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
                    if cancel_event.is_set():
                        break
                    if not raw_line:
                        continue

                    if api_mode == "native":
                        chunk = json.loads(raw_line)
                        # Ollama reports mid-stream failures (model not found,
                        # out of memory, ...) as {"error": "..."}.
                        if chunk.get("error"):
                            raise RuntimeError(
                                f"Ollama reported an error: {chunk['error']}"
                            )
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

                        # Speak completed sentences while the rest streams.
                        if stream_tts:
                            speech_buffer += token
                            ready, speech_buffer = self._split_speakable(speech_buffer)
                            if ready.strip():
                                spoke_streamed = True
                                self._queue_tts(ready, label="Speaking response")

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

            if cancel_event.is_set():
                self.ui_queue.put(("response_chunk", " [cancelled]\n"))
                self.ui_queue.put(("status", "Response cancelled."))
                return

            if assistant_text:
                with self._messages_lock:
                    # Skip if the conversation was cleared mid-request.
                    if generation == self._conversation_generation:
                        self.messages.append({"role": "user", "content": user_text})
                        self.messages.append({
                            "role": "assistant",
                            "content": assistant_text,
                        })
                self.ui_queue.put(("response_chunk", "\n"))

                if stream_tts:
                    if speech_buffer.strip():
                        self._queue_tts(speech_buffer, label="Speaking response")
                    elif not spoke_streamed:
                        self.ui_queue.put(("status", "Response had nothing speakable."))
                elif auto_speak:
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
                "Cannot connect to Ollama.\n\n"
                f"{exc}\n\n"
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
                f"Ollama HTTP error:\n{exc}\n\n{detail}"
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
                f"Ollama request error:\n"
                f"{type(exc).__name__}: {exc}"
            ))
        finally:
            first_token.set()
            with self._response_wait_lock:
                if self._response_wait_event is first_token:
                    self._response_wait_event = None
                if self._active_request_cancel is cancel_event:
                    self._active_request_cancel = None
            self.is_processing = False
