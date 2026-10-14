<!--
  Copyright 2026 Christopher Hinds, Stratum LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  SPDX-License-Identifier: Apache-2.0
-->

# MyVoiceAssistant

A hands-free voice assistant that runs entirely on your own hardware. You speak to it through a microphone; it transcribes your speech locally, sends the text to an [Ollama](https://ollama.com) language model on your network, and speaks the answer back in a natural female voice while an animated avatar's mouth moves in time with the audio.

Nothing leaves your network: speech recognition, the language model and speech synthesis are all local.

```
USB microphone
   │  voice-activity detection, wake word ("Hey Alice")
   ▼
Speech-to-text ── whisper.cpp server on the GPU (Vulkan)
   │               └─ fallback: Faster-Whisper on the CPU
   ▼
Ollama (streaming chat, any model you have pulled)
   │
   ▼
Text-to-speech ── Piper neural voice (en_US-hfc_female-medium)
   │               └─ fallback: espeak-ng
   ▼
Speakers (PipeWire / PulseAudio / ALSA)  +  lip-synced avatar
```

It was built for a **Mac Pro 6,1 running Ubuntu 26.04** (Xeon CPU, AMD FirePro GPUs via Mesa RADV Vulkan), but should work on any x86_64 Linux machine with a microphone.

## Features

- **Wake words**: say *"Hey Alice"* to start a conversation and *"Alice, go to sleep"* to stop. You can also use the Start/Stop buttons.
- **Speaks as it thinks**: the reply is spoken sentence by sentence while Ollama is still generating, rather than waiting for the full answer.
- **Interruptible**: the **Stop Speaking** button, or the stop phrase, cancels the reply and any queued speech.
- **GPU speech recognition**: a persistent `whisper-server` keeps the Whisper model loaded on the GPU, transcribing about 11 seconds of speech in under a second on the Mac Pro.
- **Standby reminders**: while Ollama is thinking, it says *"I'm researching your request, please stand by"* so you know it heard you.
- **Conversation memory**: recent turns are sent back to Ollama as context, so follow-up questions work.
- **Clean speech**: markdown, code blocks and symbols are removed before speaking.
- **Diagnostics**: built-in microphone, speaker, Ollama, audio and STT tests.

## Requirements

| Component | Notes |
|---|---|
| Linux x86_64 | Developed on Ubuntu 26.04 |
| Python 3.12+ | Tested on Python 3.14 |
| An Ollama server | On this machine or elsewhere on your network, with at least one model pulled (e.g. `ollama pull llama3.2`) |
| Microphone and speakers | Any device PortAudio can see; USB microphones work well |
| Vulkan-capable GPU | *Optional.* Without one, speech recognition runs on the CPU with Faster-Whisper |

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/chriswhinds/MyVoiceAssistant.git
   cd MyVoiceAssistant
   ```

2. **Install system packages, the Python environment and the Piper voice**

   ```bash
   chmod +x install_ubuntu_26_04_v7.sh build_whisper_vulkan.sh
   ./install_ubuntu_26_04_v7.sh
   ```

   This uses `sudo apt` to install audio, Tk, PortAudio, espeak-ng and Vulkan packages. It then creates `.venv/`, installs `requirements.txt`, and downloads the Piper voice to `~/.local/share/ollama-voice-assistant/piper/`.

3. **(Recommended) Build whisper.cpp with Vulkan for GPU speech recognition**

   ```bash
   ./build_whisper_vulkan.sh base.en
   ```

   This builds `whisper-cli` and `whisper-server` into `~/.local/share/ollama-voice-assistant/whisper.cpp/` and downloads the `ggml-base.en.bin` model. Pass a different model name (e.g. `small.en`) for better accuracy at some cost in speed. If you skip this step, the app uses Faster-Whisper on the CPU.

4. **Point the app at your Ollama server**

   Edit `ollama_voice_assistant_config.json` and set the host, port and model:

   ```json
   "ollama": {
     "host": "192.168.1.165",
     "port": 11434,
     "model": "llama3.2"
   }
   ```

   Use `127.0.0.1` if Ollama runs on the same machine.

## Running

```bash
source .venv/bin/activate
python MyVoiceAssistant_v7_11.py
```

or, without activating the environment:

```bash
.venv/bin/python MyVoiceAssistant_v7_11.py
```

On launch the app checks its dependencies, warns about anything missing, and arms the wake-word listener.

### First-run checklist

1. Choose your microphone from the **Microphone** list and press **Test Mic**. It should report that a signal is present.
2. Press **Test Speaker**. You should hear the assistant's voice.
3. Press **Load Models**, then pick an Ollama model from the **Model** list (or **Test Ollama** to check the connection).
4. Say **"Hey Alice"**, wait for *"I'm Listening"*, then ask your question. After about 5 seconds of silence, your speech is sent to Ollama.
5. Say **"Alice, go to sleep"** to return to standby.

### Window overview

| Area | Purpose |
|---|---|
| Voice and Ollama Settings | Microphone, STT backend/model/threads, Ollama URL and model, system prompt |
| Start / Stop Listening | Manually enter or leave active listening (same as the wake words) |
| Stop Speaking | Cancel the current reply and silence speech immediately |
| Test Ollama / Test Speaker / Test Mic | Quick checks of each part of the pipeline |
| Audio Diagnostics / STT Diagnostics | Show detected audio devices, playback backend, Vulkan GPU and Whisper status |
| Speak Last Response | Replay the most recent answer |
| Clear / Save Conversation | Reset the chat history, or save the transcript to a text file |
| Reload Config | Re-read `ollama_voice_assistant_config.json` without restarting |

## Configuration

All settings live in `ollama_voice_assistant_config.json` beside the script. Any value you leave out falls back to a built-in default. To use a config file elsewhere, set `OLLAMA_VOICE_CONFIG=/path/to/config.json`.

| Setting | Default | Description |
|---|---|---|
| `ollama.host`, `ollama.port` | `127.0.0.1`, `11434` | Ollama server address |
| `ollama.model` | `llama3.2` | Model used for replies (can also be changed in the UI) |
| `ollama.max_history_turns` | `12` | How many previous question/answer pairs are sent as context |
| `ollama.keep_alive` | `30m` | How long Ollama keeps the model loaded between requests |
| `system_prompt` | *assistant persona* | Instructions given to the model; also editable in the UI |
| `standby.enabled` / `interval_seconds` / `phrase` | `true` / `10` / *"I'm researching…"* | Spoken reminder while waiting for Ollama |
| `speech.endpoint_silence_seconds` | `5.0` | Seconds of silence that end your request |
| `speech.max_utterance_seconds` | `120` | Longest single request |
| `speech.low_latency_mode` | `true` | Prefer backends that keep the speech model loaded |
| `speech.stream_tts` | `true` | Speak sentence by sentence while the reply streams |
| `speech.avatar_sync_offset_ms` | `80` | Delay the avatar's mouth to line up with audio playback |
| `trigger_words.enabled` | `true` | Use wake words |
| `trigger_words.auto_arm_on_launch` | `true` | Start listening for the wake word when the app opens |
| `trigger_words.start` / `stop` | *see file* | Phrases that start and stop active listening |
| `trigger_words.start_acknowledgement` / `stop_acknowledgement` | *see file* | What the assistant says when it wakes or sleeps |
| `trigger_words.post_tts_guard_seconds` | `0.25` | Pause after speaking before listening again, to avoid hearing itself |
| `diagnostics.trace_enabled` | `false` | Print detailed listener, STT and Ollama traces to the terminal |

### Environment variables

| Variable | Purpose |
|---|---|
| `OLLAMA_VOICE_CONFIG` | Path to an alternative config file |
| `WHISPER_CPP_ROOT` | whisper.cpp install folder, if not `~/.local/share/ollama-voice-assistant/whisper.cpp` |
| `WHISPER_CPP_MODEL` | Explicit path to a GGML Whisper model |
| `PIPER_VOICE_MODEL` | Explicit path to a Piper `.onnx` voice (its `.onnx.json` must sit beside it) |

### Speech-to-text backends

The **Local STT backend** list in the window chooses how speech is transcribed:

- **Auto** (default): uses `whisper-server` on the Vulkan GPU when it is available, otherwise Faster-Whisper on the CPU.
- **Vulkan (whisper.cpp)**: always uses the GPU. It uses `whisper-server` and falls back to one-shot `whisper-cli`.
- **Faster-Whisper CPU**: always uses the CPU.

Measured on the Mac Pro 6,1 with `base.en` and an 11-second clip: `whisper-server` 0.8 s, `whisper-cli` 2.0 s, Faster-Whisper 2.9 s.

## Troubleshooting

| Symptom | What to try |
|---|---|
| "Ubuntu prerequisites need attention" on launch | Re-run `./install_ubuntu_26_04_v7.sh`; the dialog lists what is missing |
| Wake word never triggers | Use **Test Mic**: a low signal means you should raise the input gain in Ubuntu's sound settings or pick another device. Turn on `diagnostics.trace_enabled` to watch the live levels |
| Speech is robotic | Piper failed to load and espeak-ng is being used; check the status bar message and that the voice files exist under `~/.local/share/ollama-voice-assistant/piper/` |
| Transcription is slow | Open **STT Diagnostics** to confirm the GPU and `whisper-server` are detected; build them with `./build_whisper_vulkan.sh` |
| "Cannot connect to Ollama" | Check the host and port in the config, and that Ollama listens on the network (`OLLAMA_HOST=0.0.0.0` on the server) |
| GPU / Vulkan problems | Run `./diagnose_macpro61_vulkan.sh` for a full report |

## Project layout

```
MyVoiceAssistant_v7_11.py        entry point
voice_assistant/
  app.py                         composes the modules and owns shared state
  config.py                      JSON config, Ollama URL handling, tracing
  listener.py                    microphone, wake words, voice-activity detection
  stt.py                         speech-to-text backend selection
  whisper_server.py              persistent whisper.cpp server process
  ollama_client.py               model selection and streaming chat
  tts.py                         Piper / espeak-ng synthesis and playback
  avatar.py                      animated, lip-synced avatar
  ui.py                          Tk window and UI event queue
  text_utils.py                  text and audio helpers
ollama_voice_assistant_config.json
install_ubuntu_26_04_v7.sh       system + Python setup
build_whisper_vulkan.sh          whisper.cpp Vulkan build
diagnose_macpro61_vulkan.sh      GPU / audio diagnostics
LICENSE                          Apache License 2.0
README_v7.txt                    version history
OllamaVoiceAssistant_v7_*.py     earlier single-file versions, kept for reference
```

See `README_v7.txt` for the detailed change history.

## License

Copyright 2026 Christopher Hinds, Stratum LLC.

Licensed under the [Apache License, Version 2.0](LICENSE). You may not use this project except in compliance with the License.
