Ollama Voice Assistant v7 - Female Avatar / Local STT / AMD Vulkan
======================================================================

Target:
  Mac Pro 6,1
  Ubuntu Linux 26.04
  Local Ollama
  Fully local speech recognition

Changes from v6
---------------
1. Female human-style animated Tkinter avatar.
2. Piper neural female TTS:
     en_US-hfc_female-medium
   espeak-ng female voice remains an automatic fallback.
3. Listening capture increased from 15 seconds to 120 seconds.
4. SpeechRecognition pause threshold increased to 3 seconds for more natural
   pauses during longer dictation.
5. Immediately after a user request is transcribed, the assistant says:
     "I'm researching your request, please stand by."
   Ollama processing begins while this acknowledgement is playing.
6. The microphone remains blocked while assistant speech is on the speakers,
   preventing feedback.
7. Final Ollama answer is queued behind the acknowledgement and spoken using
   the same neural female voice.
8. Avatar mouth motion remains tied to the real generated WAV amplitude.

Install
-------
  chmod +x install_ubuntu_26_04_v7.sh
  ./install_ubuntu_26_04_v7.sh

Optional Vulkan STT acceleration:
  ./build_whisper_vulkan.sh base.en

Run
---
  source .venv/bin/activate
  python OllamaVoiceAssistant_v7_LocalSTT_AMD_Vulkan_Female.py

Ollama
------
Expected local API:
  http://localhost:11434

Notes about 2-minute listening
------------------------------
phrase_time_limit is now 120 seconds. The recognizer can still end an utterance
before 120 seconds after a sustained pause. pause_threshold has therefore been
raised to 3 seconds, allowing normal thinking pauses while dictating.

TTS
---
Primary:
  Piper en_US-hfc_female-medium

Voice directory:
  ~/.local/share/ollama-voice-assistant/piper/

Fallback:
  espeak-ng -v en-us+f3

Everything remains local after installation/model download.

v7.11.0 - MyVoiceAssistant (MyVoiceAssistant_v7_11.py)
------------------------------------------------------
Based on v7.10.9. Run:
  source .venv/bin/activate
  python MyVoiceAssistant_v7_11.py

Layout (the old single 3,300-line file is now a package):
  MyVoiceAssistant_v7_11.py   entry point
  voice_assistant/
    app.py             composes the mixins below, owns shared state
    config.py          JSON config, Ollama URL sanitizing, tracing
    listener.py        microphone, trigger words, VAD listen loop
    stt.py             speech-to-text backend selection
    whisper_server.py  persistent whisper.cpp server process
    ollama_client.py   model selection, streaming chat
    tts.py             Piper/espeak-ng synthesis, interruptible playback
    avatar.py          animated avatar
    ui.py              Tk layout, UI event queue
    text_utils.py      pure text/audio helpers

Speech-to-text
  Auto now prefers a persistent whisper-server on the Vulkan GPU (the model
  stays loaded: ~0.8 s for 11 s of audio vs ~2.0 s for one-shot whisper-cli
  and ~2.9 s for Faster-Whisper CPU on the Mac Pro 6,1). If the server
  fails it falls back to Faster-Whisper (Auto) or whisper-cli (Vulkan).
  whisper-server is built by build_whisper_vulkan.sh alongside whisper-cli.
  The server is stopped on exit, and killed automatically if the app crashes.

Fixes
  - Error/diagnostic dialogs showed literal "\n" instead of line breaks.
  - "diagnostics" section of the config was never loaded, so
    trace_enabled=false had no effect.
  - A single speech-to-text failure stopped the listener permanently.
  - Ollama mid-stream {"error": ...} messages are now shown instead of
    "empty response".
  - Whisper filler output ([BLANK_AUDIO], "(silence)", "Thanks for
    watching") is no longer sent to Ollama.
  - A start trigger heard while a request was in flight could launch a
    second concurrent request.
  - Clearing the conversation during a reply no longer re-adds that reply.
  - Worker threads no longer read Tk variables; system prompt, Ollama URL
    and auto-speak changes now apply immediately to the next request.
  - Identical error dialogs are suppressed for 30 s (status bar instead).
  - Correct installer name in messages; empty CPU-threads box no longer
    crashes; whisper.cpp language matches the model file actually used.
  - System prompt typos fixed; prompt now asks for no markdown.

Improvements
  - Replies are spoken sentence-by-sentence while Ollama is still
    streaming (speech.stream_tts).
  - Markdown/code is stripped before speaking.
  - "Stop Speaking" button; the stop trigger phrase also cancels the
    in-flight reply and any queued speech.
  - History trimmed to ollama.max_history_turns; ollama.keep_alive keeps
    the model loaded between requests.
  - vulkaninfo result cached (re-probed by STT Diagnostics).
  - numpy RMS / mouth envelope (much faster on long replies).
  - speech.avatar_sync_offset_ms aligns mouth movement with audio start.
