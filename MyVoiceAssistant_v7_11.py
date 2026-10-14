#!/usr/bin/env python3
"""MyVoiceAssistant v7.11 entry point.

The application lives in the voice_assistant/ package:

    config.py         JSON config, Ollama URL sanitizing, tracing
    listener.py       microphone, trigger words, VAD listen loop
    stt.py            whisper.cpp server/CLI (Vulkan) and Faster-Whisper (CPU)
    whisper_server.py persistent whisper-server process manager
    ollama_client.py  model selection and streaming chat
    tts.py            Piper/espeak-ng synthesis and interruptible playback
    avatar.py         amplitude-driven animated avatar
    ui.py             Tk layout and UI event queue
    text_utils.py     pure text/audio helpers
    app.py            composes the mixins and owns shared state
"""

from voice_assistant.app import main

if __name__ == "__main__":
    main()
