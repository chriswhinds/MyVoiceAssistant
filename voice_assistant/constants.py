"""Build identifier and project paths shared by every module."""

from pathlib import Path

APP_BUILD = "v7.11.0"

# Folder holding MyVoiceAssistant_v7_11.py and ollama_voice_assistant_config.json.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
