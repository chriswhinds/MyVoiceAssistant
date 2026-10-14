#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_DIR="${HOME}/.local/share/ollama-voice-assistant/piper"

echo "Installing Ollama Voice Assistant v7 prerequisites for Ubuntu 26.04..."

sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip python3-tk python3-dev \
  build-essential portaudio19-dev libportaudio2 \
  espeak-ng pipewire-bin pulseaudio-utils alsa-utils \
  git cmake ninja-build pkg-config \
  libvulkan-dev mesa-vulkan-drivers vulkan-tools glslc spirv-headers \
  ffmpeg curl

cd "${APP_DIR}"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

echo
echo "Downloading Piper neural female voice..."
mkdir -p "${VOICE_DIR}"

python -m piper.download_voices \
  --data-dir "${VOICE_DIR}" \
  en_US-hfc_female-medium

echo
echo "Voice files:"
ls -lh "${VOICE_DIR}/en_US-hfc_female-medium.onnx" \
       "${VOICE_DIR}/en_US-hfc_female-medium.onnx.json"

echo
echo "Local speech environment is ready."
echo
echo "Optional/recommended AMD Vulkan STT acceleration:"
echo "  ./build_whisper_vulkan.sh base.en"
echo
echo "Run the application:"
echo "  source .venv/bin/activate"
echo "  python MyVoiceAssistant_v7_11.py"
