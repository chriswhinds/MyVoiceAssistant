#!/usr/bin/env bash
# Copyright 2026 Christopher Hinds, Stratum LLC
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
