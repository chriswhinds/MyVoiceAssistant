#!/usr/bin/env bash
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

set -euo pipefail

ROOT="${WHISPER_CPP_ROOT:-$HOME/.local/share/ollama-voice-assistant/whisper.cpp}"
MODEL="${1:-base.en}"

echo "Installing Vulkan/whisper.cpp build prerequisites..."
sudo apt update
sudo apt install -y \
  git cmake ninja-build build-essential pkg-config \
  libvulkan-dev mesa-vulkan-drivers vulkan-tools glslc spirv-headers \
  ffmpeg curl

mkdir -p "$(dirname "$ROOT")"

if [ ! -d "$ROOT/.git" ]; then
  git clone https://github.com/ggml-org/whisper.cpp.git "$ROOT"
else
  git -C "$ROOT" pull --ff-only
fi

echo
echo "Checking Vulkan..."
if ! vulkaninfo --summary; then
  echo
  echo "ERROR: Vulkan is not currently usable."
  echo "The application can still use Faster-Whisper CPU."
  exit 2
fi

echo
echo "Building whisper.cpp with Vulkan..."
cmake -S "$ROOT" -B "$ROOT/build" \
  -DGGML_VULKAN=1 \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$ROOT/build" -j"$(nproc)"

echo
echo "Downloading GGML model: $MODEL"
(
  cd "$ROOT"
  sh ./models/download-ggml-model.sh "$MODEL"
)

echo
echo "Vulkan whisper.cpp installation complete."
echo "CLI:   $ROOT/build/bin/whisper-cli"
echo "Model: $ROOT/models/ggml-$MODEL.bin"
echo
echo "Set these only if you installed somewhere else:"
echo "  export WHISPER_CPP_ROOT=\"$ROOT\""
echo "  export WHISPER_CPP_MODEL=\"$ROOT/models/ggml-$MODEL.bin\""
