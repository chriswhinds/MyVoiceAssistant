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

set -u

echo "=== Mac Pro 6,1 / Ubuntu Local STT GPU Diagnostics ==="
echo
echo "--- PCI display devices ---"
lspci -nnk | grep -EA4 'VGA|3D|Display' || true

echo
echo "--- Kernel / architecture ---"
uname -a
uname -m

echo
echo "--- Vulkan ---"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary || true
else
  echo "vulkaninfo not installed"
fi

echo
echo "--- Mesa ---"
dpkg -l | grep -E 'mesa-vulkan|libvulkan' || true

echo
echo "--- Whisper.cpp ---"
ROOT="${WHISPER_CPP_ROOT:-$HOME/.local/share/ollama-voice-assistant/whisper.cpp}"
if [ -x "$ROOT/build/bin/whisper-cli" ]; then
  "$ROOT/build/bin/whisper-cli" -h 2>&1 | head -40
else
  echo "whisper-cli not found at $ROOT/build/bin/whisper-cli"
fi

echo
echo "--- Ollama ---"
curl -sS http://localhost:11434/api/tags 2>/dev/null | head -c 1000 || true
echo
