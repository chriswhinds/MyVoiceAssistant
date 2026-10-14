#!/usr/bin/env bash
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
