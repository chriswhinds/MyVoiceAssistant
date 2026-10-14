"""Pure text/audio helpers: STT noise filtering, speech cleanup, PCM RMS."""

import math
import re
from array import array

try:
    import numpy as np
except Exception:  # numpy ships with faster-whisper; pure-Python fallback otherwise
    np = None


class TextUtilsMixin:
    # Whisper commonly emits these for silence, breathing or room noise.
    # Real short replies such as "thank you" or "okay" are deliberately kept.
    _STT_NOISE_PHRASES = {
        "", "you", "thank you for watching", "thanks for watching",
        "please subscribe", "uh", "um", "hmm", "mm",
    }

    @classmethod
    def _filter_stt_noise(cls, text):
        """Strip Whisper tags like [BLANK_AUDIO]/(silence); drop pure noise."""
        cleaned = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        normalized = re.sub(r"[^a-z0-9' ]+", "", cleaned.lower()).strip()
        if normalized in cls._STT_NOISE_PHRASES:
            return ""
        return cleaned

    @staticmethod
    def _clean_text_for_speech(text):
        """Remove markdown/code so the TTS voice does not read symbols aloud."""
        text = text or ""
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"```.*$", " ", text, flags=re.S)  # unterminated block
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links/images
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # headings
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # bullets
        text = re.sub(r"^\s*>\s?", "", text, flags=re.M)  # quotes
        text = re.sub(r"(\*\*|\*|~~)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
        # Underscore emphasis only at word edges, so snake_case survives.
        text = re.sub(r"(?<!\w)(__|_)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
        text = re.sub(r"^\s*([-*_]\s*){3,}$", " ", text, flags=re.M)  # rules
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("|", " ").replace("#", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _split_speakable(buffer, min_chars=40):
        """
        Split complete sentences off the front of a streaming buffer.

        Returns (ready_text, remainder). Short sentences are merged so the
        voice does not sound choppy; text inside an open ``` block is held
        back until the block closes.
        """
        if buffer.count("```") % 2 == 1:
            return "", buffer
        last_cut = 0
        for match in re.finditer(r"(?<=[.!?:;])[\"')\]]*\s+|\n\s*\n", buffer):
            if match.end() >= min_chars:
                last_cut = match.end()
        if last_cut == 0:
            return "", buffer
        return buffer[:last_cut], buffer[last_cut:]

    @staticmethod
    def _pcm16_rms(raw_bytes):
        """Return RMS amplitude for little-endian signed 16-bit PCM."""
        if not raw_bytes or len(raw_bytes) < 2:
            return 0.0
        if np is not None:
            samples = np.frombuffer(
                raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 2)],
                dtype="<i2",
            ).astype(np.float64)
            return float(np.sqrt(np.mean(samples * samples)))
        samples = array("h")
        samples.frombytes(raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 2)])
        total = 0.0
        for sample in samples:
            value = float(sample)
            total += value * value
        return math.sqrt(total / len(samples))

    @staticmethod
    def _normalize_trigger_text(text):
        words = re.findall(r"[a-z0-9']+", (text or "").lower())
        return " ".join(words).strip()

    @staticmethod
    def _normalize_ui_text(value):
        # Only real CR/LF are normalized now that message strings use real
        # newlines; rewriting a literal backslash-n would corrupt code that
        # the model returns.
        if value is None:
            return ""
        return str(value).replace("\r\n", "\n").replace("\r", "\n")
