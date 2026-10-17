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

"""Amplitude-driven animated avatar drawn on a Tk canvas (UI thread only)."""

import math
import random
import time


class AvatarMixin:
    def _animate_avatar(self):
        """
        UI-thread animation loop.

        Ubuntu WAV-based TTS exposes an amplitude envelope, so the animation
        follows the real playback window. While TTS is active, mouth opening,
        jaw position and head motion vary naturally; when playback returns,
        the face immediately settles back to idle.
        """
        if self.shutdown_requested:
            return

        now = time.time()
        self.avatar_phase += 0.30

        # Natural blink every few seconds.
        if self.avatar_blink > 0:
            self.avatar_blink -= 1
        elif now >= self.avatar_next_blink:
            self.avatar_blink = 3
            self.avatar_next_blink = now + random.uniform(2.5, 5.0)

        speaking = bool(self.avatar_speaking)

        if speaking:
            # Prefer the actual amplitude envelope from the espeak-ng WAV.
            mouth = None

            if self.avatar_envelope and self.avatar_envelope_started:
                elapsed_ms = (
                    time.monotonic() - self.avatar_envelope_started
                ) * 1000.0

                index = int(
                    elapsed_ms / float(self.avatar_envelope_frame_ms)
                )

                if 0 <= index < len(self.avatar_envelope):
                    mouth = self.avatar_envelope[index]

            # Fallback animation during the tiny interval before playback starts.
            if mouth is None:
                target = 0.18 + 0.75 * random.random()
                self.avatar_mouth_level = (
                    self.avatar_mouth_level * 0.45 + target * 0.55
                )
                mouth = self.avatar_mouth_level

            # Smooth abrupt amplitude changes for more natural jaw motion.
            self.avatar_mouth_level = (
                self.avatar_mouth_level * 0.35 + mouth * 0.65
            )
            mouth = max(0.06, min(1.0, self.avatar_mouth_level))
        else:
            self.avatar_mouth_level *= 0.50
            mouth = self.avatar_mouth_level

        blink = self.avatar_blink > 0
        self._draw_avatar(mouth, blink, self.avatar_phase if speaking else 0.0)

        self.root.after(70, self._animate_avatar)

    def _draw_avatar(self, mouth_level, blink, phase):
        """Draw a more human-looking female assistant avatar."""
        c = self.avatar_canvas
        c.delete("all")

        try:
            w = max(c.winfo_width(), 330)
            h = max(c.winfo_height(), 330)
        except Exception:
            w, h = 330, 330

        cx = w / 2
        cy = h / 2 + 8

        if self.avatar_speaking:
            cx += 2.0 * math.sin(phase * 0.42)
            cy += 1.5 * math.sin(phase * 0.66)

        # Soft background.
        c.create_oval(
            cx - 145, cy - 155, cx + 145, cy + 170,
            fill="#dde8f0", outline=""
        )

        # Shoulders and blouse.
        c.create_arc(
            cx - 130, cy + 90, cx + 130, cy + 235,
            start=0, extent=180,
            fill="#44576f", outline="#34475f", width=2
        )
        c.create_polygon(
            cx - 38, cy + 94,
            cx, cy + 132,
            cx + 38, cy + 94,
            fill="#f6f7fb", outline=""
        )

        # Long hair behind the face.
        hair = "#4a3028"
        hair_highlight = "#65483c"
        c.create_oval(
            cx - 111, cy - 143, cx + 111, cy + 145,
            fill=hair, outline="#34221d", width=2
        )
        c.create_polygon(
            cx - 107, cy - 55,
            cx - 126, cy + 145,
            cx - 70, cy + 160,
            cx - 55, cy + 38,
            fill=hair, outline=""
        )
        c.create_polygon(
            cx + 107, cy - 55,
            cx + 126, cy + 145,
            cx + 70, cy + 160,
            cx + 55, cy + 38,
            fill=hair, outline=""
        )

        # Neck.
        skin = "#e7b58f"
        skin_edge = "#a8755d"
        c.create_rectangle(
            cx - 34, cy + 68, cx + 34, cy + 118,
            fill=skin, outline=""
        )

        # Face: slightly tapered jaw rather than a simple oval.
        face_points = [
            cx - 82, cy - 92,
            cx - 91, cy - 28,
            cx - 78, cy + 38,
            cx - 53, cy + 80,
            cx, cy + 105,
            cx + 53, cy + 80,
            cx + 78, cy + 38,
            cx + 91, cy - 28,
            cx + 82, cy - 92,
        ]
        c.create_polygon(
            face_points,
            fill=skin, outline=skin_edge, width=2, smooth=True
        )

        # Hairline and side framing.
        c.create_arc(
            cx - 92, cy - 137, cx + 92, cy - 35,
            start=3, extent=174, style="arc",
            outline=hair_highlight, width=22
        )
        c.create_arc(
            cx - 104, cy - 105, cx - 42, cy + 62,
            start=88, extent=122, style="arc",
            outline=hair, width=17
        )
        c.create_arc(
            cx + 42, cy - 105, cx + 104, cy + 62,
            start=-30, extent=122, style="arc",
            outline=hair, width=17
        )

        # Eyebrows.
        brow_y = cy - 36
        c.create_arc(
            cx - 69, brow_y - 9, cx - 24, brow_y + 11,
            start=15, extent=145, style="arc",
            outline="#5b4034", width=3
        )
        c.create_arc(
            cx + 24, brow_y - 9, cx + 69, brow_y + 11,
            start=20, extent=145, style="arc",
            outline="#5b4034", width=3
        )

        # Eyes and lashes.
        eye_y = cy - 13
        if blink:
            c.create_arc(
                cx - 70, eye_y - 4, cx - 24, eye_y + 10,
                start=200, extent=140, style="arc",
                outline="#2f2928", width=3
            )
            c.create_arc(
                cx + 24, eye_y - 4, cx + 70, eye_y + 10,
                start=200, extent=140, style="arc",
                outline="#2f2928", width=3
            )
        else:
            for ex in (cx - 47, cx + 47):
                c.create_oval(
                    ex - 21, eye_y - 12, ex + 21, eye_y + 12,
                    fill="#fafcff", outline="#6f6460", width=2
                )
                pupil_shift = 1.5 * math.sin(phase * 0.22) if self.avatar_speaking else 0
                c.create_oval(
                    ex - 7 + pupil_shift, eye_y - 8,
                    ex + 7 + pupil_shift, eye_y + 8,
                    fill="#6b583e", outline=""
                )
                c.create_oval(
                    ex - 3 + pupil_shift, eye_y - 4,
                    ex + 3 + pupil_shift, eye_y + 4,
                    fill="#1d2227", outline=""
                )
                c.create_oval(
                    ex - 1 + pupil_shift, eye_y - 5,
                    ex + 2 + pupil_shift, eye_y - 2,
                    fill="white", outline=""
                )

            # Upper lashes.
            c.create_arc(
                cx - 70, eye_y - 15, cx - 24, eye_y + 7,
                start=18, extent=145, style="arc",
                outline="#2f2928", width=3
            )
            c.create_arc(
                cx + 24, eye_y - 15, cx + 70, eye_y + 7,
                start=18, extent=145, style="arc",
                outline="#2f2928", width=3
            )

        # Nose with subtle shading.
        c.create_line(
            cx - 2, cy - 1,
            cx - 7, cy + 30,
            cx + 2, cy + 34,
            smooth=True, width=2, fill="#b47f65"
        )
        c.create_arc(
            cx - 12, cy + 24, cx + 13, cy + 41,
            start=205, extent=130, style="arc",
            outline="#b47f65", width=2
        )

        # Cheek highlights.
        c.create_oval(cx - 74, cy + 30, cx - 48, cy + 42, fill="#e9a59d", outline="")
        c.create_oval(cx + 48, cy + 30, cx + 74, cy + 42, fill="#e9a59d", outline="")

        # Mouth driven by the actual speech amplitude.
        mouth_y = cy + 61

        if mouth_level < 0.08:
            # Resting smile.
            c.create_arc(
                cx - 39, mouth_y - 11,
                cx + 39, mouth_y + 20,
                start=205, extent=130,
                style="arc", outline="#a23f55", width=4
            )
            c.create_arc(
                cx - 31, mouth_y - 4,
                cx + 31, mouth_y + 14,
                start=20, extent=140,
                style="arc", outline="#d77588", width=2
            )
        else:
            open_h = 8 + mouth_level * 31
            open_w = 50 + mouth_level * 22

            # Lip outline.
            c.create_oval(
                cx - open_w / 2 - 3,
                mouth_y - open_h / 2 - 2,
                cx + open_w / 2 + 3,
                mouth_y + open_h / 2 + 2,
                fill="#b14b62", outline="#96394f", width=2
            )

            # Mouth interior.
            c.create_oval(
                cx - open_w / 2,
                mouth_y - open_h / 2,
                cx + open_w / 2,
                mouth_y + open_h / 2,
                fill="#5c2730", outline=""
            )

            if mouth_level > 0.36:
                c.create_arc(
                    cx - open_w / 2 + 5,
                    mouth_y - open_h / 2 + 2,
                    cx + open_w / 2 - 5,
                    mouth_y - open_h / 2 + max(12, open_h * 0.45),
                    start=0, extent=180,
                    fill="#fffaf7", outline="#fffaf7", style="pieslice"
                )

            if mouth_level > 0.58:
                c.create_arc(
                    cx - open_w * 0.27,
                    mouth_y + 3,
                    cx + open_w * 0.27,
                    mouth_y + open_h / 2 - 1,
                    start=180, extent=180,
                    fill="#cf7180", outline="#cf7180", style="pieslice"
                )

        # Earrings.
        c.create_oval(cx - 94, cy + 28, cx - 86, cy + 38, fill="#d7b450", outline="")
        c.create_oval(cx + 86, cy + 28, cx + 94, cy + 38, fill="#d7b450", outline="")

        # Speech halo.
        if self.avatar_speaking:
            pulse = 5 + 3 * abs(math.sin(phase * 0.62))
            c.create_oval(
                cx - 119 - pulse, cy - 150 - pulse,
                cx + 119 + pulse, cy + 158 + pulse,
                outline="#7ba5c3", width=2
            )
