"""
ARIA — Weather Condition Detector + Frame Preprocessor
Runs before every detection pass. Applies CLAHE / defogging / denoising
based on detected scene conditions.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class Condition(Enum):
    CLEAR = "clear"
    RAIN  = "rain"
    FOG   = "fog"
    NIGHT = "night"
    DUST  = "dust"


@dataclass
class WeatherResult:
    condition:   Condition
    visibility:  float       # 0.0 (blind) → 1.0 (perfect)
    applied:     list[str]   = field(default_factory=list)


class WeatherProcessor:
    """
    Detects scene condition and applies the appropriate preprocessing.
    Fast enough to run every frame at 25 Hz on a Pi 4B.
    """

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, WeatherResult]:
        result = self._assess(frame)
        processed = self._enhance(frame.copy(), result)
        return processed, result

    # ── Assessment ────────────────────────────────────────────────────────────

    def _assess(self, frame: np.ndarray) -> WeatherResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast   = float(gray.std())
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 50:
            return WeatherResult(Condition.NIGHT, brightness / 50.0)
        if contrast < 28 and sharpness < 80:
            return WeatherResult(Condition.FOG,  contrast / 28.0)
        if sharpness < 150 and contrast > 28:
            return WeatherResult(Condition.RAIN, min(sharpness / 150.0, 1.0))
        if contrast < 40 and brightness > 100:
            return WeatherResult(Condition.DUST, contrast / 40.0)
        return WeatherResult(Condition.CLEAR, 1.0)

    # ── Enhancement ───────────────────────────────────────────────────────────

    def _enhance(self, frame: np.ndarray, r: WeatherResult) -> np.ndarray:
        if r.condition == Condition.NIGHT:
            frame = self._clahe(frame, clip=3.0)
            r.applied.append("clahe_night")

        elif r.condition == Condition.FOG:
            frame = self._dehaze(frame)
            r.applied.append("dark_channel_dehaze")

        elif r.condition == Condition.RAIN:
            frame = cv2.medianBlur(frame, 3)
            frame = self._clahe(frame, clip=2.0)
            r.applied.append("rain_denoise_clahe")

        elif r.condition == Condition.DUST:
            frame = self._clahe(frame, clip=2.5)
            r.applied.append("clahe_dust")

        return frame

    def _clahe(self, frame: np.ndarray, clip: float = 2.0) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _dehaze(self, img: np.ndarray, omega: float = 0.95) -> np.ndarray:
        """Dark channel prior (He et al. 2009) — lightweight real-time version."""
        dark = np.min(img, axis=2).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_ch = cv2.erode(dark, k)

        flat = dark_ch.flatten()
        top  = np.argsort(flat)[-max(1, int(0.001 * len(flat))):]
        atm  = np.max(img.reshape(-1, 3)[top], axis=0).astype(np.float32)

        norm = img.astype(np.float32) / (atm + 1e-6)
        k2   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        t    = np.clip(1 - omega * cv2.erode(np.min(norm, axis=2), k2), 0.1, 1.0)

        out = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = (img[:, :, c].astype(np.float32) - atm[c]) / t + atm[c]
        return np.clip(out, 0, 255).astype(np.uint8)
