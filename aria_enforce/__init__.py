"""
ARIA-Enforce — Image-mode traffic violation detection, classification,
evidence generation, e-challan issuance and analytics.

This package is the Round-2 submission surface. It REUSES the production
ARIA modules (detector, ocr, weather) but adds an image-first path so a
reviewer can run everything from a single uploaded photo — no Docker,
no MQTT, no database server required.

Design rules:
  * Never import or require the live-video / MQTT / RL stack.
  * Degrade gracefully: if the helmet/seatbelt model weights are missing,
    those checks are skipped (not crashed) and clearly marked "model pending".
  * Everything is single-image first; video is an optional batch loop.
"""

__all__ = [
    "config",
    "severity",
    "challan",
    "rider_model",
    "image_violations",
]
