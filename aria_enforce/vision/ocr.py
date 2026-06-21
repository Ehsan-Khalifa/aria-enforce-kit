"""
ARIA — License Plate OCR + Mandatory Anonymization
Raw plate text NEVER leaves this module.
"""

from __future__ import annotations
import os
import re
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("aria.ocr")

# Indian plate: GJ-01-AB-1234 or GJ01AB1234
_PLATE_RE = re.compile(r'([A-Z]{2})\s*(\d{2})\s*([A-Z]{1,3})\s*(\d{3,4})')


class PlateReader:
    """
    OCR + anonymization pipeline.

    If PaddleOCR is not installed (e.g. on minimal Pi build),
    falls back to a stub that returns "UNKNOWN" — the rest of the
    system continues to work, violations just won't have plate info.
    """

    def __init__(self):
        self._salt = self._load_salt()
        self._ocr  = self._init_ocr()

    def read_and_anonymize(self, plate_roi) -> dict:
        """
        Args:
            plate_roi: numpy BGR crop of the license plate region
        Returns:
            {
              "anonymized":  "GJ-01-XXXX-A4F2",
              "state_code":  "GJ",
              "valid":       True,
            }
        Raw text is NEVER in the return value.
        """
        raw = self._run_ocr(plate_roi)
        if not raw:
            return {"anonymized": "UNKNOWN", "state_code": "XX", "valid": False}

        match = _PLATE_RE.search(raw.upper().replace(" ", "").replace("-", ""))
        if not match:
            return {"anonymized": "INVALID", "state_code": "XX", "valid": False}

        state, rto, series, number = match.groups()
        identifier = f"{series}{number}"

        hashed = hashlib.blake2b(
            identifier.encode(),
            key=self._salt,
            digest_size=4,
        ).hexdigest().upper()

        return {
            "anonymized": f"{state}-{rto}-XXXX-{hashed}",
            "state_code": state,
            "valid": True,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_ocr(self, roi) -> Optional[str]:
        if self._ocr is None:
            return None
        try:
            result = self._ocr.ocr(roi)
            # PaddleOCR 3.x: result is a list of dicts per page,
            # each with a 'rec_texts' / 'rec_scores' key, or the classic
            # nested list format — handle both.
            if not result:
                return None
            texts = []
            for page in result:
                if isinstance(page, dict):
                    # PaddleOCR 3.x structured output
                    for txt, score in zip(page.get("rec_texts", []),
                                          page.get("rec_scores", [])):
                        if score > 0.65:
                            texts.append(txt)
                elif isinstance(page, list):
                    # Legacy nested list: [[bbox, (text, score)], ...]
                    for line in page:
                        if line and line[1][1] > 0.65:
                            texts.append(line[1][0])
            return " ".join(texts) if texts else None
        except Exception as e:
            logger.debug(f"OCR error: {e}")
        return None

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 3.x: use_doc_orientation_classify replaces use_angle_cls
            try:
                ocr = PaddleOCR(use_doc_orientation_classify=False, lang="en",
                                show_log=False)
            except TypeError:
                # Fallback for older 2.x installs still in the environment
                ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
            logger.info("PaddleOCR loaded.")
            return ocr
        except ImportError:
            logger.warning("PaddleOCR not installed. Plate reading disabled. "
                           "Install with: pip install paddleocr paddlepaddle-gpu")
            return None

    def _load_salt(self) -> bytes:
        salt_hex = os.environ.get("ARIA_PLATE_SALT")
        if not salt_hex:
            # Generate a temporary salt — warn loudly
            import secrets
            salt_hex = secrets.token_hex(32)
            logger.warning(
                "ARIA_PLATE_SALT not set in .env — using a random salt. "
                "Anonymized plates will be inconsistent across restarts. "
                "Set ARIA_PLATE_SALT in .env for production."
            )
        try:
            return bytes.fromhex(salt_hex)
        except ValueError:
            logger.error("ARIA_PLATE_SALT is not valid hex. Using random salt.")
            import secrets
            return secrets.token_bytes(32)
