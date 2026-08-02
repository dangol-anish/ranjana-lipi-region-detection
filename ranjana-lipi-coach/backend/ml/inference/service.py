"""Object-oriented facade for the runtime ML inference pipeline."""

from __future__ import annotations

from typing import Any

from ml.inference.pipeline import analyze_attempt


class RanjanaInferenceService:
    """Service-class wrapper around the validated inference pipeline.

    The core image processing, model loading, reconstruction, and feedback
    algorithms remain in pipeline.py. This facade gives the backend an OOP
    boundary while preserving the already-tested functional implementation.
    """

    def analyze_practice_attempt(
        self,
        image_bytes: bytes,
        target_class: str,
        rows: int = 3,
        cols: int = 3,
        device_name: str = "cpu",
    ) -> dict[str, Any]:
        return analyze_attempt(
            image_bytes=image_bytes,
            target_class=target_class,
            rows=rows,
            cols=cols,
            device_name=device_name,
        )


inference_service = RanjanaInferenceService()
