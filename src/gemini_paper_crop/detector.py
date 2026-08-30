import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai

from gemini_paper_crop.models import PageDetections, gemini_response_schema


@dataclass(frozen=True)
class DetectionCall:
    detections: PageDetections
    raw_text: str
    interaction_id: str | None
    usage: dict[str, Any] | None
    elapsed_seconds: float


class GeminiDetector:
    def __init__(
        self,
        *,
        model: str = "gemini-3.5-flash",
        thinking_level: str = "medium",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.client = client or genai.Client()

    def detect_page(
        self,
        page_path: Path,
        *,
        page_number: int,
        prompt: str,
    ) -> DetectionCall:
        encoded_image = base64.b64encode(page_path.read_bytes()).decode("ascii")
        started = time.monotonic()
        interaction = self.client.interactions.create(
            model=self.model,
            input=[
                {
                    "type": "image",
                    "data": encoded_image,
                    "mime_type": "image/png",
                    "resolution": "high",
                },
                {
                    "type": "text",
                    "text": f"{prompt.rstrip()}\n\nPage number: {page_number}",
                },
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(),
            },
            generation_config={"thinking_level": self.thinking_level},
        )
        elapsed = time.monotonic() - started

        raw_text = interaction.output_text
        if not raw_text:
            raise RuntimeError("Gemini returned no text output")
        detections = PageDetections.model_validate_json(raw_text).model_copy(
            update={"page_number": page_number}
        )
        usage = None
        if interaction.usage is not None:
            usage = interaction.usage.model_dump(mode="json")
        return DetectionCall(
            detections=detections,
            raw_text=raw_text,
            interaction_id=interaction.id or None,
            usage=usage,
            elapsed_seconds=elapsed,
        )
