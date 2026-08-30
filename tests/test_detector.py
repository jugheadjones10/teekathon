import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from gemini_paper_crop.detector import GeminiDetector


class FakeInteractions:
    def __init__(self) -> None:
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="interaction-123",
            output_text=json.dumps(
                {
                    "page_number": 99,
                    "regions": [
                        {
                            "type": "mcq_question",
                            "question_number": "1",
                            "fragment_index": 1,
                            "box_2d": [100, 100, 900, 900],
                            "needs_review": False,
                        }
                    ],
                }
            ),
            usage=SimpleNamespace(model_dump=lambda mode: {"total_tokens": 123}),
        )


def test_detector_sends_high_resolution_image_and_structured_schema(
    tmp_path: Path,
) -> None:
    page_path = tmp_path / "page.png"
    Image.new("RGB", (20, 30), "white").save(page_path)
    interactions = FakeInteractions()
    client = SimpleNamespace(interactions=interactions)
    detector = GeminiDetector(client=client, model="gemini-3.5-flash")

    call = detector.detect_page(page_path, page_number=3, prompt="Find regions")

    assert call.detections.page_number == 3
    assert call.interaction_id == "interaction-123"
    assert call.usage == {"total_tokens": 123}
    assert interactions.request is not None
    assert interactions.request["model"] == "gemini-3.5-flash"
    assert interactions.request["generation_config"] == {"thinking_level": "medium"}
    image_input = interactions.request["input"][0]
    assert image_input["type"] == "image"
    assert image_input["mime_type"] == "image/png"
    assert image_input["resolution"] == "high"
    assert image_input["data"]
    assert interactions.request["input"][1]["text"].endswith("Page number: 3")
    assert interactions.request["response_format"]["mime_type"] == ("application/json")
