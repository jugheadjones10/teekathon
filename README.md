# Gemini paper crop experiment

This project asks `gemini-3.5-flash` to locate Primary 6 Science questions and
answer-key regions. Gemini returns structured bounding boxes; the local runner
turns those boxes into crops, annotated page images, and an HTML report.

The three PDFs in `papers/` are copied test fixtures. The runner never reads or
writes the original TEEBLOC checkout.

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

## Documentation used

- Gemini bounding boxes: https://ai.google.dev/gemini-api/docs/image-understanding#object-detection
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini 3.5 Flash: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
- PyMuPDF page rendering: https://pymupdf.readthedocs.io/en/latest/recipes-images.html
