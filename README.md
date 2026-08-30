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

The project uses the current Gemini Interactions API through
`google-genai==2.20.0`. The API key is read only from the environment or the
gitignored `.env` file.

## Try a small run first

This processes the first three pages of the shortest copied paper:

```bash
uv run crop-papers \
  --paper Science-P6-2024-CA1-Anglo_Chinese-3149.pdf \
  --page-limit 3 \
  --run-name smoke-3-pages

open runs/smoke-3-pages/index.html
```

Each paper report shows the rendered page beside Gemini's bounding-box overlay,
followed by every generated crop. The run directory also contains the validated
JSON response and API usage metadata for each page.

## Process all three copied papers

With no `--paper` arguments, the CLI processes every PDF in `papers/`:

```bash
uv run crop-papers --run-name all-three-papers
open runs/all-three-papers/index.html
```

Requests are intentionally sequential. If a run is interrupted, repeat it with
the same name and `--resume`; successful page responses are reused while failed
pages are retried:

```bash
uv run crop-papers --run-name all-three-papers --resume
```

Useful options:

```text
--thinking-level minimal|low|medium|high   default: medium
--dpi N                                   default: 200
--padding N                               crop padding in pixels; default: 12
--prompt PATH                             default: prompts/crop-v1.txt
--model MODEL_ID                          default: gemini-3.5-flash
```

Generated runs are ignored by Git. The copied source PDFs are committed so the
experiment does not depend on the original TEEBLOC checkout.

## Development checks

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

## Documentation used

- Gemini bounding boxes: https://ai.google.dev/gemini-api/docs/image-understanding#object-detection
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini 3.5 Flash: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
- PyMuPDF page rendering: https://pymupdf.readthedocs.io/en/latest/recipes-images.html
