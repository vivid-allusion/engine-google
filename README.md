# engine-google — Google AI Engine for studiolot

Engine wrapping the [Google Generative AI](https://ai.google.dev) API
(google-genai SDK) behind studiolot's uniform Engine interface.
Generators call this Engine instead of importing the google-genai SDK
directly. Supports Gemini models for text generation and vision tasks.

## Quick Start

```bash
# Clone into a studiolot project
git clone https://github.com/vivid-allusion/engine-google.git \
  00_APPLICATIONS/ENGINES/engine-google/

# Or pip install
pip install git+https://github.com/vivid-allusion/engine-google.git
```

## Usage

```python
from engine_google import Engine, InputFile
from pathlib import Path

profile = {
    "platform": "google",
    "endpoint": "gemini-2.0-flash",
    "media_type": "text",
}

engine = Engine(profile, "/tmp/output", api_key="AIza...")
results = engine.run([InputFile(path=Path("prompt.md"), prompt="Summarize AI")])

for r in results:
    print(r.status, r.path)
```

## API Key

Set `GOOGLE_API_KEY` in your environment or `.env` file:

```bash
export GOOGLE_API_KEY=AIza...
```

Get a key at https://aistudio.google.com/apikey

## Repo Structure

```
engine-google/
├── engine.py          ← Engine class wrapping google-genai SDK
├── datatypes.py       ← InputFile, OutputFile, ProgressEvent, EngineError
├── metadata.py        ← Zero-dependency identity constants
├── __init__.py        ← Re-exports
├── pyproject.toml     ← pip install definition
├── requirements.txt   ← google-genai>=1.0
├── endpoints/
│   ├── IMG-Models/    ← .gitkeep (TOML catalog deferred to follow-up batch)
│   ├── VID-Models/    ← .gitkeep
│   ├── TXT-Models/    ← .gitkeep
│   └── Vision-Models/ ← .gitkeep
└── tests/
    └── test_engine.py ← 20 unit tests
```

## Note

The endpoint TOML catalog (model definitions) is deferred to a follow-up
batch. The Engine scaffold is fully functional — TOML files can be dropped
into `endpoints/` at any time without code changes.

## Contract

See [ENGINE_CONTRACT.md](https://github.com/vivid-allusion/studiolot/blob/main/docs/architecture/ENGINE_CONTRACT.md)
in the studiolot repo for the full Engine specification.
