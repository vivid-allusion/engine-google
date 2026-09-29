## Architecture: Engine SDK Wrapper

This repo is an **Engine** in the studiolot ecosystem. It wraps the
**Google AI** API and exposes a uniform interface that Vehicles call.

### The layers

```
studiolot (TUI) → Vehicle (script) → **Google AI Engine** → Provider API
```

Vehicles like Frame Composer and Motion Conductor are SDK-agnostic. They
discover this Engine, load it via `engine_loader.py`, and call
`engine.run(inputs)`. This Engine handles all **Google AI**-specific logic.

### Contract

- **`Engine.__init__` is PURE.** No network I/O, no SDK imports in the
  constructor. It only stores: profile dict, output_dir, api_key,
  on_progress callback.
- **`Engine.run(inputs: list[InputFile]) -> list[OutputFile]`** is the ONLY
  entry point Vehicles call. Returns ALL results — success and failure —
  as OutputFile objects. Never raises for per-Markdown-file failures.
- **`EngineError`** is for unrecoverable pre-flight failures only: missing
  API key, invalid profile, provider auth rejection.
- **`datatypes.py`** defines InputFile, OutputFile, ProgressEvent,
  EngineError. Single source of truth per Engine.
- **`metadata.py`** is zero-dependency (stdlib only). studiolot imports this
  (NOT engine.py) to read PROVIDER_NAME for the TUI label. This avoids
  pulling in the **Google AI** SDK transitively.
- **`endpoints/`** TOMLs are the canonical model catalog. One file per model,
  defining valid parameter ranges.
- **`profiles/standby/`** contains publishable YAML profiles. When installed,
  these are seeded into Vehicle repos' `USER-FILES/02.STANDBY/`.

### Provider details

| Field | Value |
|-------|-------|
| **Platform** | `google` |
| **API key env var** | `GOOGLE_API_KEY` |
| **Key pattern** | `^AIza[A-Za-z0-9_-]{30,}$` |
| **Homepage** | https://ai.google.dev |

### Engine discovery

Vehicles find this Engine via `engine_loader.py`:
- Repo directory: `engine-google` 
- Python package: `engine_google`
- `engine_loader.py` handles the hyphen→underscore mapping
- Discovery order: local clone in `00_APPLICATIONS/ENGINES/` first, pip-installed package as fallback

### Source File Map

| File | Purpose |
|------|---------|
| `engine_google/engine.py` | Engine implementation — all SDK calls live here |
| `engine_google/datatypes.py` | InputFile, OutputFile, ProgressEvent, EngineError |
| `engine_google/metadata.py` | Zero-dependency identity constants (studiolot reads this) |
| `engine_google/__init__.py` | Re-exports Engine + all datatypes |
| `engine_google/endpoints/` | TOML model catalog (IMG/VID/TXT/Vision) |
| `engine_google/profiles/standby/` | Publishable YAML profiles |
| `tests/test_engine.py` | Unit tests |
| `pyproject.toml` | Package metadata, pip-installable |
| `requirements.txt` | Provider SDK + dependencies |

### Reference

Full contract: `~/Nextcloud/00-DEVELOPMENT/MISC_DEV_TOOLS/studiolot/docs/architecture/ENGINE_CONTRACT.md`

### Session History

- 2026-08-05 — Created AGENTS.md, .env.example, .gitignore
