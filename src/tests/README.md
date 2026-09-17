# AdvancedOpenposeLoader Unit Tests

Cheap unit tests that run locally without GPU or model loading.

## Running Tests

From the repo root:
```bash
cd src/tests
PYTHONPATH=.. python3 -m pytest -v
```

Or run all at once:
```bash
cd src/tests
PYTHONPATH=.. python3 -m pytest test_utils.py test_spatial_fade.py test_node_schema.py -v
```

## Test Coverage

| Test File | What's Tested |
|-----------|---------------|
| `test_utils.py` | Strength defaults, pose cleanup/restoration |
| `test_spatial_fade.py` | Fade mask creation, fade application |
| `test_node_schema.py` | Node INPUT_TYPES, RETURN_TYPES, instantiation |

## Design

- **No real model loading** — all tests use mocks
- **No GPU required** — runs on any Python installation
- **Fast** — completes in <1 second
- **Self-contained** — no external dependencies beyond pytest and torch

## Adding Tests

When adding new functionality:
1. Add corresponding test file in `src/tests/`
2. Mock any external dependencies (comfy modules, torch CUDA)
3. Verify with `PYTHONPATH=.. python3 -m pytest -v`