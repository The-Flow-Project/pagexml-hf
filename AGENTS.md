# AGENTS.md

## Project Overview

`pagexml-hf` converts PAGE XML data (from Transkribus or similar) + images into HuggingFace `datasets.Dataset` objects. It supports ZIP archives, local folders, and existing HF datasets as input, with 5 export modes: `raw_xml`, `text`, `region`, `line`, `window`. Fork of `transkribus-hf`.

## Architecture & Data Flow

```
Input (ZIP / folder / HF dataset)
  → XmlParser (parser.py)          — generators yielding row dicts
  → XmlConverter (converter.py)    — builds base Dataset via Dataset.from_generator, delegates to exporter
  → *Exporter (exporters.py)       — Dataset.map() transforms rows into final schema
  → HubUploader (hub_utils.py)     — uploads parquet shards to HF Hub
```

- **`XmlParser`** returns *generators* (`parse_zip`, `parse_folder`, `parse_dataset`). The converter stores the generator function + kwargs and replays them via `Dataset.from_generator`.
- **`XmlConverter`** takes `gen_func` and `gen_kwargs` (not a live generator). It also provides `convert_and_upload()` as a convenience method combining conversion + Hub upload.
- **`PRE_FEATURES`** (defined in both `converter.py` and `exporters.py`) is the intermediate Arrow schema. Each exporter defines its own **`POST_FEATURES`** — the final HF dataset schema. Feature compatibility is checked before upload when `append=True`.
- **`ImageProcessor`** (`imageutils.py`) handles EXIF orientation fixes, PNG encoding, polygon-masked cropping via `scikit-image`, and random image augmentation (rotation, dilation, erosion, downscaling) for line-level data augmentation.
- **`hub_utils.py`** contains several helper classes beyond `HubUploader`: `ProjectGrouper` (groups samples by project), `ReadmeGenerator`/`ReadmeParser` (create/update dataset cards), `YamlGenerator`/`FeatureYamlConverter` (YAML frontmatter for HF Hub). Parquet shards are organized as `data/<split>/<project_name>/<timestamp>-<shard>.parquet`.
- Dataset caching is explicitly disabled (`disable_caching()` in `converter.py`).
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` is set globally in `exporters.py` to handle incomplete image files.

## Key Conventions

- **Python version**: Requires `>=3.10` (uses `X | Y` union syntax throughout).
- **Logging**: Use `loguru.logger` everywhere (not stdlib `logging`). Logger setup is in `logger.py`; writes to `logs/` directory.
- **Image format**: All images are stored as `{"bytes": <png_bytes>, "path": None}` dicts matching HF `datasets.Image(decode=False)`. `ImageProcessor.encode_image()` outputs PNG.
- **Namespace handling**: PAGE XML namespaces are auto-detected per document via `root.nsmap` in `_parse_page_xml` — never hardcode a namespace URI.
- **Encoding**: `chardet` is used for fallback encoding detection of XML files (`_decode_bytes` in `parser.py`).
- **Version**: Single source of truth is `__version__` in `pagexml_hf/__init__.py`, read dynamically by setuptools via `pyproject.toml`.
- **Line augmentation**: `LineExporter` supports `line_augment` parameter (max 5) to produce random augmented copies per line via `ImageProcessor.random_augment_image()`.

## Development Workflow

```bash
# Install (editable)
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"

# Run tests
pytest                         # all tests
pytest -v tests/test_parser.py # single module
pytest -m "not slow"           # skip slow/integration tests

# CLI entry point
pagexml-hf <source> --mode line --local-only --output-dir ./out
# or: python -m pagexml_hf <source> ...
```

## Adding a New Export Mode

1. Create a new class in `exporters.py` inheriting `BaseExporter`.
2. Define `POST_FEATURES` as a class attribute (HF `Features` object).
3. Implement `process_dataset(self, dataset) -> Dataset` using `dataset.map(fn, batched=True, features=self.POST_FEATURES, remove_columns=...)`.
4. Register the class in `XmlConverter.EXPORT_MODES` dict in `converter.py`.
5. Add the mode string to the CLI `--mode` choices in `cli.py`.
6. Export it from `__init__.py` and add to `__all__`.

## Testing Patterns

- Shared fixtures live in `tests/conftest.py` — `sample_xml_content`, `sample_image_bytes`, `sample_dataset_dict`, `sample_coordinates`, `temp_dir` provide reusable PAGE XML and image data.
- Tests use `pytest` markers: `slow`, `integration`, `unit`.
- HF Hub and network calls are mocked (`unittest.mock.patch`); no real uploads in tests.
- Each module has a matching `tests/test_<module>.py` file.

## Important Gotchas

- `parse_xml=True` must be passed to parser generators when the export mode needs regions/lines (all modes except `raw_xml`). The CLI handles this automatically via `args.mode`.
- `PRE_FEATURES` is duplicated in `converter.py` and `exporters.py` — keep both in sync if changed.
- The `SourcePathAction` in `cli.py` auto-detects input type (HF dataset, URL, local zip, folder) and returns a `(path, type)` tuple — don't bypass this when extending CLI args.

