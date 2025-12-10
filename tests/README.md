# Testing Documentation

This directory contains PyTest tests for the pagexml-hf package.

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and pytest configuration
├── test_converter.py     # Tests for XmlConverter class
├── test_exporters.py     # Tests for all Exporter classes
├── test_hub_utils.py     # Tests for HubUploader and related classes
├── test_imageutils.py    # Tests for ImageProcessor class
└── test_parser.py        # Tests for XmlParser class (existing)
```

## Running Tests

### Run all tests:

```bash
pytest
```

### Run with verbose output:

```bash
pytest -v
```

### Run specific test file:

```bash
pytest tests/test_imageutils.py
```

### Run specific test class:

```bash
pytest tests/test_imageutils.py::TestImageProcessor
```

### Run specific test:

```bash
pytest tests/test_imageutils.py::TestImageProcessor::test_initialization_default
```

## Test Categories

Tests are organized into several categories:

### 1. ImageProcessor Tests (`test_imageutils.py`)

- Image loading and encoding
- Cropping without mask
- Cropping with mask (polygon masking)
- Size filters (min_width, min_height)
- Edge cases (invalid coords, empty coords)

### 2. Exporter Tests (`test_exporters.py`)

- POST_FEATURES verification for all exporters
- Initialization tests
- Export mode registration
- Window exporter overlap validation

### 3. HubUploader Tests (`test_hub_utils.py`)

- Project grouping
- README parsing (splits and projects)
- Token retrieval
- Feature compatibility checking

### 4. Converter Tests (`test_converter.py`)

- Initialization with different source types
- Export mode validation
- Feature compatibility checking
- Statistics computation

### 5. Parser Tests (`test_parser.py`)

- XML parsing (existing tests)
- Coordinate parsing
- Reading order extraction
