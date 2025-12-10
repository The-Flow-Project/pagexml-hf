"""
PyTest Tests for Exporters
===========================
"""

import pytest
from datasets import Features

from pagexml_hf.exporters import (
    RawXMLExporter,
    TextExporter,
    RegionExporter,
    LineExporter,
    WindowExporter,
)


class TestExporterFeatures:
    """Test that all exporters have correct POST_FEATURES defined."""

    def test_raw_xml_exporter_has_post_features(self):
        """Test RawXMLExporter has POST_FEATURES."""
        assert hasattr(RawXMLExporter, 'POST_FEATURES')
        assert isinstance(RawXMLExporter.POST_FEATURES, Features)

        expected_keys = {'image', 'xml_content', 'filename', 'project_name'}
        assert set(RawXMLExporter.POST_FEATURES.keys()) == expected_keys

    def test_text_exporter_has_post_features(self):
        """Test TextExporter has POST_FEATURES."""
        assert hasattr(TextExporter, 'POST_FEATURES')
        assert isinstance(TextExporter.POST_FEATURES, Features)

        expected_keys = {'image', 'text', 'filename', 'project_name'}
        assert set(TextExporter.POST_FEATURES.keys()) == expected_keys

    def test_region_exporter_has_post_features(self):
        """Test RegionExporter has POST_FEATURES."""
        assert hasattr(RegionExporter, 'POST_FEATURES')
        assert isinstance(RegionExporter.POST_FEATURES, Features)

        expected_keys = {
            'image', 'text', 'region_id', 'region_reading_order',
            'region_type', 'filename', 'project_name'
        }
        assert set(RegionExporter.POST_FEATURES.keys()) == expected_keys

    def test_line_exporter_has_post_features(self):
        """Test LineExporter has POST_FEATURES."""
        assert hasattr(LineExporter, 'POST_FEATURES')
        assert isinstance(LineExporter.POST_FEATURES, Features)

        expected_keys = {
            'image', 'text', 'line_id', 'line_reading_order',
            'region_id', 'region_reading_order', 'region_type',
            'filename', 'project_name'
        }
        assert set(LineExporter.POST_FEATURES.keys()) == expected_keys

    def test_window_exporter_has_post_features(self):
        """Test WindowExporter has POST_FEATURES."""
        assert hasattr(WindowExporter, 'POST_FEATURES')
        assert isinstance(WindowExporter.POST_FEATURES, Features)

        expected_keys = {
            'image', 'text', 'window_size', 'window_index',
            'line_ids', 'line_reading_order', 'filename', 'project_name'
        }
        assert set(WindowExporter.POST_FEATURES.keys()) == expected_keys


class TestExporterInitialization:
    """Test exporter initialization."""

    def test_raw_xml_exporter_init(self):
        """Test RawXMLExporter initialization."""
        exporter = RawXMLExporter(batch_size=64)
        assert exporter.batch_size == 64

    def test_text_exporter_init(self):
        """Test TextExporter initialization."""
        exporter = TextExporter(batch_size=128)
        assert exporter.batch_size == 128

    def test_region_exporter_init(self):
        """Test RegionExporter initialization."""
        exporter = RegionExporter(batch_size=32)
        assert exporter.batch_size == 32

    def test_line_exporter_init(self):
        """Test LineExporter initialization."""
        exporter = LineExporter(batch_size=16)
        assert exporter.batch_size == 16

    def test_window_exporter_init_default(self):
        """Test WindowExporter initialization with defaults."""
        exporter = WindowExporter(batch_size=32, window_size=2, overlap=0)
        assert exporter.batch_size == 32
        assert exporter.window_size == 2
        assert exporter.overlap == 0

    def test_window_exporter_init_custom(self):
        """Test WindowExporter initialization with custom values."""
        exporter = WindowExporter(batch_size=64, window_size=5, overlap=2)
        assert exporter.batch_size == 64
        assert exporter.window_size == 5
        assert exporter.overlap == 2

    def test_window_exporter_invalid_overlap(self):
        """Test WindowExporter raises error if overlap >= window_size."""
        with pytest.raises(ValueError, match="Overlap must be less than window size"):
            WindowExporter(window_size=2, overlap=2)

        with pytest.raises(ValueError, match="Overlap must be less than window size"):
            WindowExporter(window_size=2, overlap=3)


class TestExporterModes:
    """Test that exporter modes are correctly registered."""

    def test_export_modes_dict(self):
        """Test that EXPORT_MODES contains all exporters."""
        from pagexml_hf.converter import XmlConverter

        expected_modes = {
            'raw_xml': RawXMLExporter,
            'text': TextExporter,
            'region': RegionExporter,
            'line': LineExporter,
            'window': WindowExporter,
        }

        assert XmlConverter.EXPORT_MODES == expected_modes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

