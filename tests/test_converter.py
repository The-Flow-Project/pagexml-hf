"""
PyTest Tests for XmlConverter
==============================
"""

import pytest
from unittest.mock import patch
from datasets import Dataset

from pagexml_hf.converter import XmlConverter


class TestXmlConverterInitialization:
    """Test XmlConverter initialization."""

    @pytest.fixture
    def mock_gen_func(self):
        """Mock generator function."""

        def gen():
            """
            Generator function for sample data.
            """
            yield {
                'image': {'bytes': b'fake_image', 'path': None},
                'xml_content': '<xml></xml>',
                'filename': 'test.xml',
                'project_name': 'test_project',
                'image_width': 1000,
                'image_height': 1500,
                'regions': [],
            }

        return gen

    def test_initialization_with_local_path(self, mock_gen_func):
        """Test initialization with local path."""
        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
            source_path="/path/to/data.zip",
            source_type="zip",
        )

        assert converter.source_name == "data.zip"
        assert converter.gen_func == mock_gen_func
        assert converter.gen_kwargs == {}

    def test_initialization_with_huggingface_path(self, mock_gen_func):
        """Test initialization with HuggingFace dataset path."""
        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
            source_path="user/dataset",
            source_type="huggingface",
        )

        assert converter.source_name == "user/dataset"

    def test_initialization_without_path(self, mock_gen_func):
        """Test initialization without source path."""
        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
            source_path=None,
            source_type=None,
        )

        assert converter.source_name == "unknown_source"

    def test_export_modes_available(self):
        """Test that all export modes are registered."""
        assert 'raw_xml' in XmlConverter.EXPORT_MODES
        assert 'text' in XmlConverter.EXPORT_MODES
        assert 'region' in XmlConverter.EXPORT_MODES
        assert 'line' in XmlConverter.EXPORT_MODES
        assert 'window' in XmlConverter.EXPORT_MODES


class TestXmlConverterConvert:
    """Test XmlConverter convert method."""

    @pytest.fixture
    def mock_gen_func(self):
        """Mock generator function with sample data."""

        def gen():
            """
            Generator function for sample data with regions and lines.
            """
            yield {
                'image': {'bytes': b'fake_image', 'path': None},
                'xml_content': '<xml></xml>',
                'filename': 'test.xml',
                'project_name': 'test_project',
                'image_width': 1000,
                'image_height': 1500,
                'regions': [
                    {
                        'id': 'r1',
                        'type': 'paragraph',
                        'coords': [(0, 0), (100, 0), (100, 100), (0, 100)],
                        'text_lines': [
                            {
                                'id': 'l1',
                                'text': 'Test line',
                                'coords': [(10, 10), (90, 10), (90, 30), (10, 30)],
                                'baseline': [(10, 20), (90, 20)],
                                'reading_order': 0,
                                'region_id': 'r1',
                            }
                        ],
                        'reading_order': 0,
                        'full_text': 'Test line',
                    }
                ],
            }

        return gen

    def test_convert_invalid_mode(self, mock_gen_func):
        """Test that convert raises error for invalid mode."""
        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
        )

        with pytest.raises(ValueError, match="Invalid export mode"):
            converter.convert(export_mode="invalid_mode")

    def test_convert_valid_modes(self, mock_gen_func):
        """Test that all valid modes are accepted."""
        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
        )

        # These should not raise errors (though they may fail during actual processing)
        valid_modes = ['raw_xml', 'text', 'region', 'line', 'window']
        for mode in valid_modes:
            try:
                # Just check that the mode is validated
                converter.convert(export_mode=mode, batch_size=1)
            except Exception as e:
                # May fail during processing, but should pass mode validation
                if "Invalid export mode" in str(e):
                    pytest.fail(f"Mode {mode} should be valid")


class TestXmlConverterFeatureCompatibility:
    """Test feature compatibility checking."""

    @pytest.fixture
    def mock_gen_func(self):
        """Mock generator function."""

        def gen():
            """
            Test generator function for feature compatibility.
            """
            yield {
                'image': {'bytes': b'fake', 'path': None},
                'xml_content': '<xml></xml>',
                'filename': 'test.xml',
                'project_name': 'test',
                'image_width': 1000,
                'image_height': 1500,
                'regions': [],
            }

        return gen

    @patch('pagexml_hf.converter.HubUploader')
    @patch('pagexml_hf.converter.repo_exists')
    @patch('pagexml_hf.converter.load_dataset')
    def test_check_exporter_feature_compatibility_repo_not_exists(
            self,
            mock_repo_exists,
            mock_gen_func,
    ):
        """Test feature check when repo doesn't exist."""
        mock_repo_exists.return_value = False

        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
        )

        # Should not raise error if repo doesn't exist
        try:
            converter._check_exporter_feature_compatibility(
                repo_id="test/repo",
                export_mode="text",
                token="fake_token",
            )
        except Exception as e:
            pytest.fail(f"Should not raise error: {e}")

    @patch('pagexml_hf.converter.HubUploader._get_token')
    def test_check_exporter_feature_compatibility_gets_token(
            self,
            mock_get_token,
            mock_gen_func,
    ):
        """Test that feature check gets token correctly."""
        mock_get_token.return_value = "token123"

        converter = XmlConverter(
            gen_func=mock_gen_func,
            gen_kwargs={},
        )

        with patch('pagexml_hf.converter.repo_exists') as mock_repo_exists:
            mock_repo_exists.return_value = False

            converter._check_exporter_feature_compatibility(
                repo_id="test/repo",
                export_mode="text",
                token=None,
            )

            mock_get_token.assert_called_once()


class TestXmlConverterStatistics:
    """Test statistics computation."""

    def test_compute_stats_no_regions(self):
        """Test computing stats for dataset without regions."""

        def gen():
            """
            Test generator function with no regions.
            """
            yield {
                'image': {'bytes': b'fake', 'path': None},
                'xml_content': '<xml></xml>',
                'filename': 'test1.xml',
                'project_name': 'project_a',
                'image_width': 1000,
                'image_height': 1500,
                'regions': [],
            }
            yield {
                'image': {'bytes': b'fake', 'path': None},
                'xml_content': '<xml></xml>',
                'filename': 'test2.xml',
                'project_name': 'project_b',
                'image_width': 1000,
                'image_height': 1500,
                'regions': [],
            }

        converter = XmlConverter(gen_func=gen, gen_kwargs={})

        # Create dataset manually to test stats
        dataset = Dataset.from_dict({
            'regions': [[], []],
            'project_name': ['project_a', 'project_b'],
        })

        converter._compute_stats(dataset)

        assert converter.stats_cache is not None
        assert converter.stats_cache['total_pages'] == 2
        assert converter.stats_cache['total_regions'] == 0
        assert converter.stats_cache['total_lines'] == 0
        assert set(converter.stats_cache['projects']) == {'project_a', 'project_b'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
