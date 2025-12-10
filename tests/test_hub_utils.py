"""
PyTest Tests for HubUploader and Related Classes
=================================================
"""

import pytest
from unittest.mock import Mock, patch
from datasets import Dataset, Features, Value

from pagexml_hf.hub_utils import (
    HubUploader,
    ProjectGrouper,
    ReadmeParser,
)


class TestProjectGrouper:
    """Test ProjectGrouper class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with projects."""
        return Dataset.from_dict({
            'project_name': ['alpha', 'beta', 'alpha', 'gamma', 'beta'],
            'data': [1, 2, 3, 4, 5]
        })

    def test_group_by_project(self, sample_dataset):
        """Test grouping dataset by project_name."""
        groups = ProjectGrouper.group_by_project(sample_dataset)

        assert 'alpha' in groups
        assert 'beta' in groups
        assert 'gamma' in groups

        assert groups['alpha'] == [0, 2]
        assert groups['beta'] == [1, 4]
        assert groups['gamma'] == [3]

    def test_group_by_project_with_none(self):
        """Test grouping with None project names."""
        dataset = Dataset.from_dict({
            'project_name': ['alpha', None, '', 'alpha'],
            'data': [1, 2, 3, 4]
        })

        groups = ProjectGrouper.group_by_project(dataset)

        assert 'alpha' in groups
        assert 'unknown_project' in groups

        assert groups['alpha'] == [0, 3]
        assert groups['unknown_project'] == [1, 2]

    def test_group_by_project_single_project(self):
        """Test grouping with single project."""
        dataset = Dataset.from_dict({
            'project_name': ['alpha'] * 5,
            'data': [1, 2, 3, 4, 5]
        })

        groups = ProjectGrouper.group_by_project(dataset)

        assert len(groups) == 1
        assert groups['alpha'] == [0, 1, 2, 3, 4]


class TestReadmeParser:
    """Test ReadmeParser class."""

    @pytest.fixture
    def sample_readme(self):
        """Sample README content."""
        return """---
dataset_info:
  config_name: default
  splits:
  - name: train
    num_examples: 1000
    num_bytes: 500000
  - name: test
    num_examples: 200
    num_bytes: 100000
---

# Dataset Card

## Dataset Summary

This dataset contains 1200 samples.

### Projects Included

- project_alpha
- project_beta
- project_gamma

## Dataset Structure
"""

    def test_parse_splits_from_readme(self, sample_readme):
        """Test parsing split information from README."""
        splits = ReadmeParser.parse_splits_from_readme(sample_readme)

        assert splits == {'train': 1000, 'test': 200}

    def test_parse_splits_from_readme_no_splits(self):
        """Test parsing README without splits."""
        readme = "# Dataset Card\nSome content"
        splits = ReadmeParser.parse_splits_from_readme(readme)

        assert splits == {}

    def test_parse_projects_from_readme(self, sample_readme):
        """Test parsing project list from README."""
        projects = ReadmeParser.parse_projects_from_readme(sample_readme)

        assert projects == ['project_alpha', 'project_beta', 'project_gamma']

    def test_parse_projects_from_readme_no_section(self):
        """Test parsing README without projects section."""
        readme = "# Dataset Card\nSome content"
        projects = ReadmeParser.parse_projects_from_readme(readme)

        assert projects == []

    def test_parse_projects_from_readme_empty_section(self):
        """Test parsing README with empty projects section."""
        readme = """
### Projects Included

## Next Section
"""
        projects = ReadmeParser.parse_projects_from_readme(readme)

        assert projects == []


class TestHubUploader:
    """Test HubUploader class."""

    def test_initialization(self):
        """Test HubUploader initialization."""
        uploader = HubUploader(source_name="test_source")
        assert uploader.source_name == "test_source"

    @patch('pagexml_hf.hub_utils.get_token')
    def test_get_token_from_cache(self, mock_get_token):
        """Test getting token from HuggingFace cache."""
        mock_get_token.return_value = "cached_token"

        token = HubUploader._get_token(None)

        assert token == "cached_token"
        mock_get_token.assert_called_once()

    def test_get_token_from_parameter(self):
        """Test getting token from parameter."""
        token = HubUploader._get_token("my_token")
        assert token == "my_token"

    @patch.dict('os.environ', {'HF_TOKEN': 'env_token'})
    def test_get_token_from_env(self):
        """Test getting token from environment variable."""
        token = HubUploader._get_token(None)
        assert token == "env_token"

    @patch('pagexml_hf.hub_utils.get_token')
    @patch.dict('os.environ', {}, clear=True)
    def test_get_token_raises_error_if_not_found(self, mock_get_token):
        """Test that error is raised if no token found."""
        mock_get_token.return_value = None

        with pytest.raises(ValueError, match="No HuggingFace token found"):
            HubUploader._get_token(None)


class TestFeatureCompatibilityChecker:
    """Test FeatureCompatibilityChecker class."""

    @pytest.fixture
    def compatible_features(self):
        """Create compatible features."""
        return Features({
            'image': Value('string'),
            'text': Value('string'),
            'filename': Value('string'),
        })

    @pytest.fixture
    def incompatible_features(self):
        """Create incompatible features."""
        return Features({
            'image': Value('string'),
            'text': Value('string'),
            'extra_field': Value('int32'),
        })


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
