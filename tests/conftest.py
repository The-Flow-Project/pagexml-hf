"""
PyTest Configuration and Shared Fixtures
=========================================

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import io


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format='JPEG')
    return img_bytes_io.getvalue()


@pytest.fixture
def sample_xml_content():
    """Sample PageXML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <ReadingOrder>
            <OrderedGroup id="ro_test">
                <RegionRefIndexed index="0" regionRef="region_1"/>
            </OrderedGroup>
        </ReadingOrder>
        <TextRegion type="paragraph" id="region_1">
            <Coords points="10,10 200,10 200,100 10,100"/>
            <TextLine id="line_1" custom="readingOrder {index:0;}">
                <Coords points="20,20 180,20 180,40 20,40"/>
                <Baseline points="20,35 180,35"/>
                <TextEquiv>
                    <Unicode>Sample text line</Unicode>
                </TextEquiv>
            </TextLine>
            <TextEquiv>
                <Unicode>Sample text line</Unicode>
            </TextEquiv>
        </TextRegion>
    </Page>
</PcGts>"""


@pytest.fixture
def sample_coordinates():
    """Sample polygon coordinates for testing."""
    return [(10, 10), (100, 10), (100, 100), (10, 100)]


@pytest.fixture
def sample_dataset_dict():
    """Sample dataset dictionary for testing."""
    return {
        'image': [{'bytes': b'fake_image_1', 'path': None}],
        'xml_content': ['<xml>content1</xml>'],
        'filename': ['file1.xml'],
        'project_name': ['test_project'],
        'image_width': [1000],
        'image_height': [1500],
        'regions': [[]],
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

