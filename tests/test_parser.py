"""
PyTest Tests for XmlParser
===========================
"""

import pytest
import xml.etree.ElementTree as ET
from typing import Dict

from pagexml_hf.parser import XmlParser


class TestXmlParser:
    """Test cases for XmlParser."""

    @pytest.fixture
    def parser(self):
        """Create XmlParser instance."""
        return XmlParser()

    @pytest.fixture
    def sample_xml(self):
        """Sample PageXML content for testing."""
        return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1247" imageHeight="1920">
        <ReadingOrder>
            <OrderedGroup id="ro_test">
                <RegionRefIndexed index="0" regionRef="region_1"/>
                <RegionRefIndexed index="1" regionRef="region_2"/>
            </OrderedGroup>
        </ReadingOrder>
        <TextRegion type="paragraph" id="region_1">
            <Coords points="5,3 100,3 100,50 5,50"/>
            <TextLine id="line_1" custom="readingOrder {index:0;}">
                <Coords points="10,10 90,10 90,30 10,30"/>
                <Baseline points="10,20 90,20"/>
                <TextEquiv>
                    <Unicode>Test line 1</Unicode>
                </TextEquiv>
            </TextLine>
            <TextLine id="line_2" custom="readingOrder {index:1;}">
                <Coords points="10,35 90,35 90,45 10,45"/>
                <Baseline points="10,40 90,40"/>
                <TextEquiv>
                    <Unicode>Test line 2</Unicode>
                </TextEquiv>
            </TextLine>
            <TextEquiv>
                <Unicode>Test line 1
Test line 2</Unicode>
            </TextEquiv>
        </TextRegion>
    </Page>
</PcGts>"""

    def test_parser_initialization(self, parser):
        """Test XmlParser initialization."""
        assert parser.namespace is not None
        assert 'pc' in parser.namespace
        assert parser.namespace['pc'].endswith('pagecontent/2013-07-15')

    def test_parse_coords(self, parser, sample_xml):
        """Test coordinate parsing."""
        root = ET.fromstring(sample_xml)
        coords_elem = root.find(".//pc:Coords", parser.namespace)

        coords = parser._parse_coords(coords_elem)
        expected = [(5, 3), (100, 3), (100, 50), (5, 50)]

        assert coords == expected

    def test_parse_coords_with_spaces(self, parser):
        """Test coordinate parsing with different spacing."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Coords points="  10 , 20   30,40  50 , 60  "/>
</PcGts>"""
        root = ET.fromstring(xml)
        coords_elem = root.find(".//pc:Coords", parser.namespace)

        coords = parser._parse_coords(coords_elem)
        expected = [(10, 20), (30, 40), (50, 60)]

        assert coords == expected

    def test_parse_coords_none(self, parser):
        """Test coordinate parsing with None element."""
        coords = parser._parse_coords(None)
        assert coords == []

    def test_parse_reading_order(self, parser, sample_xml):
        """Test reading order parsing."""
        root = ET.fromstring(sample_xml)
        reading_order = parser._parse_reading_order(root)

        expected = {"region_1": 0, "region_2": 1}
        assert reading_order == expected

    def test_parse_reading_order_no_order(self, parser):
        """Test parsing when no reading order is present."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="100" imageHeight="100">
    </Page>
</PcGts>"""
        root = ET.fromstring(xml)
        reading_order = parser._parse_reading_order(root)

        assert reading_order == {}

    def test_get_text_equiv(self, parser, sample_xml):
        """Test text extraction."""
        root = ET.fromstring(sample_xml)
        region_elem = root.find(".//pc:TextRegion", parser.namespace)

        text = parser._get_text_equiv(region_elem)
        expected = "Test line 1\nTest line 2"

        assert text == expected

    def test_get_text_equiv_none(self, parser):
        """Test text extraction with None element."""
        text = parser._get_text_equiv(None)
        assert text == ""

    def test_get_text_equiv_empty(self, parser):
        """Test text extraction with empty Unicode element."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <TextRegion>
        <TextEquiv>
            <Unicode></Unicode>
        </TextEquiv>
    </TextRegion>
</PcGts>"""
        root = ET.fromstring(xml)
        region_elem = root.find(".//pc:TextRegion", parser.namespace)

        text = parser._get_text_equiv(region_elem)
        assert text == ""

    def test_parse_page_xml(self, parser, sample_xml):
        """Test complete page XML parsing."""
        page_data = parser._parse_page_xml(sample_xml)

        assert isinstance(page_data, Dict)
        assert page_data["image_filename"] == "test.jpg"
        assert page_data["image_width"] == 1247
        assert page_data["image_height"] == 1920
        assert page_data["project_name"] == "test_project"
        assert len(page_data["regions"]) == 1

        region = page_data["regions"][0]
        assert region.id == "region_1"
        assert region.type == "paragraph"
        assert len(region.text_lines) == 2

        line1 = region.text_lines[0]
        assert line1.id == "line_1"
        assert line1.text == "Test line 1"
        assert line1.reading_order == 0
        assert line1.region_id == "region_1"

    def test_parse_page_xml_with_baseline(self, parser, sample_xml):
        """Test that baseline is correctly parsed."""
        page_data = parser._parse_page_xml(sample_xml)

        line1 = page_data["regions"][0].text_lines[0]
        assert line1.baseline == [(10, 20), (90, 20)]

        line2 = page_data["regions"][0].text_lines[1]
        assert line2.baseline == [(10, 40), (90, 40)]

    def test_macos_metadata_file_filtering(self, parser):
        """Test that macOS metadata files are properly filtered out."""
        # Test files that should be filtered out
        assert parser._is_macos_metadata_file("__MACOSX/file.xml") is True
        assert parser._is_macos_metadata_file("._file.xml") is True
        assert parser._is_macos_metadata_file("project/._page.xml") is True
        assert parser._is_macos_metadata_file("project/.DS_Store") is True
        assert parser._is_macos_metadata_file(".DS_Store") is True

        # Test files that should NOT be filtered out
        assert parser._is_macos_metadata_file("project/page/file.xml") is False
        assert parser._is_macos_metadata_file("project/page/valid.xml") is False
        assert parser._is_macos_metadata_file("normal_file.xml") is False
        assert parser._is_macos_metadata_file("data.xml") is False

    def test_parse_page_xml_multiple_regions(self, parser):
        """Test parsing with multiple regions."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <ReadingOrder>
            <OrderedGroup id="ro">
                <RegionRefIndexed index="0" regionRef="r1"/>
                <RegionRefIndexed index="1" regionRef="r2"/>
            </OrderedGroup>
        </ReadingOrder>
        <TextRegion type="paragraph" id="r1">
            <Coords points="10,10 100,10 100,50 10,50"/>
            <TextEquiv><Unicode>Region 1</Unicode></TextEquiv>
        </TextRegion>
        <TextRegion type="heading" id="r2">
            <Coords points="10,60 100,60 100,100 10,100"/>
            <TextEquiv><Unicode>Region 2</Unicode></TextEquiv>
        </TextRegion>
    </Page>
</PcGts>"""

        page_data = parser._parse_page_xml(xml)

        assert len(page_data["regions"]) == 2
        assert page_data["regions"][0].id == "r1"
        assert page_data["regions"][0].type == "paragraph"
        assert page_data["regions"][0].reading_order == 0
        assert page_data["regions"][1].id == "r2"
        assert page_data["regions"][1].type == "heading"
        assert page_data["regions"][1].reading_order == 1

    def test_parse_page_xml_empty_region(self, parser):
        """Test parsing region with no text lines."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion type="paragraph" id="r1">
            <Coords points="10,10 100,10 100,50 10,50"/>
            <TextEquiv><Unicode></Unicode></TextEquiv>
        </TextRegion>
    </Page>
</PcGts>"""

        page_data = parser._parse_page_xml(xml)

        assert len(page_data["regions"]) == 1
        assert page_data["regions"][0].full_text == ""
        assert len(page_data["regions"][0].text_lines) == 0

    def test_parse_line_reading_order_from_custom(self, parser):
        """Test parsing reading order from custom attribute."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion type="paragraph" id="r1">
            <Coords points="10,10 100,10 100,50 10,50"/>
            <TextLine id="l1" custom="readingOrder {index:5;}">
                <Coords points="10,10 90,10 90,20 10,20"/>
                <TextEquiv><Unicode>Line 1</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""

        page_data = parser._parse_page_xml(xml)

        line = page_data["regions"][0].text_lines[0]
        assert line.reading_order == 5

    def test_parse_line_without_reading_order(self, parser):
        """Test parsing line without reading order."""
        xml = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion type="paragraph" id="r1">
            <Coords points="10,10 100,10 100,50 10,50"/>
            <TextLine id="l1">
                <Coords points="10,10 90,10 90,20 10,20"/>
                <TextEquiv><Unicode>Line 1</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""

        page_data = parser._parse_page_xml(xml)

        line = page_data["regions"][0].text_lines[0]
        assert line.reading_order == 0  # Default value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
