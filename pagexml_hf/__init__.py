"""
transkribus-hf: Convert Transkribus ZIP files/Page XML export folder to HuggingFace datasets
"""

from .converter import XmlConverter
from .exporters import (
    LineExporter,
    RawXMLExporter,
    RegionExporter,
    TextExporter,
    WindowExporter,
)
from .logger import setup_logger
from .parser import XmlParser

__version__ = "0.9.6+fork.1"
__license__ = "MIT"
__authors__ = ["wjbmattingly", "jnswidmer"]

__all__ = [
    "XmlConverter",
    "XmlParser",
    "RawXMLExporter",
    "TextExporter",
    "RegionExporter",
    "LineExporter",
    "WindowExporter",
    "setup_logger",
]
