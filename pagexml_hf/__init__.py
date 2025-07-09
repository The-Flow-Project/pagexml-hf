"""
transkribus-hf: Convert Transkribus ZIP files/Page XML export folder to HuggingFace datasets
"""

from .converter import XmlConverter
from .parser import XmlParser
from .exporters import (
    RawXMLExporter,
    TextExporter,
    RegionExporter,
    LineExporter,
    WindowExporter,
)

__version__ = "0.1.0-fork"
__license__ = "MIT"
__authors__ = ["wjbmattingly", "l0rn0r"]

__all__ = [
    "XmlConverter",
    "XmlParser",
    "RawXMLExporter",
    "TextExporter",
    "RegionExporter",
    "LineExporter",
    "WindowExporter",
]
