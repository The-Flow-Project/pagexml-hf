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
from .logger import setup_logger

"""
LOGGING_LEVEL = "INFO"
setup_logger(LOGGING_LEVEL)
"""

__version__ = "0.9.0+fork.1"
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
]
