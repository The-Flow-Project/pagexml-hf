"""
Parser for Transkribus ZIP files and PAGE XML format.
"""

import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import chardet
import datasets
import lxml.etree as et
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

from .logger import logger


@dataclass
class TextLine:
    """Represents a text line in the PAGE XML."""

    id: str
    text: Optional[str]
    coords: List[Tuple[int, int]]
    baseline: Optional[List[Tuple[int, int]]]
    reading_order: int
    region_id: str

    def to_dict(self):
        """
        Convert TextLine object to a Python dictionary.
        """
        return {
            "id": self.id,
            "text": self.text if self.text else "",
            "coords": self.coords,
            "baseline": self.baseline if self.baseline else None,
            "reading_order": self.reading_order,
            "region_id": self.region_id,
        }


class XmlParser:
    """
    Parser for Transkribus ZIP files containing PAGE XML format or
    Page XML files in a folder structure with images or image-URL in the XML.
    """

    def __init__(self):
        logger.info(f"Initializing XmlParser")

    def parse_zip(self, zip_path: str, parse_xml: bool = False) -> pd.DataFrame | None:
        """
        Parse a Transkribus ZIP file and extract all page data.

        Args:
            zip_path: Path or URL to the ZIP file
            parse_xml: Whether or not to parse the XML file and to get regions and lines (depends on mode)

        Returns:
            List of PageData objects
        """

        if zip_path.startswith("http://") or zip_path.startswith("https://"):
            try:
                response = requests.get(zip_path, timeout=30)
                response.raise_for_status()
                zip_data_io = io.BytesIO(response.content)
                zip_file = zipfile.ZipFile(zip_data_io)
            except requests.exceptions.Timeout as e:
                raise ValueError(f"Download from {zip_path} timed out: {e}") from e
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Download from {zip_path} failed: {e}") from e
        else:
            if not zipfile.is_zipfile(zip_path):
                raise ValueError(f"{zip_path} is not a valid ZIP file")
            zip_file = zipfile.ZipFile(zip_path)

        # Open ZipFile
        with zip_file:
            # Get all files in the ZIP
            file_list = zip_file.namelist()

            image_files = [
                Path(f) for f in file_list
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
                   and not self._is_macos_metadata_file(f)
            ]

            # remove image files from file_list (faster)
            file_list = [
                Path(f) for f in file_list
                if f.lower().endswith(".xml")
                   and not self._is_macos_metadata_file(f)
                   and not self._is_metadata_file(f)
            ]
            logger.debug(f"Found {len(file_list)} files, {len(image_files)} images")
            # Create empty pd.DataFrame
            xml_dataframe = self._create_dataframe(parse_xml)

            for xml_filename in tqdm(file_list, total=len(file_list), desc="Parsing XML files"):
                row = {}
                logger.debug(f"Processing file: {str(xml_filename)}")
                xml_content = self._read_xml_with_encoding(zip_file, str(xml_filename))
                if not xml_content:
                    logger.warning(f"Skipping file: {str(xml_filename)}")
                    continue

                row["xml_content"] = xml_content
                parent = xml_filename.parent
                row["project_name"] = parent.name if parent.name != "page" else parent.parent.name
                imagepath = next((i for i in image_files if i.stem in str(xml_filename)), None)
                row["image"] = None
                try:
                    logger.debug(f"parsing image: {str(xml_filename)}")
                    page_data = self._parse_page_xml(xml_content)
                    if parse_xml:
                        row["regions"] = page_data["regions"]
                        row["image_width"] = page_data["image_width"]
                        row["image_height"] = page_data["image_height"]
                    if imagepath is None:
                        row["image"] = {"bytes": self._get_image_from_url(page_data["image_url"]), "path": None}
                except Exception as e:
                    logger.error(f"Error parsing page {str(xml_filename)}: {e}")
                    continue
                if row["image"] is None and imagepath:
                    logger.debug(f"image is None and imagepath: {imagepath}")
                    try:
                        row["image"] = {"bytes": zip_file.read(str(imagepath)), "path": None}
                        row["filename"] = imagepath.name
                    except Exception as e:
                        logger.error(f"Failed to attach image: {e}")
                        continue
                if row["image"] is None:
                    logger.debug(f"image is None")
                    logger.error(f"Failed to attach image: {e}")
                    continue

                xml_dataframe.loc[len(xml_dataframe)] = row

            if xml_dataframe.empty:
                return None
            else:
                return xml_dataframe

    def parse_folder(self, folder_path: str, parse_xml: bool = False) -> pd.DataFrame | None:
        """
        Parse PAGE XML files from a folder path mimicking Transkribus structure.

        Args:
            folder_path: Path to the folder
            parse_xml: Whether or not to parse the XML file and to get regions and lines (depends on mode)

        Returns:
            List of PageData objects
        """
        xml_dataframe = self._create_dataframe(parse_xml)

        folder = Path(folder_path)
        filelist = [
            p for p in folder.rglob("*")
            if p.is_file()
               and not self._is_metadata_file(str(p))
               and not self._is_macos_metadata_file(str(p))
        ]
        xml_files = [f for f in filelist if f.suffix.lower() == ".xml"]
        image_files = [i for i in filelist if i.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]

        logger.info(f"Found {len(xml_files)} XML files")
        logger.info(f"Found {len(image_files)} images")

        for xml_filename in tqdm(xml_files, total=len(xml_files), desc="Parsing XML files"):
            row = {}
            logger.debug(f"Processing file: {str(xml_filename)}")
            xml_content = self._file_loader(str(xml_filename))
            if not xml_content:
                logger.debug(f"Skipping file: {str(xml_filename)}")
                continue

            row["xml_content"] = xml_content
            parent = xml_filename.parent
            row["project_name"] = parent.name if parent.name != "page" else parent.parent.name
            imagepath = next((i.absolute() for i in image_files if i.stem in str(xml_filename)), None)
            row["image"] = None
            if parse_xml or imagepath is None:
                try:
                    page_data = self._parse_page_xml(xml_content)
                    if parse_xml:
                        row["regions"] = page_data["regions"]
                        row["image_width"] = page_data["image_width"]
                        row["image_height"] = page_data["image_height"]
                    if imagepath is None:
                        row["image"] = self._get_image_from_url(page_data["image_url"])
                except Exception as e:
                    logger.error(f"Error parsing page {str(xml_filename)}: {e}")
                    continue
            if row["image"] is None and imagepath:
                try:
                    with open(imagepath, "rb") as f:
                        row["image"] = f.read()
                    row["filename"] = imagepath.name
                except Exception as e:
                    logger.error(f"Failed to attach image: {e}")
                    continue
            if row["image"] is None:
                logger.error(f"Failed to attach image from xml file: {str(xml_filename)}")
                continue

            xml_dataframe.loc[len(xml_dataframe)] = row

        if xml_dataframe.empty:
            return None
        else:
            return xml_dataframe

    def parse_dataset(
            self,
            dataset: Union[str, datasets.Dataset],
            token: str | None = None,
            parse_xml: bool = False,
    ) -> pd.DataFrame | None:
        """
        Parse a HuggingFace dataset containing PAGE XML files.
        It only fetches the 'train' split!

        Args:
            dataset: HuggingFace dataset ID or Dataset object
            token: Optional HuggingFace token for private datasets
            parse_xml: Whether or not to parse the XML file (depends on mode)

        Returns:
            List of PageData objects
        """

        xml_dataframe = self._create_dataframe(parse_xml)
        logger.info(f"Loading dataset {dataset}...")
        if isinstance(dataset, str):
            try:
                ds = load_dataset(dataset, split="train", token=token)
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset {dataset}: {e} \
                        (did you set a token for private datasets?)"
                ) from e
        elif isinstance(dataset, datasets.Dataset):
            ds = dataset
        else:
            raise ValueError("dataset must be a string or a datasets.Dataset object")

        column_names = ds.column_names if isinstance(ds.column_names, list) else []
        if not set(column_names).issuperset(['image', 'xml']):
            raise ValueError(f"Dataset {dataset} must contain 'xml' and 'image' columns")

        logger.debug(f"Dataset loaded:\n{ds}")

        try:
            if type(ds.features["image"]) == datasets.features.image.Image:
                if ds.features["image"].decode:
                    ds = ds.cast_column("image", datasets.Image(decode=False))
                else:
                    pass
        except Exception as e:
            raise ValueError(
                f"Dataset {dataset} must contain 'image' with Image features: {e}"
            )

        dataset_iterable = ds.to_iterable_dataset()
        for item in dataset_iterable:
            row = {}
            xml_content = item.get("xml")
            if xml_content is None:
                logger.warning("Skipping item without XML content")
                continue

            row["xml_content"] = xml_content
            row["project_name"] = item.get("project_name", "unknown_project")
            row["image"] = None

            page_data = self._parse_page_xml(xml_content)
            if page_data:
                try:
                    image = item.get("image")["bytes"]
                    if image is not None:
                        row["image"] = image
                except Exception as e:
                    logger.error(f"Failed to parse image: {e}")
                    continue

                row["filename"] = page_data["image_filename"]

                if parse_xml:
                    row["regions"] = page_data["regions"]
                    row["image_width"] = page_data["image_width"]
                    row["image_height"] = page_data["image_height"]
            else:
                logger.warning("Skipping item with invalid XML content")
                continue

            xml_dataframe.loc[len(xml_dataframe)] = row

        if xml_dataframe.empty:
            return None
        else:
            return xml_dataframe

    def _file_loader(self, file_path: str) -> str | None:
        """
        Load XML file content with automatic encoding detection.
        """
        try:
            with open(file_path, "rb") as f:
                raw_content = f.read()
            return self._decode_bytes(raw_content, file_path)
        except (OSError, IOError, UnicodeDecodeError) as err:
            logger.error(f"Error reading {file_path}: {err}")
            return None

    def _read_xml_with_encoding(
            self, zip_file: zipfile.ZipFile, xml_filename: str
    ) -> Optional[str]:
        """Read XML content with automatic encoding detection and fallback."""
        try:
            raw_content = zip_file.read(xml_filename)
            return self._decode_bytes(raw_content, xml_filename)
        except (KeyError, OSError, IOError, zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logger.error(f"Error reading {xml_filename}: {e}")
            return None

    @staticmethod
    def _decode_bytes(raw_content: bytes, source_name: str = "") -> Optional[str]:
        try:
            return raw_content.decode()  # Default to UTF-8
        except UnicodeDecodeError:
            pass

        detected = chardet.detect(raw_content)
        if detected and detected["confidence"] > 0.7:
            encoding = detected["encoding"]
            if encoding is not None:
                try:
                    return raw_content.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    pass

        for enc in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                return raw_content.decode(enc)
            except UnicodeDecodeError:
                continue

        logger.error(f"Could not decode {source_name} with any supported encoding")
        return None

    @staticmethod
    def _is_macos_metadata_file(file_path: str) -> bool:
        """Check if a file is a macOS metadata file that should be skipped."""
        # Skip __MACOSX directory and ._ prefixed files
        if "__MACOSX" in file_path or file_path.startswith("._"):
            return True

        # Skip other common macOS metadata patterns
        if "/." in file_path and not file_path.endswith(".xml"):
            return True

        return False

    @staticmethod
    def _is_metadata_file(file_path: str) -> bool:
        """Check if a file is a metadata file that should be skipped."""
        base = os.path.basename(file_path)
        return base.lower() in ['mets.xml', 'metadata.xml']

    def _parse_page_xml(self, xml_content: str) -> Dict | None:
        """Parse a single PAGE XML file."""
        try:
            # logger.debug(f"Parsing XML {xml_content.encode()}")
            tree = et.parse(
                io.BytesIO(xml_content.encode()),
                parser=et.XMLParser(
                    encoding='utf-8',
                    ns_clean=True,
                    compact=False,
                )
            )
            root = tree.getroot()
            logger.debug(f"DefaultNamespace: {root.nsmap.get(None)}")
            self.namespace = {'pc': root.nsmap.get(None)}
            logger.debug(f"Namespace: {self.namespace}")

            # Get page element
            page_elem = root.find("pc:Page", self.namespace)
            if page_elem is None:
                logger.warning("No Page element found in XML")
                return None

            # Extract page metadata
            image_filename = page_elem.get("imageFilename", "")
            image_width = int(page_elem.get("imageWidth", 0))
            image_height = int(page_elem.get("imageHeight", 0))
            image_url = self._parse_imgurl(root)

            # Parse reading order
            reading_order = self._parse_reading_order(root)

            # Parse text regions
            regions = self._parse_text_regions(root, reading_order)

            return {
                "image_filename": image_filename,
                "image_width": image_width,
                "image_height": image_height,
                "image_url": image_url,
                "regions": regions,
            }

        except et.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None

    def _parse_reading_order(self, root: et.Element) -> Dict[str, int]:
        """Parse the reading order from the XML."""
        reading_order = {}

        reading_order_elem = root.find(".//pc:ReadingOrder", self.namespace)
        if reading_order_elem is not None:
            for region_ref in reading_order_elem.findall(
                    ".//pc:RegionRefIndexed", self.namespace
            ):
                region_id = region_ref.get("regionRef", "")
                index = int(region_ref.get("index", 0))
                reading_order[region_id] = index

        return reading_order

    def _parse_text_regions(
            self, root: et.Element, reading_order: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Parse all text regions from the XML."""
        regions = []

        for region_elem in root.findall(".//pc:TextRegion", self.namespace):
            region_id = region_elem.get("id", "")
            region_type = region_elem.get("type", "paragraph")

            # Parse coordinates
            coords: List[Tuple[int, int]] = self._parse_coords(region_elem.find("pc:Coords", self.namespace))

            # Parse text lines
            text_lines: List[Dict[str, Any]] = self._parse_text_lines(region_elem, region_id)

            # Get full text from TextEquiv
            full_text: str = self._get_text_equiv(region_elem)

            # Get reading order
            region_reading_order: int = reading_order.get("region_id", 0)

            region = {
                "id": region_id,
                "type": region_type,
                "coords": coords,
                "text_lines": text_lines,
                "reading_order": region_reading_order,
                "full_text": full_text,
            }

            regions.append(region)

        # Sort regions by reading order
        regions.sort(key=lambda r: r["reading_order"])

        return regions

    def _parse_text_lines(
            self, region_elem: et.Element, region_id: str
    ) -> List[Dict[str, Any]]:
        """Parse text lines within a region."""
        lines = []

        for line_elem in region_elem.findall("pc:TextLine", self.namespace):
            line_id = line_elem.get("id", "")

            # Parse coordinates
            coords = self._parse_coords(line_elem.find("pc:Coords", self.namespace))

            # Parse baseline
            baseline_elem = line_elem.find("pc:Baseline", self.namespace)
            baseline = (
                self._parse_coords(baseline_elem) if baseline_elem is not None else None
            )

            # Get text content
            text = self._get_text_equiv(line_elem)

            # Extract reading order from custom attribute
            reading_order = self._extract_reading_order_from_custom(line_elem)

            line = {
                "id": line_id,
                "text": text,
                "coords": coords,
                "baseline": baseline,
                "reading_order": reading_order,
                "region_id": region_id,
            }

            lines.append(line)

        # Sort lines by reading order
        lines.sort(key=lambda ln: ln["reading_order"])

        return lines

    def _parse_imgurl(self, root: et.Element) -> str | None:
        """Parse image URL from the PAGE XML."""
        img_url_elem = root.find(".//pc:TranskribusMetadata", self.namespace)
        # If from Transkribus, the image URL might be in the TranskribusMetadata element
        image_url = img_url_elem.get("imgUrl") if img_url_elem is not None else None
        # If not found, try to get it from the Page element
        if not image_url:
            page_elem = root.find("pc:Page", self.namespace)
            image_url = page_elem.get("imageURL", None)

        return image_url

    @staticmethod
    def _get_image_from_url(image_url: str) -> bytes | None:
        if image_url and (image_url.startswith("http") or image_url.startswith("https")):
            try:
                response = requests.get(image_url, timeout=20)
                response.raise_for_status()
                return response.content
            except requests.exceptions.Timeout:
                logger.error(f'Image download from {image_url} timed out')
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f'Image download from {image_url} failed: {e}')
                return None
        return None

    @staticmethod
    def _parse_coords(coords_elem: et.Element | None) -> List[Tuple[int, int]]:
        """Parse coordinates from a Coords element."""
        if coords_elem is None:
            return []

        points_str = coords_elem.get("points", "")
        if not points_str:
            return []

        coords = []
        for point in points_str.split():
            if "," in point:
                x, y = point.split(",")
                coords.append((int(x), int(y)))

        return coords

    def _get_text_equiv(self, element: et.Element) -> str:
        unicode_el = element.find("pc:TextEquiv/pc:Unicode", self.namespace)
        if unicode_el is not None and unicode_el.text:
            return unicode_el.text.strip()

        lines = element.findall("pc:TextLine", self.namespace)
        if not lines:
            return ""

        line_texts = [
            self._get_text_equiv(line).strip()
            for line in lines
        ]

        return "\n".join(line_texts)

    @staticmethod
    def _extract_reading_order_from_custom(element: et.Element) -> int:
        """Extract reading order from custom attribute."""
        custom = element.get("custom", "")
        if "readingOrder" in custom:
            match = re.search(r"readingOrder\s*\{\s*index\s*:\s*(\d+)", custom)
            if match:
                return int(match.group(1))
        return 0

    @staticmethod
    def _create_dataframe(parse_xml: bool = False) -> pd.DataFrame:
        dtypes = {"xml_content": str, "project_name": str, "filename": str, "image": object}
        dataframe = pd.DataFrame(
            columns=["xml_content", "project_name", "filename", "image"],
        )
        dataframe.astype(dtypes)

        if parse_xml:
            dtypes_extended = {
                "image_width": int,
                "image_height": int,
                "regions": object,
            }
            dataframe_extended = pd.DataFrame(
                columns=["image_width", "image_height", "regions"],
            )
            dataframe_extended.astype(dtypes_extended)
            dataframe = pd.concat(
                [dataframe, dataframe_extended],
                axis=1,
            )
            logger.debug(dataframe.info())
            logger.debug(dataframe.head())
        return dataframe
