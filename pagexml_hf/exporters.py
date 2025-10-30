"""
Exporters for converting parsed Transkribus data to different HuggingFace dataset formats.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from PIL import Image, ImageFile
import numpy as np
import cv2
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    Features,
    Value,
    Image as DatasetImage,
    disable_caching
)

from .parser import PageData, TextLine
from .logger import logger

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
disable_caching()


class BaseExporter(ABC):
    """Base class for all exporters."""

    def __init__(
            self,
            pages: List[PageData],
    ):
        self.pages = pages
        self.failed_images = []
        self.processed_count = 0
        self.skipped_count = 0

    @abstractmethod
    def export(self, pages: List[PageData]) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export pages to a HuggingFace dataset."""

    def _crop_region(
            self,
            image: Image.Image,
            coords: List[Tuple[int, int]],
            mask: bool = False,
            min_width: Optional[int] = None,
            min_height: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """Crop a region from an image based on coordinates, \
            optimized by pre-cropping to bounding box."""
        if not coords:
            logger.warning("Warning: No coordinates provided for cropping.")
            self.skipped_count += 1
            return None

        try:
            # Calculate bounding box for the coordinates
            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            min_x, max_x = max(0, min(x_coords)), min(image.width, max(x_coords))
            min_y, max_y = max(0, min(y_coords)), min(image.height, max(y_coords))

            if min_x >= max_x or min_y >= max_y:
                logger.warning(f"Warning: Invalid crop coordinates: ({min_x}, {min_y}, {max_x}, {max_y})")
                self.skipped_count += 1
                return None

            if min_width and int(max_x - min_x) < min_width:
                self.skipped_count += 1
                return None

            if min_height and int(max_y - min_y) < min_height:
                self.skipped_count += 1
                return None

            # Bild und Koordinaten auf Bounding Box beschränken
            image_cropped = image.crop((min_x, min_y, max_x, max_y))
            shifted_coords = [(x - min_x, y - min_y) for (x, y) in coords]
            img_array = cv2.cvtColor(np.array(image_cropped), cv2.COLOR_RGB2BGR)

            if mask:
                mask_img = np.zeros(img_array.shape[:2], dtype=np.uint8)
                pts = np.array([shifted_coords], dtype=np.int32)
                cv2.fillPoly(mask_img, pts, color=(255, 0, 0))
                white_bg = np.ones_like(img_array, dtype=np.uint8) * 255
                result = np.where(mask_img[:, :, None] == 255, img_array, white_bg)
            else:
                result = img_array

            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)

        except Exception as e:
            logger.warning(f"Warning: Error cropping region: {e}")
            return None

    @staticmethod
    def _calculate_bounding_box(
            coords_list: List[List[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        """Calculate the bounding box that encompasses multiple coordinate sets."""
        if not coords_list:
            return []

        all_coords = []
        for coords in coords_list:
            all_coords.extend(coords)

        if not all_coords:
            return []

        x_coords = [coord[0] for coord in all_coords]
        y_coords = [coord[1] for coord in all_coords]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Return as rectangle coordinates
        logger.debug(f"calculated bounding box for coordinates: {min_x, min_y}, {max_x, max_y}")
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    def _print_summary(
            self,
            dataset: Optional[
                Union[
                    Dataset,
                    DatasetDict,
                    IterableDataset,
                    dict[str, IterableDataset]
                ]
            ] = None
    ) -> None:
        """Print processing summary."""
        logger.info(f"Processing Summary:")
        print("#" * 60)
        if not dataset:
            print("No dataset created.")
            return
        if self.processed_count == 0 and self.skipped_count == 0:
            print("⚠️ No new items processed — dataset likely loaded from cache.")
        else:
            print("\nProcessing Summary:")
            print(f"  Successfully processed: {self.processed_count}")
            print(f"  Skipped due to errors: {self.skipped_count}")
            if self.failed_images:
                print("  Failed images:")
                for image_path, error in self.failed_images[:5]:  # Show first 5 errors
                    print(f"    {image_path}: {error}")
                if len(self.failed_images) > 5:
                    print(f"    ... and {len(self.failed_images) - 5} more")


class RawXMLExporter(BaseExporter):
    """Export raw images with their corresponding XML content."""

    def export(self, pages: List[PageData]) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export pages as image + raw XML pairs."""
        logger.info(f"Exporting raw XML content with images... (Processed: {len(pages)})")

        def generate_examples():
            """Generate examples from pages with images and XML content."""
            for page in tqdm(pages, desc="Generating RawXML dataset"):
                image = page.image
                if image:
                    self.processed_count += 1
                    yield {
                        "image": image,
                        "xml": page.xml_content,
                        "filename": page.image_filename,
                        "project": page.project_name,
                    }
                else:
                    self.skipped_count += 1
                    self.failed_images.append([page.image_filename, "No image found"])

        features = Features(
            {
                "image": DatasetImage(),
                "xml": Value("string"),
                "filename": Value("string"),
                "project": Value("string"),
            }
        )

        #try:
        dataset = Dataset.from_generator(
            generate_examples, features=features  # default: cache_dir=None
        )

        logger.info(f"Dataset generated: {len(dataset)}")
        self._print_summary(dataset)
        return dataset


class TextExporter(BaseExporter):
    """Export images with concatenated text content."""

    def export(self, pages: List[PageData]) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export pages as image + full text pairs."""

        def generate_examples():
            """
            Generator for dataset creation.
            """
            for page in tqdm(pages, desc="Generating Text dataset"):
                image = page.image
                if image is not None:
                    # Concatenate all text from regions in reading order
                    full_text = "\n".join(
                        [
                            region.full_text
                            for region in page.regions
                            if region.full_text
                        ]
                    )
                    self.processed_count += 1
                    yield {
                        "image": image,
                        "text": full_text,
                        "filename": page.image_filename,
                        "project": page.project_name,
                    }
                else:
                    self.skipped_count += 1
                    self.failed_images.append([page.image_filename, "No image found"])

        features = Features(
            {
                "image": DatasetImage(),
                "text": Value("string"),
                "filename": Value("string"),
                "project": Value("string"),
            }
        )

        try:
            dataset = Dataset.from_generator(
                generate_examples, features=features  # default: cache_dir=None
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        self._print_summary(dataset)
        return dataset


class RegionExporter(BaseExporter):
    """Export individual regions as separate images with metadata."""

    def export(
            self,
            pages: List[PageData],
            mask: bool = False,
            min_width: Optional[int] = None,
            min_height: Optional[int] = None,
            allow_empty: bool = False,
    ) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export each region as a separate dataset entry."""

        def generate_examples():
            """
            Generator for dataset creation.
            """
            check_region_text = any([region.full_text for page in pages for region in page.regions])
            if not check_region_text and not allow_empty:
                raise ValueError("No region contains text. \
                                 Use allow_empty=True to export empty regions.")
            for page in tqdm(pages, desc="Generating Region dataset"):
                full_image = page.image
                if full_image is not None:
                    for region in page.regions:
                        if region.full_text or allow_empty:
                            region_image = self._crop_region(
                                full_image,
                                region.coords,
                                mask=mask,
                                min_width=min_width,
                                min_height=min_height,
                            )
                            if region_image is not None:
                                self.processed_count += 1
                                yield {
                                    "image": region_image,
                                    "text": region.full_text,
                                    "region_type": region.type,
                                    "region_id": region.id,
                                    "reading_order": region.reading_order,
                                    "filename": page.image_filename,
                                    "project": page.project_name,
                                }
                else:
                    self.skipped_count += 1
                    self.failed_images.append([page.image_filename, "No image found"])

        features = Features(
            {
                "image": DatasetImage(),
                "text": Value("string"),
                "region_type": Value("string"),
                "region_id": Value("string"),
                "reading_order": Value("int32"),
                "filename": Value("string"),
                "project": Value("string"),
            }
        )

        try:
            dataset = Dataset.from_generator(
                generate_examples, features=features  # default: cache_dir=None
            )
        except ValueError as e:
            logger.error(f"ValueError creating dataset: {e}")
            dataset = None
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        self._print_summary(dataset)
        return dataset


class LineExporter(BaseExporter):
    """Export individual text lines as separate images with metadata."""

    def export(
            self,
            pages: List[PageData],
            mask: bool = False,
            min_width: Optional[int] = None,
            min_height: Optional[int] = None,
            allow_empty: bool = False,
    ) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export each text line as a separate dataset entry."""

        def generate_examples():
            """
            Generator for dataset creation.
            """
            check_region_text = any(region.full_text for page in pages for region in page.regions)
            if not check_region_text and not allow_empty:
                raise ValueError("No region contains text. \
                                 Use allow_empty=True to export empty regions.")
            for page in tqdm(pages, desc="Generating Line dataset"):
                full_image = page.image
                if full_image is None:
                    logger.warning(f"Warning: No image found for page {page.image_filename}")
                    self.skipped_count += 1
                    self.failed_images.append([page.image_filename, "No image found"])
                    continue
                for region in page.regions:
                    for line in region.text_lines:
                        if line.text or allow_empty:
                            line_image = self._crop_region(
                                full_image,
                                line.coords,
                                mask=mask,
                                min_width=min_width,
                                min_height=min_height,
                            )
                            if line_image is not None:
                                self.processed_count += 1
                                yield {
                                    "image": line_image,
                                    "text": line.text if line.text else "",
                                    "line_id": line.id,
                                    "line_reading_order": line.reading_order,
                                    "region_id": line.region_id,
                                    "region_reading_order": region.reading_order,
                                    "region_type": region.type,
                                    "filename": page.image_filename,
                                    "project": page.project_name,
                                }

        features = Features(
            {
                "image": DatasetImage(),
                "text": Value("string"),
                "line_id": Value("string"),
                "line_reading_order": Value("int32"),
                "region_id": Value("string"),
                "region_reading_order": Value("int32"),
                "region_type": Value("string"),
                "filename": Value("string"),
                "project": Value("string"),
            }
        )

        try:
            dataset = Dataset.from_generator(
                generate_examples, features=features  # default: cache_dir=None
            )
        except ValueError as e:
            logger.error(f"ValueError creating dataset: {e}")
            dataset = None
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        self._print_summary(dataset)
        return dataset


class WindowExporter(BaseExporter):
    """Export sliding windows of text lines with configurable window size and overlap."""

    def __init__(
            self,
            pages: List[PageData],
            window_size: int = 2,
            overlap: int = 0,
    ):
        """
        Initialize the window exporter.

        Args:
            pages: The PagesData list to export.
            window_size: Number of lines per window (1, 2, 3, etc.)
            overlap: Number of lines to overlap between windows
        """
        super().__init__(pages=pages)
        self.window_size = window_size
        self.overlap = overlap

        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")

    def export(self, pages: List[PageData], mask: bool = False) -> Union[
        Dataset,
        DatasetDict,
        IterableDataset,
        dict[str, IterableDataset],
        None
    ]:
        """Export sliding windows of lines as separate dataset entries."""

        def generate_examples():
            """
            Generator for dataset creation.
            """
            for page in tqdm(pages, desc="Generating Window dataset"):
                full_image = page.image
                if full_image is None:
                    self.skipped_count += 1
                    self.failed_images.append([page.image_filename, "No image found"])
                    continue

                for region in page.regions:
                    # Generate sliding windows for this region
                    windows = self._create_windows(region.text_lines)
                    for window_idx, window_lines in enumerate(windows):
                        # Calculate bounding box for all lines in this window
                        line_coords = [
                            line.coords for line in window_lines if line.coords
                        ]
                        if line_coords is not None:
                            window_coords = self._calculate_bounding_box(
                                line_coords
                            )
                            window_image = self._crop_region(
                                full_image, window_coords, mask
                            )
                            if window_image is not None:
                                # Combine text from all lines in window
                                window_text = "\n".join(
                                    [
                                        line.text
                                        for line in window_lines
                                        if line.text
                                    ]
                                )
                                # Create line info for metadata
                                line_ids = [line.id for line in window_lines]
                                line_orders = [
                                    line.reading_order for line in window_lines
                                ]
                                self.processed_count += 1
                                yield {
                                    "image": window_image,
                                    "text": window_text,
                                    "window_size": len(window_lines),
                                    "window_index": window_idx,
                                    "line_ids": ", ".join(line_ids),
                                    "line_reading_orders": ", ".join(
                                        map(str, line_orders)
                                    ),
                                    "region_id": region.id,
                                    "region_reading_order": region.reading_order,
                                    "region_type": region.type,
                                    "filename": page.image_filename,
                                    "project": page.project_name,
                                }

        # Create dataset using generator to avoid memory issues
        features = Features(
            {
                "image": DatasetImage(),
                "text": Value("string"),
                "window_size": Value("int32"),
                "window_index": Value("int32"),
                "line_ids": Value("string"),
                "line_reading_orders": Value("string"),
                "region_id": Value("string"),
                "region_reading_order": Value("int32"),
                "region_type": Value("string"),
                "filename": Value("string"),
                "project": Value("string"),
            }
        )

        try:
            dataset = Dataset.from_generator(
                generate_examples, features=features  # default: cache_dir=None
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        self._print_summary(dataset)
        return dataset

    def _create_windows(self, lines: List[TextLine]) -> List[List[TextLine]]:
        """Create sliding windows of lines with specified size and overlap."""
        if not lines:
            return []

        windows = []
        step = self.window_size - self.overlap

        for i in range(0, len(lines), step):
            window = lines[i: i + self.window_size]
            if (
                    len(window) > 0
            ):  # Always include windows, even if smaller than window_size
                windows.append(window)

            # Stop if we've reached the end and the last window would be too small
            # (unless we want to include partial windows)
            if i + self.window_size >= len(lines):
                break

        return windows
