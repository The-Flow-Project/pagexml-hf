"""
Exporters for converting parsed Transkribus data to different HuggingFace dataset formats.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

from PIL import ImageFile
import pandas as pd
from datasets import (
    Dataset,
    IterableDataset,
    Features,
    Value,
    Image as DatasetImage,
    List as DatasetList,
    Sequence
)

from .logger import logger
from .imageutils import ImageProcessor, calculate_bounding_box

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Features used for dataframes with parsed xmls
PRE_FEATURES = Features(
    {
        "image": DatasetImage(decode=False),
        "xml_content": Value("string"),
        "filename": Value("string"),
        "project_name": Value("string"),
        "image_width": Value("int64"),
        "image_height": Value("int64"),
        "regions": DatasetList({
            "id": Value("string"),
            "type": Value("string"),
            "coords": DatasetList(Sequence(Value("int64"))),
            "text_lines": DatasetList({
                "id": Value("string"),
                "text": Value("string"),
                "coords": DatasetList(Sequence(Value("int64"))),
                "baseline": DatasetList(Sequence(Value("int64"))),
                "reading_order": Value("int64"),
                "region_id": Value("string"),
            }),
            "reading_order": Value("int64"),
            "full_text": Value("string"),
        }),
    }
)


class BaseExporter(ABC):
    """Base class for all exporters."""

    def __init__(
            self,
            crop: bool = False,
            min_width: int | None = None,
            min_height: int | None = None,
    ):
        self.crop: bool = crop
        self.min_width: int = min_width
        self.min_height: int = min_height

    @abstractmethod
    def export(self, pages: pd.DataFrame) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export pages to a HuggingFace dataset."""

    @staticmethod
    def _flatten_cropped_region(batch):
        logger.debug("Flatten cropped images.")
        # batch["cropped_areas"] is a list‑of‑lists (len = batch_size)
        flat_image, flat_text, flat_type, flat_id, flat_order, flat_fname, flat_proj = (
            [], [], [], [], [], [], []
        )

        for i, regions in enumerate(batch["cropped_areas"]):
            # keep the original page‑level metadata
            fname = batch["filename"][i]
            proj = batch["project_name"][i]

            for reg in regions:
                flat_image.append(reg["image"])
                flat_text.append(reg["text"])
                flat_type.append(reg["region_type"])
                flat_id.append(reg["region_id"])
                flat_order.append(reg["region_reading_order"])
                flat_fname.append(fname)
                flat_proj.append(proj)

        return {
            "image": flat_image,
            "text": flat_text,
            "region_id": flat_id,
            "region_reading_order": flat_order,
            "region_type": flat_type,
            "filename": flat_fname,
            "project_name": flat_proj,
        }

    @staticmethod
    def _flatten_cropped_line(batch):
        logger.debug("Flatten cropped line images.")
        # batch["cropped_areas"] is a list‑of‑lists (len = batch_size)
        flat_image, flat_text, flat_id, flat_order, flat_rid, flat_rorder, flat_rtype, flat_fname, flat_proj = (
            [], [], [], [], [], [], [], [], []
        )

        for i, regions in enumerate(batch["cropped_areas"]):
            # keep the original page‑level metadata
            fname = batch["filename"][i]
            proj = batch["project_name"][i]

            for reg in regions:
                flat_image.append(reg["image"])
                flat_text.append(reg["text"])
                flat_id.append(reg["line_id"])
                flat_order.append(reg["line_reading_order"])
                flat_rid.append(reg["region_id"])
                flat_rorder.append(reg["region_reading_order"])
                flat_rtype.append(reg["region_type"])
                flat_fname.append(fname)
                flat_proj.append(proj)

        return {
            "image": flat_image,
            "text": flat_text,
            "line_id": flat_id,
            "line_reading_order": flat_order,
            "region_id": flat_rid,
            "region_reading_order": flat_rorder,
            "region_type": flat_rtype,
            "filename": flat_fname,
            "project_name": flat_proj,
        }


class RawXMLExporter(BaseExporter):
    """Export raw images with their corresponding XML content."""

    def export(self, pages: pd.DataFrame) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export pages as image + raw XML pairs."""
        logger.info(f"Exporting raw XML content with images... (number of pages: {pages.shape[0]})")
        image_processor = ImageProcessor()
        features = Features(
            {
                "image": DatasetImage(decode=False),
                "xml_content": Value("string"),
                "filename": Value("string"),
                "project_name": Value("string"),
            }
        )
        logger.debug(f"Get dataset from Pandas DataFrame")
        dataset = Dataset.from_pandas(pages, preserve_index=False, features=features)

        logger.debug(f"Dataset from pandas generated")
        dataset = dataset.flatten_indices()
        dataset = dataset.to_iterable_dataset(num_shards=5)
        logger.debug(f"Start mapping for orientation correction")
        dataset = dataset.map(
            image_processor.correct_orientation,
            batched=True,
            batch_size=32,
            features=features
        )

        logger.debug(dataset.info)
        logger.debug(dataset.features)

        logger.info(f"Iterable dataset prepared")
        return dataset


class TextExporter(BaseExporter):
    """Export images with concatenated text content."""

    def export(self, pages: pd.DataFrame) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export pages as image + full text pairs."""

        logger.info(f"Exporting text content with images... (Processed: {pages.shape[0]})")
        image_processor = ImageProcessor()

        def map_examples(batch):
            """
            Generator for dataset creation.
            """
            logger.debug(f"Start mapping to add full_text")
            results = []
            for i, regions in enumerate(batch["regions"]):
                if regions is not None:
                    # Concatenate all text from regions in reading order
                    logger.debug(f"Processing region {i + 1} of {len(batch)}")
                    full_text = '\n'.join(
                        r.get("full_text", "") for r in regions
                    )
                    if not full_text:
                        full_text = ""
                    output_dict = {
                        "image": batch["image"][i],
                        "text": full_text,
                        "filename": batch["filename"][i],
                        "project_name": batch["project_name"][i],
                    }
                    logger.debug(f"Finished mapping to full text: {full_text}")
                    results.append(output_dict)

            return {
                "image": [r["image"] for r in results],
                "text": [r["text"] for r in results],
                "filename": [r["filename"] for r in results],
                "project_name": [r["project_name"] for r in results],
            }

        post_features = Features(
            {
                "image": DatasetImage(decode=False),
                "text": Value("string"),
                "filename": Value("string"),
                "project_name": Value("string"),
            }
        )
        try:
            logger.debug(f"Get dataset from Pandas DataFrame")
            dataset = Dataset.from_pandas(
                pages, preserve_index=False, features=PRE_FEATURES  # default: cache_dir=None
            )
            logger.debug(f"Dataset from pandas generated")
            dataset = dataset.flatten_indices()
            dataset = dataset.to_iterable_dataset(num_shards=5)
            dataset = dataset.map(
                image_processor.correct_orientation,
                batched=True,
                batch_size=32,
                features=PRE_FEATURES
            )
            dataset = dataset.map(
                map_examples,
                batched=True,
                batch_size=32,
                features=post_features,
                remove_columns=["xml_content", "image_width", "image_height", "regions"],
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        logger.debug(dataset.info)
        logger.debug(dataset.features)
        logger.info(f"Iterable Dataset prepared")
        return dataset


class RegionExporter(BaseExporter):
    """Export individual regions as separate images with metadata."""

    def export(
            self,
            pages: pd.DataFrame,
            mask: bool = False,
            min_width: Optional[int] = None,
            min_height: Optional[int] = None,
            allow_empty: bool = False,
    ) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export each region as a separate dataset entry."""

        logger.info(f"Exporting Region XML content with images... (Processed: {len(pages)})")
        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
            allow_empty=allow_empty
        )

        post_features = Features({
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "region_id": Value("string"),
            "region_reading_order": Value("int64"),
            "region_type": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
        })

        try:
            dataset = Dataset.from_pandas(pages, preserve_index=False, features=PRE_FEATURES)
            dataset = dataset.flatten_indices()
            dataset = dataset.to_iterable_dataset()
            dataset = dataset.map(
                image_processor.correct_orientation,
                batched=True,
                batch_size=32,
                features=PRE_FEATURES
            )
            dataset = dataset.map(
                image_processor.process_entry_cropping,
                batched=True,
                batch_size=32,
                features=PRE_FEATURES,
            )
            dataset = dataset.map(
                self._flatten_cropped_region,
                batched=True,
                batch_size=32,
                features=post_features,
                remove_columns=["cropped_areas", "xml_content", "image_width", "image_height", "regions"],
            )
        except ValueError as e:
            logger.error(f"ValueError creating dataset: {e}")
            dataset = None
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset


class LineExporter(BaseExporter):
    """Export individual text lines as separate images with metadata."""

    def export(
            self,
            pages: pd.DataFrame,
            mask: bool = False,
            min_width: Optional[int] = None,
            min_height: Optional[int] = None,
            allow_empty: bool = False,
    ) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export each text line as a separate dataset entry."""
        logger.info(f"Exporting line content with images... (Processed: {len(pages)})")

        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
            allow_empty=allow_empty,
            line_based=True,
        )

        post_features = Features({
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "line_id": Value("string"),
            "line_reading_order": Value("int64"),
            "region_id": Value("string"),
            "region_reading_order": Value("int64"),
            "region_type": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
        })

        try:
            dataset = Dataset.from_pandas(pages, preserve_index=False, features=PRE_FEATURES)
            dataset = dataset.flatten_indices()
            dataset = dataset.to_iterable_dataset()
            dataset = dataset.map(
                image_processor.correct_orientation,
                batched=True,
                batch_size=32,
                features=PRE_FEATURES
            )
            dataset = dataset.map(
                image_processor.process_entry_cropping,
                batched=True,
                batch_size=32,
                features=PRE_FEATURES,
            )
            dataset = dataset.map(
                self._flatten_cropped_line,
                batched=True,
                batch_size=32,
                features=post_features,
                remove_columns=["cropped_areas", "xml_content", "image_width", "image_height", "regions"],
            )
        except ValueError as e:
            logger.error(f"ValueError creating dataset: {e}")
            dataset = None
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset


class WindowExporter(BaseExporter):
    """Export sliding windows of text lines with configurable window size and overlap."""

    def __init__(
            self,
            window_size: int = 2,
            overlap: int = 0,
    ):
        """
        Initialize the window exporter.

        Args:
            window_size: Number of lines per window (1, 2, 3, etc.)
            overlap: Number of lines to overlap between windows
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap

        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")

    def export(self, pages: pd.DataFrame, mask: bool = False, allow_empty: bool = False) -> Union[
        Dataset,
        IterableDataset,
        None
    ]:
        """Export sliding windows of lines as separate dataset entries."""
        logger.info(f"Exporting window content with images... (Processed: {len(pages)})")

        image_processor = ImageProcessor(mask_crop=mask, allow_empty=allow_empty)

        def map_examples(batch):
            """
            Function for dataset mapping.
            """
            result = []
            for region in batch["regions"]:
                # Generate sliding windows for this region
                windows = self._create_windows(region["text_lines"])
                for window_idx, window_lines in enumerate(windows):
                    # Calculate bounding box for all lines in this window
                    line_coords = [
                        line["coords"] for line in window_lines if line["coords"] is not None
                    ]
                    if line_coords is not None:
                        window_coords = calculate_bounding_box(line_coords)
                        window_image = image_processor.crop_with_coords(
                            batch["image"], window_coords
                        )
                        if window_image is not None:
                            # Combine text from all lines in window
                            window_text = "\n".join(
                                [
                                    line["text"]
                                    for line in window_lines
                                    if line["text"] or allow_empty
                                ]
                            )
                            # Create line info for metadata
                            line_ids = [line["id"] for line in window_lines]
                            line_orders = [
                                line["reading_order"] for line in window_lines
                            ]
                            result.append({
                                "image": window_image,
                                "text": window_text,
                                "window_size": len(window_lines),
                                "window_index": window_idx,
                                "line_ids": ", ".join(line_ids),
                                "line_reading_orders": ", ".join(
                                    map(str, line_orders)
                                ),
                                "region_id": region["id"],
                                "region_reading_order": region["reading_order"],
                                "region_type": region["type"],
                                "filename": batch["filename"],
                                "project_name": batch["project_name"],
                            })
            return result

        # Create dataset using generator to avoid memory issues
        features = Features(
            {
                "image": Value("bytes"),
                "xml_content": Value("string"),
                "filename": Value("string"),
                "project_name": Value("string"),
                "image_width": Value("int32"),
                "image_height": Value("int32"),
                "regions": DatasetList(Dict[str, Any]),
            }
        )

        try:
            logger.debug(pages.head())
            dataset = Dataset.from_pandas(pages, features=features, preserve_index=False)
            dataset = dataset.to_iterable_dataset()
            dataset = dataset.map(map_examples)
            dataset = dataset.cast_column("image", DatasetImage(decode=True))
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset

    def _create_windows(self, lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
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