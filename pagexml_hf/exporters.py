"""
Exporters for converting parsed Transkribus data to different HuggingFace dataset formats.
"""

import json
from abc import ABC, abstractmethod

from datasets import Dataset, Features, List, Value, disable_caching
from datasets import Image as DatasetImage
from loguru import logger
from PIL import ImageFile

from .imageutils import ImageProcessor

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseExporter(ABC):
    """Base class for all exporters."""

    # Used in all exporter classes and in the XmlConverter class.
    POLYGON_FEATURE = List(feature=List(feature=Value("int64")))
    POST_FEATURES = None

    def __init__(
        self,
        batch_size: int = 32,
    ):
        disable_caching()
        self.batch_size: int = batch_size

    @abstractmethod
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Export pages to a HuggingFace dataset."""


class RawXMLExporter(BaseExporter):
    """Export raw images with their corresponding XML content."""

    POST_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "xml_content": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
        }
    )

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)

    def process_dataset(self, dataset: Dataset) -> Dataset | None:
        """Export pages as image + raw XML pairs."""
        logger.info(
            f"Exporting raw XML content with images... (number of pages: {len(dataset)})"
        )

        image_processor = ImageProcessor()

        def map_raw(batch):
            """
            Map the raw XMLs for image correction
            """
            out_images = []
            for img_entry in batch["image"]:
                image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                if image is None:
                    logger.warning("Skipping image due to load failure")
                    continue
                final_bytes = image_processor.encode_image(image)
                out_images.append({"bytes": final_bytes, "path": None})

            return {
                "image": out_images,
                "xml_content": batch["xml_content"],
                "filename": batch["filename"],
                "project_name": batch["project_name"],
            }

        logger.debug("Start mapping for orientation correction")
        try:
            dataset = dataset.map(
                map_raw,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                # num_proc=1,
            )

            logger.debug(dataset.info)
            logger.debug(dataset.features)

            logger.info("Raw XML dataset prepared")
        except Exception as e:
            logger.error(f"Error preparing raw XML dataset: {e}")
            raise e
        return dataset


class TextExporter(BaseExporter):
    """Export images with concatenated text content."""

    POST_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
        }
    )

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)

    def process_dataset(self, dataset: Dataset) -> Dataset | None:
        """Export pages as image + full text pairs."""

        logger.info(
            f"Exporting text content with images... (Parsed {len(dataset)} pages)"
        )
        failed: bool = False
        image_processor = ImageProcessor()

        def process_batch(batch):
            """
            Generator for dataset creation.
            """
            out = {k: [] for k in self.POST_FEATURES}

            logger.debug("Start mapping to add full_text")

            for i in range(len(batch["image"])):
                # ------- Handle Image Orientation ------
                img_entry = batch["image"][i]
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                if pil_image is None:
                    logger.warning("Skipping page due to image load failure")
                    continue

                # ------ Text Extraction ------
                regions_list = batch["regions"][i]
                if regions_list:
                    page_text_parts = [
                        r.get("full_text") for r in regions_list if r.get("full_text")
                    ]
                    full_text = "\n".join(page_text_parts)
                else:
                    full_text = ""

                final_bytes = image_processor.encode_image(pil_image)
                out["image"].append({"bytes": final_bytes, "path": None})
                out["text"].append(full_text)
                out["filename"].append(batch["filename"][i])
                out["project_name"].append(batch["project_name"][i])

            return out

        try:
            logger.debug("Map the provided dataset.")
            dataset = dataset.map(
                process_batch,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                # num_proc=1,
            )
            logger.debug(dataset.info)
            logger.debug(dataset.features)
            logger.info("Text Dataset prepared")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            failed = True

        if failed:
            logger.warning("No success creating dataset; return None.")
            return None
        return dataset


class RegionExporter(BaseExporter):
    """Export individual regions as separate images with metadata."""

    POST_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "region_id": Value("string"),
            "region_reading_order": Value("int32"),
            "region_type": Value("string"),
            "region_coords": BaseExporter.POLYGON_FEATURE,
            "filename": Value("string"),
            "project_name": Value("string"),
        }
    )

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)

    def process_dataset(
        self,
        dataset: Dataset,
        mask: bool = False,
        min_width: int = 0,
        min_height: int = 0,
        allow_empty: bool = False,
    ) -> Dataset | None:
        """Export each region as a separate dataset entry."""

        logger.info(
            f"Exporting Region XML content with images... (Processed: {len(dataset)})"
        )
        failed: bool = False

        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
        )

        def map_regions(batch):
            """
            Mapping the regions
            """
            logger.info("Start mapping to regions...")
            out = {k: [] for k in self.POST_FEATURES}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                if pil_image is None:
                    logger.warning("Skipping page due to image load failure")
                    continue

                regions_list = batch["regions"][i]
                if not regions_list:
                    continue

                for r in regions_list:
                    if not allow_empty and not r.get("full_text"):
                        continue

                    cropped_image = image_processor.crop_from_image(
                        pil_image, r["coords"]
                    )
                    crop_bytes = None

                    if cropped_image:
                        crop_bytes = image_processor.encode_image(cropped_image)

                    if crop_bytes:
                        out["image"].append({"bytes": crop_bytes, "path": None})
                        out["text"].append(r.get("full_text", ""))
                        out["region_id"].append(r["id"])
                        out["region_reading_order"].append(r["reading_order"])
                        out["region_type"].append(r["type"])
                        out["region_coords"].append(r["coords"])
                        out["filename"].append(batch["filename"][i])
                        out["project_name"].append(batch["project_name"][i])

            return out

        try:
            logger.debug("Map the provided dataset.")
            dataset = dataset.map(
                map_regions,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                # num_proc=1,
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            failed = True

        if failed:
            logger.warning("No success creating dataset; return None.")
            return None
        return dataset


class LineExporter(BaseExporter):
    """Export individual text lines as separate images with metadata."""

    POST_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "line_id": Value("string"),
            "line_reading_order": Value("int64"),
            "line_coords": BaseExporter.POLYGON_FEATURE,
            "line_baseline": BaseExporter.POLYGON_FEATURE,
            "line_augmentation": Value("string"),
            "region_id": Value("string"),
            "region_reading_order": Value("int64"),
            "region_type": Value("string"),
            "region_coords": BaseExporter.POLYGON_FEATURE,
            "filename": Value("string"),
            "project_name": Value("string"),
        }
    )

    def __init__(self, batch_size: int = 32):
        super().__init__(batch_size)

    def process_dataset(
        self,
        dataset: Dataset,
        mask: bool = False,
        min_width: int = 0,
        min_height: int = 0,
        allow_empty: bool = False,
        line_augment: int = 0,
    ) -> Dataset | None:
        """Export each text line as a separate dataset entry."""
        logger.info(
            f"Exporting line content with images... (Processed: {len(dataset)})"
        )
        failed: bool = False

        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
        )

        def map_lines(batch):
            """
            Mapping the lines and cropping them
            """
            out = {k: [] for k in self.POST_FEATURES}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                if pil_image is None:
                    logger.warning("Skipping page due to image load failure")
                    continue

                regions_list = batch["regions"][i]
                for r in regions_list:
                    lines = r.get("text_lines", [])
                    for line in lines:
                        if not allow_empty and not line.get("text"):
                            continue

                        augmented_cropped_images = []
                        cropped_image = image_processor.crop_from_image(
                            pil_image, line["coords"]
                        )
                        if cropped_image:
                            augmented_cropped_images.append(
                                (image_processor.encode_image(cropped_image), "{}")
                            )
                        if line_augment and line_augment > 0 and cropped_image:
                            for _ in range(line_augment):
                                unique = False
                                augmented_image = None
                                config = {}

                                max_retries = 10
                                retries = 0
                                while not unique and retries < max_retries:
                                    augmented_image, config = (
                                        image_processor.random_augment_image(
                                            cropped_image
                                        )
                                    )
                                    config_str = json.dumps(config)
                                    if config_str not in [
                                        c for _, c in augmented_cropped_images
                                    ]:
                                        unique = True
                                    retries += 1
                                if augmented_image:
                                    augmented_image_bytes = (
                                        image_processor.encode_image(augmented_image)
                                    )
                                    if augmented_image_bytes:
                                        augmented_cropped_images.append(
                                            (augmented_image_bytes, json.dumps(config))
                                        )
                        for img, config in augmented_cropped_images:
                            if img:
                                out["image"].append({"bytes": img, "path": None})
                                out["text"].append(line.get("text", ""))
                                out["line_id"].append(line["id"])
                                out["line_reading_order"].append(line["reading_order"])
                                out["line_coords"].append(line["coords"])
                                out["line_baseline"].append(line["baseline"])
                                out["line_augmentation"].append(
                                    config if config else "original"
                                )
                                out["region_id"].append(line["region_id"])
                                out["region_reading_order"].append(r["reading_order"])
                                out["region_type"].append(r["type"])
                                out["region_coords"].append(r["coords"])
                                out["filename"].append(batch["filename"][i])
                                out["project_name"].append(batch["project_name"][i])
            return out

        try:
            dataset = dataset.map(
                map_lines,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                # num_proc=1,
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            failed = True

        if failed:
            logger.warning("No success creating dataset; return None.")
            return None
        return dataset


class WindowExporter(BaseExporter):
    """Export sliding windows of text lines with configurable window size and overlap."""

    # Note: POST_FEATURES is the same regardless of window_size parameter
    POST_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "text": Value("string"),
            "window_size": Value("int64"),
            "window_index": Value("int64"),
            "line_ids": Value("string"),
            "line_reading_order": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
        }
    )

    def __init__(
        self,
        batch_size: int = 32,
        window_size: int = 2,
        overlap: int = 0,
    ):
        """
        Initialize the window exporter.

        Args:
            batch_size (int, optional): For dataset mapping. Defaults to 32.
            window_size: Number of lines per window (1, 2, 3, etc.)
            overlap: Number of lines to overlap between windows
        """
        super().__init__(batch_size)
        self.window_size = window_size
        self.overlap = overlap

        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")

    def process_dataset(
        self, dataset: Dataset, mask: bool = False, allow_empty: bool = False
    ) -> Dataset | None:
        """Export sliding windows of lines as separate dataset entries."""
        logger.info(
            f"Exporting window content with images (processed: {len(dataset)})."
        )
        failed: bool = False

        image_processor = ImageProcessor(mask_crop=mask)

        def map_windows(batch):
            """
            Function for dataset mapping.
            """
            out = {k: [] for k in self.POST_FEATURES}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                if pil_image is None:
                    logger.warning("Skipping page due to image load failure")
                    continue

                regions_list = batch["regions"][i] or []
                step = self.window_size - self.overlap
                if step < 1:
                    step = 1

                # AIDEV-NOTE: window per region to avoid windows crossing paragraph boundaries
                for r in regions_list:
                    lines_in_region = r.get("text_lines", [])
                    logger.debug(f"Length of lines in region: {len(lines_in_region)}")
                    idx = 0

                    for j in range(0, len(lines_in_region), step):
                        window_lines = lines_in_region[j : j + self.window_size]
                        logger.debug(
                            f"Length of window lines: {len(window_lines)} (j: {j}, idx: {idx})"
                        )
                        if not window_lines:
                            continue

                        window_text = "\n".join(
                            [line["text"] or "" for line in window_lines]
                        )
                        window_ids = ", ".join([line["id"] for line in window_lines])

                        all_coords = []
                        for line in window_lines:
                            all_coords.extend(line["coords"])

                        if not all_coords:
                            continue
                        logger.debug(f"Length of all coords: {len(all_coords)}")
                        crop_bytes = None
                        cropped_image = image_processor.crop_from_image(
                            pil_image, all_coords
                        )
                        if cropped_image:
                            crop_bytes = image_processor.encode_image(cropped_image)
                        if crop_bytes:
                            out["image"].append({"bytes": crop_bytes, "path": None})
                            out["text"].append(window_text)
                            out["window_size"].append(len(window_lines))
                            out["window_index"].append(idx)
                            out["line_ids"].append(window_ids)
                            out["line_reading_order"].append(
                                ", ".join(
                                    [
                                        str(line["reading_order"])
                                        for line in window_lines
                                    ]
                                )
                            )
                            out["filename"].append(str(batch["filename"][i]))
                            out["project_name"].append(str(batch["project_name"][i]))

                        idx += len(window_lines)
                        logger.debug("Next step.")
            return out

        try:
            dataset = dataset.map(
                map_windows,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
                load_from_cache_file=False,
                # num_proc=1,
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            failed = True

        if failed:
            logger.warning("No success creating dataset; return None.")
            return None
        return dataset
