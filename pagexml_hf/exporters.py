"""
Exporters for converting parsed Transkribus data to different HuggingFace dataset formats.
"""

from abc import ABC, abstractmethod

from PIL import ImageFile
from datasets import (
    Dataset,
    Features,
    Value,
    Image as DatasetImage,
    List
)

from .logger import logger
from .imageutils import ImageProcessor

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

POLYGON_FEATURE = List(feature=List(feature=Value("int64")))

PRE_FEATURES = Features(
    {
        "image": DatasetImage(decode=False),
        "xml_content": Value("string"),
        "filename": Value("string"),
        "project_name": Value("string"),
        "image_width": Value("int64"),
        "image_height": Value("int64"),

        "regions": List(feature={
            "id": Value("string"),
            "type": Value("string"),
            "coords": POLYGON_FEATURE,

            "text_lines": List(feature={
                "id": Value("string"),
                "text": Value("string"),
                "coords": POLYGON_FEATURE,
                "baseline": POLYGON_FEATURE,
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
            batch_size: int = 32,
    ):
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
        logger.info(f"Exporting raw XML content with images... (number of pages: {len(dataset)})")

        image_processor = ImageProcessor()

        def map_raw(batch):
            """
            Map the raw XMLs for image correction
            """
            out_images = []
            for img_entry in batch["image"]:
                image = image_processor.load_and_fix_orientation(img_entry["bytes"])
                final_bytes = image_processor.encode_image(image)
                out_images.append({"bytes": final_bytes, "path": None})

            return {
                "image": out_images,
                "xml_content": batch["xml_content"],
                "filename": batch["filename"],
                "project_name": batch["project_name"],
            }

        logger.debug(f"Start mapping for orientation correction")
        try:
            dataset = dataset.map(
                map_raw,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
            )

            logger.debug(dataset.info)
            logger.debug(dataset.features)

            logger.info(f"Raw XML dataset prepared")
        except Exception as e:
            logger.error(f"Error preparing raw XML dataset: {e}")
            dataset = None
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

        logger.info(f"Exporting text content with images... (Parsed {len(dataset)} pages)")

        def process_batch(batch):
            """
            Generator for dataset creation.
            """
            image_processor = ImageProcessor()

            output_images = []
            output_texts = []

            logger.debug(f"Start mapping to add full_text")

            for i in range(len(batch["image"])):
                # ------ Text Extraction ------
                regions_list = batch["regions"][i]

                if regions_list and len(regions_list) > 0:
                    page_text_parts = []
                    for r in regions_list:
                        txt = r.get("full_text")
                        if txt:
                            page_text_parts.append(txt)

                    full_text = '\n'.join(page_text_parts)
                else:
                    full_text = ""

                output_texts.append(full_text)

                # ------- Handle Image Orientation ------
                img_entry = batch["image"][i]
                original_bytes = img_entry["bytes"]
                final_bytes = image_processor.load_and_fix_orientation(original_bytes)
                final_bytes = image_processor.encode_image(final_bytes)

                output_images.append({"bytes": final_bytes, "path": None})

            return {
                "image": output_images,
                "text": output_texts,
                "filename": batch["filename"],
                "project_name": batch["project_name"],
            }

        try:
            logger.debug(f"Map the provided dataset.")
            dataset = dataset.map(
                process_batch,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names,
            )
            logger.debug(dataset.info)
            logger.debug(dataset.features)
            logger.info(f"Text Dataset prepared")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset


class RegionExporter(BaseExporter):
    """Export individual regions as separate images with metadata."""

    POST_FEATURES = Features({
        "image": DatasetImage(decode=False),
        "text": Value("string"),
        "region_id": Value("string"),
        "region_reading_order": Value("int32"),
        "region_type": Value("string"),
        "filename": Value("string"),
        "project_name": Value("string"),
    })

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

        logger.info(f"Exporting Region XML content with images... (Processed: {len(dataset)})")

        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
        )

        def map_regions(batch):
            """
            Mapping the regions
            """
            logger.info(f"Start mapping to regions...")
            out = {k: [] for k in self.POST_FEATURES.keys()}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])

                regions_list = batch["regions"][i]
                if not regions_list:
                    continue

                for r in regions_list:
                    if not allow_empty and not r.get("full_text"):
                        continue

                    crop_bytes = image_processor.crop_from_image(pil_image, r["coords"])

                    if crop_bytes:
                        out["image"].append({"bytes": crop_bytes, "path": None})
                        out["text"].append(r.get("full_text", ""))
                        out["region_id"].append(r["id"])
                        out["region_reading_order"].append(r["reading_order"])
                        out["region_type"].append(r["type"])
                        out["filename"].append(batch["filename"][i])
                        out["project_name"].append(batch["project_name"][i])

            return out

        try:
            logger.debug(f"Map the provided dataset.")
            dataset = dataset.map(
                map_regions,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset


class LineExporter(BaseExporter):
    """Export individual text lines as separate images with metadata."""

    POST_FEATURES = Features({
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
        """Export each text line as a separate dataset entry."""
        logger.info(f"Exporting line content with images... (Processed: {len(dataset)})")

        image_processor = ImageProcessor(
            mask_crop=mask,
            min_width=min_width,
            min_height=min_height,
        )

        def map_lines(batch):
            """
            Mapping the lines and cropping them
            """
            out = {k: [] for k in self.POST_FEATURES.keys()}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])

                regions_list = batch["regions"][i]
                for r in regions_list:
                    lines = r.get("text_lines", [])
                    for line in lines:
                        if not allow_empty and not line.get("text"):
                            continue

                        crop_bytes = image_processor.crop_from_image(pil_image, line["coords"])

                        if crop_bytes:
                            out["image"].append({"bytes": crop_bytes, "path": None})
                            out["text"].append(line.get("text", ""))
                            out["line_id"].append(line["id"])
                            out["line_reading_order"].append(line["reading_order"])
                            out["region_id"].append(line["region_id"])
                            out["region_reading_order"].append(r["reading_order"])
                            out["region_type"].append(r["type"])
                            out["filename"].append(batch["filename"][i])
                            out["project_name"].append(batch["project_name"][i])
            return out

        try:
            dataset = dataset.map(
                map_lines,
                batched=True,
                batch_size=self.batch_size,
                features=self.POST_FEATURES,
                remove_columns=dataset.column_names
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

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
            self,
            dataset: Dataset,
            mask: bool = False,
            allow_empty: bool = False
    ) -> Dataset | None:
        """Export sliding windows of lines as separate dataset entries."""
        logger.info(f"Exporting window content with images (processed: {len(dataset)}).")

        image_processor = ImageProcessor(mask_crop=mask)

        def map_windows(batch):
            """
            Function for dataset mapping.
            """
            out = {k: [] for k in self.POST_FEATURES.keys()}

            for i, img_entry in enumerate(batch["image"]):
                pil_image = image_processor.load_and_fix_orientation(img_entry["bytes"])

                regions_list = batch["regions"][i] or []
                all_lines_in_page = []

                for r in regions_list:
                    all_lines_in_page.extend(r.get("text_lines", []))

                logger.debug(f"Length of all lines: {len(all_lines_in_page)}")
                step = self.window_size - self.overlap
                if step < 1:
                    step = 1

                idx = 0

                for j in range(0, len(all_lines_in_page), step):
                    window_lines = all_lines_in_page[j: j + self.window_size]
                    logger.debug(f"Length of window lines: {len(window_lines)} (j: {j}, idx: {idx})")
                    if not window_lines:
                        continue

                    window_text = "\n".join([l["text"] for l in window_lines])
                    window_ids = ", ".join([l["id"] for l in window_lines])

                    all_coords = []
                    for l in window_lines:
                        all_coords.extend(l["coords"])

                    if not all_coords:
                        continue
                    logger.debug(f"Length of all coords: {len(all_coords)}")
                    crop_bytes = image_processor.crop_from_image(pil_image, all_coords)
                    if crop_bytes:
                        out["image"].append({"bytes": crop_bytes, "path": None})
                        out["text"].append(window_text)
                        out["window_size"].append(len(window_lines))
                        out["window_index"].append(idx)
                        out["line_ids"].append(window_ids)
                        out["line_reading_order"].append(", ".join([str(l["reading_order"]) for l in window_lines]))
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
                remove_columns=dataset.column_names
            )
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            dataset = None

        return dataset
