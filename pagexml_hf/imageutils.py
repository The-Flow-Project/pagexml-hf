"""
Imageutils to handle the page/regions/line images more efficient.
"""
import numpy as np
import io
import traceback
from typing import List, Tuple, Dict, Any
from skimage import draw
from PIL import Image

from .logger import logger


class ImageProcessor:
    """
    Class to handle the page/regions/line images more efficient
    and perform orientation fix and cropping.
    """

    def __init__(
            self,
            mask_crop: bool = False,
            min_width: int = 0,
            min_height: int = 0,
            line_based: bool = False,
            allow_empty: bool = False,
    ):
        self.mask_crop = mask_crop
        self.min_width = min_width
        self.min_height = min_height
        self.line_based = line_based
        self.allow_empty = allow_empty

        self.processed_count: int = 0
        self.skipped_count: int = 0
        self.failed_images: List[Tuple(str)] = []
        logger.debug(f"ImageProcessor initialized")

    def process_entry_cropping(self, batch):
        """
        Function to map iterable datasets
        """
        try:
            result: List[List[Dict[str, Any]]] = []
            for i, b in enumerate(batch["image"]):
                cropped_images = self._crop(
                    b["bytes"],
                    batch["regions"][i],
                    batch["filename"][i],
                    batch["project_name"][i],
                )
                logger.debug(f"Cropping image: {len([c for c in cropped_images if c is not None])}")
                result.append(cropped_images)
            return {"cropped_areas": result}
        except Exception as e:
            logger.warning(f"Warning: Error cropping image {batch['filename']}: {e}")

    def _crop(
            self,
            image: bytes,
            regions: List[Dict[str, Any]],
            filename: str,
            project: str,
    ) -> List[Dict[str, Any]]:
        result = []

        logger.debug(f"Cropping image: {filename}")
        for region in regions:
            if self.line_based:
                for line in region["text_lines"]:
                    if len(line["text"]) > 0 or self.allow_empty:
                        result_line = {}
                        coords = line["coords"]
                        if not coords:
                            logger.warning("Warning: No coordinates provided for cropping.")
                            continue
                        cropped_line = self.crop_with_coords(image, coords)
                        if cropped_line:
                            logger.debug(f"Cropping line: {line['id']}")
                            result_line["image"] = {"bytes": cropped_line, "path": None}
                            result_line["text"] = line["text"]
                            result_line["line_id"] = line["id"]
                            result_line["line_reading_order"] = line["reading_order"]
                            result_line["region_id"] = region["id"]
                            result_line["region_reading_order"] = region["reading_order"]
                            result_line["region_type"] = region["type"]
                            result_line["filename"] = filename
                            result_line["project_name"] = project
                            result.append(result_line)
                        else:
                            logger.warning(f"Didn't success cropping line.")
                            continue
            else:
                if len(region["full_text"]) > 0 or self.allow_empty:
                    coords = region["coords"]
                    if not coords:
                        logger.warning("Warning: No coordinates provided for cropping.")
                        self.skipped_count += 1
                        continue
                    cropped_region = self.crop_with_coords(image, coords)
                    result_region = {}
                    if cropped_region:
                        result_region["image"] = {"bytes": cropped_region, "path": None}
                        result_region["text"] = region["full_text"]
                        result_region["region_type"] = region["type"]
                        result_region["region_id"] = region["id"]
                        result_region["region_reading_order"] = region["reading_order"]
                        result_region["filename"] = filename
                        result_region["project_name"] = project
                        self.processed_count += 1
                        result.append(result_region)
                    else:
                        self.skipped_count += 1
                        logger.warning(f"Didn't success cropping region.")
                        continue
        return result

    def crop_with_coords(self, image: bytes, coords: List[Tuple[int, int]]) -> bytes | None:
        """
        Simple cropping function
        """
        try:
            # Calculate bounding box for the coordinates
            image = Image.open(io.BytesIO(image))
            img_ndarray = np.array(image, dtype=np.uint8)
            # img_ndarray = np.frombuffer(image, dtype=np.uint8)

            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            min_x, max_x = max(0, min(x_coords)), min(img_ndarray.shape[1], max(x_coords))
            min_y, max_y = max(0, min(y_coords)), min(img_ndarray.shape[0], max(y_coords))

            if min_x >= max_x or min_y >= max_y:
                logger.warning(f"Warning: Invalid crop coordinates: ({min_x}, {min_y}, {max_x}, {max_y})")
                return None

            if self.min_width and int(max_x - min_x) < self.min_width:
                return None

            if self.min_height and int(max_y - min_y) < self.min_height:
                return None

            img_cropped = img_ndarray[min_y:max_y, min_x:max_x]

            if self.mask_crop:
                logger.debug("Cropping mask")
                shifted_coords = [(x - min_x, y - min_y) for (x, y) in coords]
                mask_img = np.zeros((img_cropped.shape[0], img_cropped.shape[1]), dtype=np.uint8)
                pts = np.array([shifted_coords], dtype=np.int32)
                rr, cc = draw.polygon(pts, mask_img.shape)
                mask_img[rr, cc] = 255

                result = np.where(mask_img[:, :, None] == 255, img_cropped, 0)
                return result.tobytes()

            return img_cropped.tobytes()

        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Warning: Error cropping region: {e}")
            return None

    def correct_orientation(self, batch):
        """
        Check and correct the orientation of an image.
        """
        logger.debug(f"Start mapping for orientation correction")
        result = []
        for i, b in enumerate(batch["image"]):
            logger.debug(f"Checking orientation of image: {i + 1} (of {len(batch['image'])})")
            image = Image.open(io.BytesIO(b["bytes"]))
            image = image.convert("RGB")
            try:
                exif = image.getexif()

                if exif:
                    # key 274 = orientation, returns 1 if not existing
                    orientation = exif.get(274, 1)

                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                self.processed_count += 1
            except (AttributeError, KeyError, IndexError):
                self.skipped_count += 1
                self.failed_images.append((batch["filename"][i], "Couldn't correct orientation"))
                pass

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            b = {"bytes": image_bytes, "path": None}
            result.append(b)
            buffer.close()
            image.close()
        logger.debug(f"Finished mapping for orientation correction")
        batch["image"] = result
        return batch


def calculate_bounding_box(
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
