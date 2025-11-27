"""
Imageutils to handle the page/regions/line images more efficient.
"""
import numpy as np
import io
from typing import List, Tuple
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
    ):
        self.mask_crop = mask_crop
        self.min_width = min_width
        self.min_height = min_height

        logger.debug(f"ImageProcessor initialized")

    @staticmethod
    def load_and_fix_orientation(image_bytes: bytes) -> Image.Image:
        """
        Tool 1: Loads bytes and returns PIL Image with fixed orientation
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            exif = image.getexif()
            # key 274 = orientation, returns 1 if not existing
            orientation = exif.get(274, 1) if exif else 1

            if orientation in [3, 6, 8]:
                image = image.convert("RGB")
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)

            return image

        except Exception as e:
            logger.debug(f"Warning: Error loading image: {image_bytes}: {e}")
            return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def encode_image(image: Image.Image) -> bytes:
        """
        Tool 2: Saves the PIL Image to JPEG bytes.
        """
        with io.BytesIO() as buffer:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=95)
            return buffer.getvalue()

    def crop_from_image(self, image: Image.Image, coords: List[Tuple[int, int]]) -> bytes | None:
        """
        Tool 3: Crops the given image from the given coordinates.
        Returns JPEG bytes.
        """
        try:
            img_ndarray = np.array(image)

            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            min_x, max_x = max(0, min(x_coords)), min(img_ndarray.shape[1], max(x_coords))
            min_y, max_y = max(0, min(y_coords)), min(img_ndarray.shape[0], max(y_coords))

            if min_x >= max_x or min_y >= max_y:
                return None
            if self.min_width and (max_x - min_x) < self.min_width:
                return None
            if self.min_height and (max_y - min_y) < self.min_height:
                return None

            img_cropped = img_ndarray[min_y:max_y, min_x:max_x]

            if self.mask_crop:
                shifted_coords = [(x - min_x, y - min_y) for (x, y) in coords]
                mask_img = np.zeros((img_cropped.shape[0], img_cropped.shape[1]), dtype=np.uint8)
                pts = np.array([shifted_coords], dtype=np.int32)
                rr, cc = draw.polygon(pts, mask_img.shape)
                mask_img[rr, cc] = 255
                img_cropped = np.where(mask_img[:, :, None] == 255, img_cropped, 0)

            return self.encode_image(Image.fromarray(img_cropped))
        except Exception as e:
            logger.debug(f"Failed cropping: {e}")
            return None

