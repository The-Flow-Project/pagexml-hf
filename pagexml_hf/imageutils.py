"""
Imageutils to handle the page/regions/line images more efficient.
"""
import numpy as np
import io
import random
from typing import List, Tuple
from skimage import draw
from PIL import Image, ImageFilter

from loguru import logger


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

        self.augmentation_keys = [
            "rotation",
            # "blurring",
            "dilation",
            "erosion",
            "downscaling",
        ]

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
            image.save(buffer, format="PNG")
            return buffer.getvalue()

    def crop_from_image(self, image: Image.Image, coords: List[Tuple[int, int]]) -> Image.Image | None:
        """
        Tool 3: Crops the given image from the given coordinates.
        Returns pillow Image if successful, None otherwise.
        """
        try:
            logger.debug("Cropping image")
            logger.debug(f"Size: {image.size}")
            img_ndarray = np.array(image, dtype=np.uint8)

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
                mask_img = np.zeros((img_cropped.shape[0], img_cropped.shape[1]))
                y_coords_shifted = [y for (x, y) in shifted_coords]
                x_coords_shifted = [x for (x, y) in shifted_coords]
                logger.debug(
                    f"Mask crop: shape={mask_img.shape}, "
                    f"x_range=[{min(x_coords_shifted)},{max(x_coords_shifted)}], "
                    f"y_range=[{min(y_coords_shifted)},{max(y_coords_shifted)}], "
                    f"n_points={len(shifted_coords)}"
                )
                rr, cc = draw.polygon(y_coords_shifted, x_coords_shifted, shape=mask_img.shape)
                mask_img[rr, cc] = 255
                img_cropped = np.where(mask_img[:, :, None] == 255, img_cropped, 255)

            return Image.fromarray(img_cropped)
        except Exception as e:
            logger.debug(f"Failed cropping: {e}")
            return None

    def augment_image(self, image: Image.Image, config: dict[str, int | float]) -> Image.Image | None:
        """
        Tool 4: Augments the given PIL Image.
        """
        try:
            # Ensure RGB mode before augmenting
            if image.mode != "RGB":
                image = image.convert("RGB")

            for key, value in config.items():
                if key not in self.augmentation_keys:
                    continue

                if key == "rotation" and -10 <= value <= 10:
                    value = float(value)
                    image = image.rotate(
                        value,
                        resample=Image.BICUBIC,
                        expand=True,
                        fillcolor=(255, 255, 255),
                    )
                elif key == "blurring":
                    image = image.filter(ImageFilter.GaussianBlur(radius=value))
                elif key == "dilation":
                    image = image.filter(ImageFilter.MaxFilter(size=int(value)))
                elif key == "erosion":
                    image = image.filter(ImageFilter.MinFilter(size=int(value)))
                elif key == "downscaling" and 1 < value <= 4:
                    w, h = image.size
                    new_w, new_h = max(1, w // int(value)), max(1, h // int(value))
                    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

            return image
        except Exception as e:
            logger.debug(f"Failed augmenting: {e}")
            return None

    def random_augment_image(self, image: Image.Image) -> tuple[Image.Image | None, dict]:
        """
        Tool 5: Randomly Augments the given PIL Image.
        """
        num_to_choose = random.randint(1, min(len(self.augmentation_keys), 2))  # max two filters
        selected = random.sample(self.augmentation_keys, num_to_choose)
        config = {}
        dil_er_done = False  # do either dilation or erosion, not both

        for i in selected:
            if i == "rotation":
                angle = random.choice([a for a in range(-10, 11) if a != 0])
                config[i] = angle
            elif (i == "dilation" or i == "erosion") and not dil_er_done:
                config[i] = 3  # more than 3 is too much
                dil_er_done = True
            elif i == "downscaling":
                config[i] = random.choice([2, 3])

        if random.choice([True, False]):  # add blurring with 50% chance
            config["blurring"] = 0.2

        logger.debug(f"Augmenting image with augmentations: {list(config.keys())}")
        return self.augment_image(image, config), config
