"""
PyTest Tests for ImageProcessor
================================
"""

import pytest
from PIL import Image
import io

from pagexml_hf.imageutils import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    @pytest.fixture
    def processor_no_mask(self):
        """ImageProcessor without mask cropping."""
        return ImageProcessor(mask_crop=False)

    @pytest.fixture
    def processor_with_mask(self):
        """ImageProcessor with mask cropping."""
        return ImageProcessor(mask_crop=True)

    @pytest.fixture
    def processor_with_filters(self):
        """ImageProcessor with size filters."""
        return ImageProcessor(min_width=50, min_height=50)

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image (100x100 RGB)."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Draw a red square in the middle
        pixels = img.load()
        for x in range(40, 60):
            for y in range(40, 60):
                pixels[x, y] = (255, 0, 0)
        return img

    @pytest.fixture
    def sample_coords(self):
        """Sample coordinates for a rectangular region."""
        return [(20, 20), (80, 20), (80, 80), (20, 80)]

    @pytest.fixture
    def sample_triangle_coords(self):
        """Sample coordinates for a triangular region."""
        return [(30, 30), (70, 30), (50, 70)]

    def test_initialization_default(self):
        """Test ImageProcessor initialization with defaults."""
        processor = ImageProcessor()
        assert processor.mask_crop is False
        assert processor.min_width is None
        assert processor.min_height is None

    def test_initialization_with_params(self):
        """Test ImageProcessor initialization with parameters."""
        processor = ImageProcessor(
            mask_crop=True,
            min_width=100,
            min_height=50
        )
        assert processor.mask_crop is True
        assert processor.min_width == 100
        assert processor.min_height == 50

    def test_load_and_fix_orientation(self, processor_no_mask, sample_image):
        """Test loading and fixing image orientation."""
        # Convert to bytes
        img_bytes_io = io.BytesIO()
        sample_image.save(img_bytes_io, format='JPEG')
        img_bytes = img_bytes_io.getvalue()

        # Load image
        loaded_img = processor_no_mask.load_and_fix_orientation(img_bytes)

        assert isinstance(loaded_img, Image.Image)
        assert loaded_img.size == (100, 100)
        assert loaded_img.mode == 'RGB'

    def test_encode_image(self, processor_no_mask, sample_image):
        """Test image encoding to JPEG bytes."""
        encoded = processor_no_mask.encode_image(sample_image)

        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Verify it's a valid JPEG
        decoded = Image.open(io.BytesIO(encoded))
        assert decoded.size == sample_image.size

    def test_crop_from_image_no_mask(
            self,
            processor_no_mask,
            sample_image,
            sample_coords
    ):
        """Test cropping without mask."""
        result = processor_no_mask.crop_from_image(sample_image, sample_coords)

        assert result is not None
        assert isinstance(result, bytes)

        # Check cropped dimensions
        cropped = Image.open(io.BytesIO(result))
        assert cropped.size == (60, 60)  # 80-20 = 60

    def test_crop_from_image_with_mask(
            self,
            processor_with_mask,
            sample_image,
            sample_triangle_coords
    ):
        """Test cropping with mask (white background)."""
        result = processor_with_mask.crop_from_image(
            sample_image,
            sample_triangle_coords
        )

        assert result is not None
        assert isinstance(result, bytes)

        # Verify it's a valid image
        cropped = Image.open(io.BytesIO(result))
        assert cropped.size[0] > 0
        assert cropped.size[1] > 0

        # Check that background pixels are white (or close to white due to JPEG)
        pixels = cropped.load()
        # Top-left corner should be outside triangle, thus white
        # Note: JPEG compression may alter values slightly
        top_left = pixels[0, 0]
        assert all(c > 200 for c in top_left), "Background should be white"

    def test_crop_from_image_with_min_width_filter(
            self,
            processor_with_filters,
            sample_image
    ):
        """Test cropping with min_width filter."""
        # Small region (30x30) - should be filtered out
        small_coords = [(10, 10), (40, 10), (40, 40), (10, 40)]

        result = processor_with_filters.crop_from_image(
            sample_image,
            small_coords
        )

        # Should return None because width (30) < min_width (50)
        assert result is None

    def test_crop_from_image_with_valid_size(
            self,
            processor_with_filters,
            sample_image
    ):
        """Test cropping with size that passes filters."""
        # Large region (60x60) - should pass
        large_coords = [(10, 10), (70, 10), (70, 70), (10, 70)]

        result = processor_with_filters.crop_from_image(
            sample_image,
            large_coords
        )

        assert result is not None
        assert isinstance(result, bytes)

    def test_crop_from_image_invalid_coords(
            self,
            processor_no_mask,
            sample_image
    ):
        """Test cropping with invalid coordinates."""
        # Coordinates outside image bounds
        invalid_coords = [(200, 200), (300, 200), (300, 300), (200, 300)]

        result = processor_no_mask.crop_from_image(
            sample_image,
            invalid_coords
        )

        # Should return None for invalid coords
        assert result is None

    def test_crop_from_image_empty_coords(
            self,
            processor_no_mask,
            sample_image
    ):
        """Test cropping with empty coordinates."""
        result = processor_no_mask.crop_from_image(sample_image, [])
        assert result is None

    def test_crop_from_image_polygon(
            self,
            processor_with_mask,
            sample_image
    ):
        """Test cropping with complex polygon coordinates."""
        # Pentagon shape
        pentagon_coords = [
            (50, 20),
            (80, 40),
            (70, 70),
            (30, 70),
            (20, 40)
        ]

        result = processor_with_mask.crop_from_image(
            sample_image,
            pentagon_coords
        )

        assert result is not None
        assert isinstance(result, bytes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
