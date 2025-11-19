"""Tests for watermark removal functionality using pytest."""

import cv2
import numpy as np
import pytest

from pdf_watermark_removal.watermark_detector import WatermarkDetector
from pdf_watermark_removal.watermark_remover import WatermarkRemover


@pytest.fixture
def watermark_image():
    """Create a test image with synthetic watermark."""
    width, height = 200, 200
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    cv2.putText(
        image,
        "WATERMARK",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (100, 100, 100),
        2,
    )
    return image


def test_watermark_detection(watermark_image):
    """Test watermark detection."""
    detector = WatermarkDetector()
    mask = detector.detect_watermark_mask(watermark_image)

    assert mask is not None
    assert mask.shape[:2] == watermark_image.shape[:2]
    assert np.any(mask > 0), "Watermark mask should not be empty"


def test_watermark_removal(watermark_image):
    """Test watermark removal."""
    remover = WatermarkRemover()
    result = remover.remove_watermark(watermark_image)

    assert result is not None
    assert result.shape == watermark_image.shape
    assert result.dtype == watermark_image.dtype


def test_mask_refinement():
    """Test mask refinement."""
    detector = WatermarkDetector()
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)

    refined = detector.refine_mask(mask, min_area=100)

    assert refined is not None
    assert np.count_nonzero(refined) > 0
