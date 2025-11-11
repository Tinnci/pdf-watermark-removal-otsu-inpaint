"""Tests for watermark removal functionality."""

import numpy as np
import cv2
from pdf_watermark_removal.watermark_detector import WatermarkDetector
from pdf_watermark_removal.watermark_remover import WatermarkRemover


def create_test_image_with_watermark(width=200, height=200):
    """Create a test image with synthetic watermark."""
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


def test_watermark_detection():
    """Test watermark detection."""
    image = create_test_image_with_watermark()
    detector = WatermarkDetector()
    mask = detector.detect_watermark_mask(image)
    
    assert mask is not None
    assert mask.shape[:2] == image.shape[:2]
    assert np.any(mask > 0), "Watermark mask should not be empty"


def test_watermark_removal():
    """Test watermark removal."""
    image = create_test_image_with_watermark()
    remover = WatermarkRemover()
    result = remover.remove_watermark(image)
    
    assert result is not None
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_mask_refinement():
    """Test mask refinement."""
    detector = WatermarkDetector()
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)
    
    refined = detector.refine_mask(mask, min_area=100)
    
    assert refined is not None
    assert np.count_nonzero(refined) > 0


if __name__ == "__main__":
    print("Running tests...")
    test_watermark_detection()
    print("✓ Watermark detection test passed")
    
    test_watermark_removal()
    print("✓ Watermark removal test passed")
    
    test_mask_refinement()
    print("✓ Mask refinement test passed")
    
    print("\nAll tests passed!")
