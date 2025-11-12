#!/usr/bin/env python3
"""Test script to verify the "Protect First, Refine Second" approach."""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pdf_watermark_removal.watermark_detector import WatermarkDetector

def create_test_image_with_text():
    """Create a test image with watermark and text to test protection order."""
    # Create a white background
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add light gray watermark (simulating the issue)
    watermark_color = (200, 200, 200)  # Light gray
    cv2.rectangle(image, (50, 50), (350, 150), watermark_color, -1)
    
    # Add dark text that should be protected
    text_color = (0, 0, 0)  # Pure black
    cv2.putText(image, "PROTECT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    cv2.putText(image, "TEXT", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    
    return image

def test_protection_order():
    """Test that protection happens before morphological operations."""
    print("Testing 'Protect First, Refine Second' approach...")
    
    # Create test image
    test_image = create_test_image_with_text()
    print(f"Test image shape: {test_image.shape}")
    
    # Test with color-based detection (where our fix should be applied)
    detector = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),  # Specify watermark color
        color_tolerance=30,
        kernel_size=5,  # Larger kernel to test potential bleeding
        protect_text=True,
        verbose=True
    )
    
    print("\n1. Detecting watermark mask with color-based method...")
    mask = detector.detect_watermark_mask(test_image)
    
    print(f"\n2. Mask statistics:")
    print(f"   - Mask shape: {mask.shape}")
    print(f"   - Non-zero pixels: {np.count_nonzero(mask)}")
    print(f"   - Total pixels: {mask.size}")
    print(f"   - Coverage: {(np.count_nonzero(mask) / mask.size) * 100:.2f}%")
    
    # Check if text areas are properly protected
    # The text should not be part of the watermark mask
    text_area = mask[80:120, 100:300]  # Approximate text region
    text_pixels = np.count_nonzero(text_area)
    print(f"\n3. Text protection check:")
    print(f"   - Text area shape: {text_area.shape}")
    print(f"   - Text pixels in mask: {text_pixels}")
    print(f"   - Text protection: {'PASS' if text_pixels == 0 else 'FAIL'}")
    
    # Test with automatic detection (should also have protection)
    print("\n4. Testing automatic detection mode...")
    detector_auto = WatermarkDetector(
        detection_method="traditional",
        auto_detect_color=True,  # Let it detect color automatically
        kernel_size=5,
        protect_text=True,
        verbose=True
    )
    
    mask_auto = detector_auto.detect_watermark_mask(test_image)
    text_pixels_auto = np.count_nonzero(mask_auto[80:120, 100:300])
    print(f"   - Auto mode text protection: {'PASS' if text_pixels_auto == 0 else 'FAIL'}")
    
    # Visual comparison
    print("\n5. Creating visualization...")
    comparison = np.hstack([test_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_auto, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite("test_protection_results.png", comparison)
    print("   - Saved visualization to test_protection_results.png")
    
    # Overall result
    both_passed = (text_pixels == 0) and (text_pixels_auto == 0)
    print(f"\n6. Overall result: {'ALL TESTS PASSED' if both_passed else 'SOME TESTS FAILED'}")
    
    return both_passed

def test_morphological_operations_order():
    """Test that morphological operations happen after protection."""
    print("\n\nTesting morphological operations order...")
    
    # Create a test image with small gaps in watermark
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    
    # Add watermark with intentional gaps
    watermark_color = (200, 200, 200)
    cv2.rectangle(image, (20, 20), (60, 80), watermark_color, -1)
    cv2.rectangle(image, (80, 20), (120, 80), watermark_color, -1)  # Gap between 60-80
    
    # Add text that should be protected
    cv2.putText(image, "TEXT", (130, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    detector = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=25,
        kernel_size=7,  # Large kernel to test gap filling
        protect_text=True,
        verbose=True
    )
    
    mask = detector.detect_watermark_mask(image)
    
    # Check if gaps are filled (morphological closing worked)
    gap_area = mask[30:70, 65:75]  # Area between the two rectangles
    gap_filled = np.count_nonzero(gap_area) > (gap_area.size * 0.5)
    
    # Check if text is still protected
    text_area = mask[40:70, 125:180]
    text_protected = np.count_nonzero(text_area) == 0
    
    print(f"   - Gap filling: {'PASS' if gap_filled else 'FAIL'}")
    print(f"   - Text protection: {'PASS' if text_protected else 'FAIL'}")
    
    return gap_filled and text_protected

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING 'PROTECT FIRST, REFINE SECOND' APPROACH")
    print("=" * 60)
    
    test1_passed = test_protection_order()
    test2_passed = test_morphological_operations_order()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Basic protection test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Morphological order test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Overall: {'ALL TESTS PASSED' if test1_passed and test2_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)