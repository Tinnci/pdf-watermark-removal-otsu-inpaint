#!/usr/bin/env python3
"""Test script for QR code detection feature."""

import cv2
import numpy as np
from src.pdf_watermark_removal.qr_detector import QRCodeDetector, QRCodeSelector
from src.pdf_watermark_removal.watermark_detector import WatermarkDetector


def create_test_image_with_qr():
    """Create a test image with a simple QR code pattern."""
    # Create a white background
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a simple black square to simulate a QR code
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)
    
    # Add some white squares inside to simulate QR code pattern
    cv2.rectangle(image, (70, 70), (90, 90), (255, 255, 255), -1)
    cv2.rectangle(image, (110, 70), (130, 90), (255, 255, 255), -1)
    cv2.rectangle(image, (70, 110), (90, 130), (255, 255, 255), -1)
    
    return image


def test_qr_detector():
    """Test QR code detection."""
    print("Testing QR Code Detection Feature...")
    
    # Test with OpenCV detector
    print("\n1. Testing OpenCV QR Detector:")
    detector = QRCodeDetector(method="opencv", verbose=True)
    
    # Create test image
    test_image = create_test_image_with_qr()
    
    # Try to detect QR codes
    qr_codes = detector.detect_qr_codes(test_image)
    print(f"Detected {len(qr_codes)} QR codes")
    
    if qr_codes:
        for i, qr in enumerate(qr_codes):
            print(f"  QR Code {i+1}:")
            print(f"    Type: {qr.qr_type.value}")
            print(f"    Category: {qr.category}")
            print(f"    Content: {qr.content}")
            print(f"    Bounding Box: {qr.bbox}")
    
    # Test QR code selector
    print("\n2. Testing QR Code Selector:")
    selector = QRCodeSelector(verbose=True)
    
    # Create some mock QR codes for testing
    from src.pdf_watermark_removal.qr_detector import QRCodeInfo, QRCodeType
    
    mock_qr_codes = [
        QRCodeInfo(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            qr_type=QRCodeType.URL,
            content="https://example.com/advertisement",
            category="advertisement"
        ),
        QRCodeInfo(
            bbox=(100, 100, 50, 50),
            confidence=0.95,
            qr_type=QRCodeType.TEXT,
            content="Document help information",
            category="documentation"
        ),
        QRCodeInfo(
            bbox=(200, 200, 50, 50),
            confidence=0.8,
            qr_type=QRCodeType.UNKNOWN,
            content="Unknown content",
            category="unknown"
        )
    ]
    
    # Test selection (this would normally be interactive)
    print(f"Testing with {len(mock_qr_codes)} mock QR codes:")
    for qr in mock_qr_codes:
        print(f"  - {qr.category}: {qr.content}")
    
    # Test grouping
    grouped = selector._group_and_summarize(mock_qr_codes)
    print(f"\nGrouped by category: {list(grouped.keys())}")
    
    # Test removal mask creation
    mask = selector.create_qr_removal_mask(test_image.shape, mock_qr_codes[:2])
    print(f"Created removal mask with {np.count_nonzero(mask)} pixels")
    
    print("\n✓ QR Code Detection Feature Test Complete!")


def test_watermark_detector_with_qr():
    """Test watermark detector with QR code detection enabled."""
    print("\n\nTesting Watermark Detector with QR Code Detection...")
    
    # Create test image
    test_image = create_test_image_with_qr()
    
    # Initialize detector with QR code detection
    detector = WatermarkDetector(
        detect_qr_codes=True,
        qr_detection_method="opencv",
        remove_all_qr_codes=False,
        qr_code_categories_to_remove=["advertisement", "unknown"],
        verbose=True
    )
    
    # Detect watermark mask (should include QR codes)
    mask = detector.detect_watermark_mask(test_image)
    
    print(f"Detected watermark mask with {np.count_nonzero(mask)} pixels")
    
    # Get QR codes
    qr_codes = detector.get_detected_qr_codes()
    print(f"Detected {len(qr_codes)} QR codes")
    
    print("✓ Watermark Detector with QR Codes Test Complete!")


if __name__ == "__main__":
    test_qr_detector()
    test_watermark_detector_with_qr()