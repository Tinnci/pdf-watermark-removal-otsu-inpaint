"""Test QR code detection functionality using pytest."""

import cv2
import numpy as np
import pytest

from pdf_watermark_removal.qr_detector import QRCodeDetector, QRCodeInfo, QRCodeType
from pdf_watermark_removal.qr_selector import QRCodeSelector
from pdf_watermark_removal.watermark_detector import WatermarkDetector


@pytest.fixture
def test_qr_image():
    """Create a simple test image with QR code-like patterns."""
    # Create white background
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Draw black square to simulate QR code
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)

    # Add white squares to simulate QR code pattern
    cv2.rectangle(image, (60, 60), (80, 80), (255, 255, 255), -1)
    cv2.rectangle(image, (120, 60), (140, 80), (255, 255, 255), -1)
    cv2.rectangle(image, (60, 120), (80, 140), (255, 255, 255), -1)

    return image


def test_qr_detector_opencv(test_qr_image):
    """Test OpenCV QR code detector."""
    detector = QRCodeDetector(method="opencv", verbose=False)

    # Test detection
    qr_codes = detector.detect_qr_codes(test_qr_image)

    # Test should work even with simple pattern (OpenCV may or may not detect)
    assert isinstance(qr_codes, list)

    # Test QR code info structure if any detected
    for qr in qr_codes:
        assert isinstance(qr, QRCodeInfo)
        assert hasattr(qr, "bbox")
        assert hasattr(qr, "qr_type")
        assert hasattr(qr, "content")
        assert hasattr(qr, "category")


def test_qr_classification():
    """Test QR code content classification."""
    detector = QRCodeDetector(method="opencv", verbose=False)

    # Test URL classification
    url_type, url_category = detector._classify_qr_content("https://example.com")
    assert url_type == QRCodeType.URL
    assert url_category == "website"

    # Test advertisement detection
    ad_type, ad_category = detector._classify_qr_content("Get 50% off! Amazing deal!")
    assert ad_type == QRCodeType.TEXT
    assert ad_category == "advertisement"

    # Test documentation detection
    doc_type, doc_category = detector._classify_qr_content("Help manual for users")
    assert doc_type == QRCodeType.TEXT
    assert doc_category == "documentation"

    # Test email detection
    email_type, email_category = detector._classify_qr_content(
        "mailto:test@example.com"
    )
    assert email_type == QRCodeType.EMAIL
    assert email_category == "email"


def test_qr_grouping():
    """Test QR code grouping functionality."""
    # Create mock QR codes
    mock_qr_codes = [
        QRCodeInfo(
            (0, 0, 10, 10), 0.9, QRCodeType.URL, "https://ad1.com", "advertisement"
        ),
        QRCodeInfo(
            (10, 10, 10, 10), 0.8, QRCodeType.URL, "https://ad2.com", "advertisement"
        ),
        QRCodeInfo(
            (20, 20, 10, 10), 0.95, QRCodeType.TEXT, "Help info", "documentation"
        ),
        QRCodeInfo(
            (30, 30, 10, 10), 0.85, QRCodeType.EMAIL, "test@example.com", "email"
        ),
    ]

    detector = QRCodeDetector(method="opencv", verbose=False)
    grouped = detector.group_qr_codes_by_category(mock_qr_codes)

    assert "advertisement" in grouped
    assert "documentation" in grouped
    assert "email" in grouped
    assert len(grouped["advertisement"]) == 2
    assert len(grouped["documentation"]) == 1
    assert len(grouped["email"]) == 1


def test_qr_mask_creation():
    """Test QR code mask creation."""
    # Create mock QR codes
    mock_qr_codes = [
        QRCodeInfo(
            (10, 10, 20, 20),
            0.9,
            QRCodeType.URL,
            "https://example.com",
            "advertisement",
        ),
        QRCodeInfo((100, 100, 30, 30), 0.8, QRCodeType.TEXT, "Help", "documentation"),
    ]

    detector = QRCodeDetector(method="opencv", verbose=False)
    mask = detector.create_qr_mask((200, 200, 3), mock_qr_codes, padding=5)

    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert np.any(mask > 0)  # Should have some masked regions


def test_watermark_detector_with_qr(test_qr_image):
    """Test watermark detector with QR code detection enabled."""
    # Test with QR detection enabled
    detector = WatermarkDetector(
        detect_qr_codes=True,
        qr_detection_method="opencv",
        remove_all_qr_codes=False,
        qr_code_categories_to_remove=["advertisement"],
        verbose=False,
    )

    # Test watermark mask detection (includes QR codes)
    mask = detector.detect_watermark_mask(test_qr_image)

    assert mask is not None
    assert mask.shape[:2] == test_qr_image.shape[:2]
    assert mask.dtype == np.uint8

    # Test QR code retrieval
    qr_codes = detector.get_detected_qr_codes()
    assert isinstance(qr_codes, list)


def test_qr_selector():
    """Test QR code selector functionality."""
    selector = QRCodeSelector(verbose=False)

    # Create mock QR codes
    mock_qr_codes = [
        QRCodeInfo(
            (0, 0, 10, 10), 0.9, QRCodeType.URL, "https://ad.com", "advertisement"
        ),
        QRCodeInfo(
            (10, 10, 10, 10), 0.8, QRCodeType.TEXT, "Help info", "documentation"
        ),
        QRCodeInfo((20, 20, 10, 10), 0.85, QRCodeType.UNKNOWN, "Unknown", "unknown"),
    ]

    # Test preset functionality
    aggressive_codes = selector.get_removals_by_preset("aggressive", mock_qr_codes)
    assert len(aggressive_codes) == 3  # Should remove all

    conservative_codes = selector.get_removals_by_preset("conservative", mock_qr_codes)
    assert len(conservative_codes) == 2  # Should remove ads and unknown

    ads_only_codes = selector.get_removals_by_preset("ads_only", mock_qr_codes)
    assert len(ads_only_codes) == 1  # Should remove only ads
