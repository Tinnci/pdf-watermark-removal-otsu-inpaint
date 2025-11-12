"""QR code detection and classification for PDF watermark removal."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np


class QRCodeType(Enum):
    """Types of QR codes based on content analysis."""

    URL = "url"
    TEXT = "text"
    CONTACT = "contact"
    WIFI = "wifi"
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    LOCATION = "location"
    CALENDAR = "calendar"
    UNKNOWN = "unknown"


@dataclass
class QRCodeInfo:
    """Information about a detected QR code."""

    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    qr_type: QRCodeType
    content: str
    category: str  # For grouping similar QR codes
    page_num: int = 1  # Track which page the QR code was found on


def _is_phone_number(text: str) -> bool:
    """Check if text looks like a phone number."""
    # Remove common separators and check for digits
    cleaned = "".join(c for c in text if c.isdigit() or c in "+()- ")
    digits = "".join(c for c in cleaned if c.isdigit())

    # Phone numbers typically have 7-15 digits
    return 7 <= len(digits) <= 15 and any(c.isdigit() for c in text)


def _is_coordinates(text: str) -> bool:
    """Check if text looks like geographic coordinates."""
    # Simple coordinate pattern matching
    import re

    coord_pattern = r"-?\d+\.\d+\s*,\s*-?\d+\.\d+"
    return bool(re.search(coord_pattern, text))


def _is_likely_advertisement(text: str) -> bool:
    """Heuristic to identify advertisement QR codes."""
    # Common advertisement keywords
    ad_keywords = [
        "promo",
        "discount",
        "sale",
        "offer",
        "deal",
        "coupon",
        "advertisement",
        "ad",
    ]
    text_lower = text.lower()

    return any(keyword in text_lower for keyword in ad_keywords)


def _is_likely_documentation(text: str) -> bool:
    """Heuristic to identify documentation QR codes."""
    # Common documentation patterns
    doc_keywords = [
        "help",
        "support",
        "manual",
        "guide",
        "documentation",
        "info",
        "about",
    ]
    text_lower = text.lower()

    return any(keyword in text_lower for keyword in doc_keywords)


def _is_url(content: str) -> bool:
    return content.startswith(("http://", "https://", "www."))


def _is_wifi(content: str) -> bool:
    return content.startswith("WIFI:")


def _is_contact(content: str) -> bool:
    return content.startswith("BEGIN:VCARD") or "VCARD" in content.upper()


def _is_email(content: str) -> bool:
    return content.startswith("mailto:") or "@" in content


def _is_phone(content: str) -> bool:
    return content.startswith("tel:") or _is_phone_number(content)


def _is_sms(content: str) -> bool:
    return content.startswith("sms:") or content.startswith("SMSTO:")


def _is_location(content: str) -> bool:
    return content.startswith("geo:") or _is_coordinates(content)


def _is_calendar(content: str) -> bool:
    return content.startswith("BEGIN:VEVENT") or "VEVENT" in content.upper()


def _classify_text_content(content: str) -> str:
    if _is_likely_advertisement(content):
        return "advertisement"
    elif _is_likely_documentation(content):
        return "documentation"
    else:
        return "general"


class QRCodeDetector:
    """Detects and classifies QR codes in images using multiple methods."""

    def __init__(self, method: str = "opencv", verbose: bool = False):
        """Initialize QR code detector.

        Args:
            method: Detection method ('opencv' or 'pyzbar')
            verbose: Enable verbose logging
        """
        self.method = method
        self.verbose = verbose
        self._opencv_detector = None
        self._pyzbar_available = False

        self._init_detector()

    def _init_detector(self):
        """Initialize the selected detection method."""
        if self.method == "opencv":
            # OpenCV QRCodeDetector is built into opencv-python
            self._opencv_detector = cv2.QRCodeDetector()
            if self.verbose:
                print("✓ Initialized OpenCV QRCodeDetector")

        elif self.method == "pyzbar":
            try:
                import pyzbar.pyzbar as pyzbar  # noqa: F401

                self._pyzbar_available = True
                if self.verbose:
                    print("✓ Initialized Pyzbar detector")
            except ImportError:
                if self.verbose:
                    print("⚠ Pyzbar not available, falling back to OpenCV")
                self.method = "opencv"
                self._opencv_detector = cv2.QRCodeDetector()

    def detect_qr_codes(
        self, image_rgb: np.ndarray, page_num: int = 1
    ) -> List[QRCodeInfo]:
        """Detect QR codes in the given image.

        Args:
            image_rgb: Input image in RGB format
            page_num: Page number (1-indexed) for tracking

        Returns:
            List of detected QR code information
        """
        if self.method == "opencv":
            return self._detect_opencv(image_rgb, page_num)
        elif self.method == "pyzbar" and self._pyzbar_available:
            return self._detect_pyzbar(image_rgb, page_num)
        else:
            return []

    def _create_qrcode_info_from_detection(
        self, data: str, bbox_coords: np.ndarray, confidence: float, page_num: int = 1
    ) -> QRCodeInfo:
        """
        Helper to create a QRCodeInfo object from raw detection data.

        Args:
            data (str): Decoded QR code content.
            bbox_coords (np.ndarray): Bounding box coordinates (e.g., from OpenCV).
            confidence (float): Confidence score of the detection.
            page_num (int): Page number where QR code was detected.

        Returns:
            QRCodeInfo: An object containing structured QR code information.
        """
        # Calculate bounding box from corner points
        x_coords = bbox_coords[:, 0]
        y_coords = bbox_coords[:, 1]

        x = int(np.min(x_coords))
        y = int(np.min(y_coords))
        width = int(np.max(x_coords) - x)
        height = int(np.max(y_coords) - y)

        # Classify the QR code content
        qr_type, category = self._classify_qr_content(data)

        return QRCodeInfo(
            bbox=(x, y, width, height),
            confidence=confidence,
            qr_type=qr_type,
            content=data,
            category=category,
            page_num=page_num,  # NEW
        )

    def _detect_opencv(
        self, image_rgb: np.ndarray, page_num: int = 1
    ) -> List[QRCodeInfo]:
        """Detect QR codes using OpenCV's built-in detector."""
        qr_codes = []

        # Convert to grayscale for OpenCV detector
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        try:
            # Detect and decode QR codes
            data, bbox, rectified_image = self._opencv_detector.detectAndDecode(gray)

            if bbox is not None and len(bbox) > 0:
                # bbox is a numpy array of shape (1, 4, 2) - 4 corner points
                bbox_points = bbox[0]  # Get the first (and usually only) QR code

                qr_info = self._create_qrcode_info_from_detection(
                    data, bbox_points, 0.9, page_num
                )  # OpenCV doesn't provide confidence, use high default
                qr_codes.append(qr_info)

                if self.verbose:
                    print(
                        f"✓ Detected QR code: {qr_info.qr_type.value} at ({qr_info.bbox[0]}, {qr_info.bbox[1]}, {qr_info.bbox[2]}, {qr_info.bbox[3]})"
                    )

        except Exception as e:
            if self.verbose:
                print(f"⚠ OpenCV QR detection error: {e}")

        return qr_codes

    def _detect_pyzbar(
        self, image_rgb: np.ndarray, page_num: int = 1
    ) -> List[QRCodeInfo]:
        """Detect QR codes using Pyzbar library."""
        import pyzbar.pyzbar as pyzbar

        qr_codes = []

        try:
            # Convert to grayscale for pyzbar
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Detect barcodes and QR codes
            decoded_objects = pyzbar.decode(gray)

            for obj in decoded_objects:
                if obj.type == "QRCODE":
                    # Extract bounding box
                    x, y, width, height = obj.rect
                    bbox_coords = np.array(
                        [
                            [x, y],
                            [x + width, y],
                            [x + width, y + height],
                            [x, y + height],
                        ]
                    )

                    # Decode the data
                    try:
                        data = obj.data.decode("utf-8")
                    except UnicodeDecodeError:
                        data = obj.data.decode("latin-1", errors="ignore")

                    qr_info = self._create_qrcode_info_from_detection(
                        data, bbox_coords, 0.95, page_num
                    )  # Pyzbar is generally reliable
                    qr_codes.append(qr_info)

                    if self.verbose:
                        print(
                            f"✓ Detected QR code: {qr_info.qr_type.value} at ({qr_info.bbox[0]}, {qr_info.bbox[1]}, {qr_info.bbox[2]}, {qr_info.bbox[3]})"
                        )

        except Exception as e:
            if self.verbose:
                print(f"⚠ Pyzbar QR detection error: {e}")

        return qr_codes

    def _classify_qr_content(self, content: str) -> tuple:
        """Classify QR code content into type and category.

        Args:
            content: The decoded content of the QR code

        Returns:
            Tuple of (QRCodeType, category_string)
        """
        # Classify by type
        if _is_url(content):
            qr_type = QRCodeType.URL
        elif _is_wifi(content):
            qr_type = QRCodeType.WIFI
        elif _is_contact(content):
            qr_type = QRCodeType.CONTACT
        elif _is_email(content):
            qr_type = QRCodeType.EMAIL
        elif _is_phone(content):
            qr_type = QRCodeType.PHONE
        elif _is_sms(content):
            qr_type = QRCodeType.SMS
        elif _is_location(content):
            qr_type = QRCodeType.LOCATION
        elif _is_calendar(content):
            qr_type = QRCodeType.CALENDAR
        else:
            qr_type = QRCodeType.TEXT

        # Classify by category
        if qr_type == QRCodeType.URL:
            category = "website"
        elif qr_type == QRCodeType.CONTACT:
            category = "contact"
        elif qr_type == QRCodeType.EMAIL:
            category = "email"
        elif qr_type == QRCodeType.PHONE:
            category = "phone"
        elif qr_type in (QRCodeType.SMS, QRCodeType.LOCATION, QRCodeType.CALENDAR):
            category = qr_type.value
        else:
            category = _classify_text_content(content)

        return qr_type, category

    def group_qr_codes_by_category(
        self, qr_codes: List[QRCodeInfo]
    ) -> Dict[str, List[QRCodeInfo]]:
        """Group QR codes by their categories for batch processing.

        Args:
            qr_codes: List of detected QR codes

        Returns:
            Dictionary mapping categories to lists of QR codes
        """
        groups = {}
        for qr in qr_codes:
            if qr.category not in groups:
                groups[qr.category] = []
            groups[qr.category].append(qr)

        return groups

    def create_qr_mask(
        self, image_shape: Tuple[int, int], qr_codes: List[QRCodeInfo], padding: int = 5
    ) -> np.ndarray:
        """Create a binary mask for detected QR code regions.

        Args:
            image_shape: Shape of the image (height, width)
            qr_codes: List of detected QR codes
            padding: Additional padding around QR codes

        Returns:
            Binary mask where QR code regions are 255
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for qr in qr_codes:
            x, y, width, height = qr.bbox

            # Apply padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_shape[1], x + width + padding)
            y2 = min(image_shape[0], y + height + padding)

            # Fill the QR code region
            mask[y1:y2, x1:x2] = 255

        return mask

    def get_qr_summary(self, qr_codes: List[QRCodeInfo]) -> Dict[str, any]:
        """Get a summary of detected QR codes.

        Args:
            qr_codes: List of detected QR codes

        Returns:
            Summary dictionary with statistics
        """
        if not qr_codes:
            return {"count": 0, "categories": {}, "types": {}}

        categories = {}
        types = {}

        for qr in qr_codes:
            # Count categories
            categories[qr.category] = categories.get(qr.category, 0) + 1

            # Count types
            qr_type_name = qr.qr_type.value
            types[qr_type_name] = types.get(qr_type_name, 0) + 1

        return {"count": len(qr_codes), "categories": categories, "types": types}
