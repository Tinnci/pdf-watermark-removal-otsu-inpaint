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

    def detect_qr_codes(self, image_rgb: np.ndarray) -> List[QRCodeInfo]:
        """Detect QR codes in the given image.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            List of detected QR code information
        """
        if self.method == "opencv":
            return self._detect_opencv(image_rgb)
        elif self.method == "pyzbar" and self._pyzbar_available:
            return self._detect_pyzbar(image_rgb)
        else:
            return []

    def _detect_opencv(self, image_rgb: np.ndarray) -> List[QRCodeInfo]:
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

                # Calculate bounding box from corner points
                x_coords = bbox_points[:, 0]
                y_coords = bbox_points[:, 1]

                x = int(np.min(x_coords))
                y = int(np.min(y_coords))
                width = int(np.max(x_coords) - x)
                height = int(np.max(y_coords) - y)

                # Classify the QR code content
                qr_type, category = self._classify_qr_content(data)

                qr_info = QRCodeInfo(
                    bbox=(x, y, width, height),
                    confidence=0.9,  # OpenCV doesn't provide confidence, use high default
                    qr_type=qr_type,
                    content=data,
                    category=category,
                )

                qr_codes.append(qr_info)

                if self.verbose:
                    print(
                        f"✓ Detected QR code: {qr_type.value} at ({x}, {y}, {width}, {height})"
                    )

        except Exception as e:
            if self.verbose:
                print(f"⚠ OpenCV QR detection error: {e}")

        return qr_codes

    def _detect_pyzbar(self, image_rgb: np.ndarray) -> List[QRCodeInfo]:
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

                    # Decode the data
                    try:
                        data = obj.data.decode("utf-8")
                    except UnicodeDecodeError:
                        data = obj.data.decode("latin-1", errors="ignore")

                    # Classify the QR code content
                    qr_type, category = self._classify_qr_content(data)

                    qr_info = QRCodeInfo(
                        bbox=(x, y, width, height),
                        confidence=0.95,  # Pyzbar is generally reliable
                        qr_type=qr_type,
                        content=data,
                        category=category,
                    )

                    qr_codes.append(qr_info)

                    if self.verbose:
                        print(
                            f"✓ Detected QR code: {qr_type.value} at ({x}, {y}, {width}, {height})"
                        )

        except Exception as e:
            if self.verbose:
                print(f"⚠ Pyzbar QR detection error: {e}")

        return qr_codes

    def _classify_qr_content(self, content: str) -> Tuple[QRCodeType, str]:
        """Classify QR code content into types and categories.

        Args:
            content: Decoded QR code content

        Returns:
            Tuple of (QRCodeType, category_string)
        """
        if not content:
            return QRCodeType.UNKNOWN, "empty"

        content = content.strip()

        # URL detection
        if content.startswith(("http://", "https://", "www.")):
            return QRCodeType.URL, "website"

        # WiFi configuration
        if content.startswith("WIFI:"):
            return QRCodeType.WIFI, "wifi"

        # Contact information (vCard)
        if content.startswith("BEGIN:VCARD") or "VCARD" in content.upper():
            return QRCodeType.CONTACT, "contact"

        # Email
        if content.startswith("mailto:") or "@" in content:
            return QRCodeType.EMAIL, "email"

        # Phone number
        if content.startswith("tel:") or self._is_phone_number(content):
            return QRCodeType.PHONE, "phone"

        # SMS
        if content.startswith("sms:") or content.startswith("SMSTO:"):
            return QRCodeType.SMS, "sms"

        # Location/Geographic coordinates
        if content.startswith("geo:") or self._is_coordinates(content):
            return QRCodeType.LOCATION, "location"

        # Calendar event
        if content.startswith("BEGIN:VEVENT") or "VEVENT" in content.upper():
            return QRCodeType.CALENDAR, "calendar"

        # Plain text - analyze content for patterns
        if len(content) > 0:
            # Check for common patterns
            if self._is_likely_advertisement(content):
                return QRCodeType.TEXT, "advertisement"
            elif self._is_likely_documentation(content):
                return QRCodeType.TEXT, "documentation"
            else:
                return QRCodeType.TEXT, "general"

        return QRCodeType.UNKNOWN, "unknown"

    def _is_phone_number(self, text: str) -> bool:
        """Check if text looks like a phone number."""
        # Remove common separators and check for digits
        cleaned = "".join(c for c in text if c.isdigit() or c in "+()- ")
        digits = "".join(c for c in cleaned if c.isdigit())

        # Phone numbers typically have 7-15 digits
        return 7 <= len(digits) <= 15 and any(c.isdigit() for c in text)

    def _is_coordinates(self, text: str) -> bool:
        """Check if text looks like geographic coordinates."""
        # Simple coordinate pattern matching
        import re

        coord_pattern = r"-?\d+\.\d+\s*,\s*-?\d+\.\d+"
        return bool(re.search(coord_pattern, text))

    def _is_likely_advertisement(self, text: str) -> bool:
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

    def _is_likely_documentation(self, text: str) -> bool:
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
