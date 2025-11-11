"""Watermark detection using Otsu threshold segmentation and color analysis."""

import cv2
import numpy as np
from collections import Counter


class WatermarkDetector:
    """Detects watermarks using Otsu threshold segmentation and color analysis."""

    def __init__(self, kernel_size=3, verbose=False, auto_detect_color=True):
        """Initialize the watermark detector.

        Args:
            kernel_size: Size of morphological kernel
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
        """
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.auto_detect_color = auto_detect_color
        self.watermark_color = None

    def detect_watermark_color(self, image_rgb):
        """Detect the dominant watermark color using color analysis.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Tuple of (B, G, R) representing watermark color
        """
        if self.verbose:
            print("Analyzing image to detect watermark color...")

        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Focus on saturation and value channels
        # Watermarks typically have lower saturation and value
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Get pixels with low saturation (likely watermark or text)
        low_sat_mask = s_channel < 50

        if np.count_nonzero(low_sat_mask) == 0:
            if self.verbose:
                print("No low-saturation pixels found, using grayscale analysis...")
            return None

        # Get the hue of low-saturation pixels
        low_sat_pixels = hsv[low_sat_mask]

        if len(low_sat_pixels) > 0:
            # Most common hue value (likely watermark color)
            hue_values = low_sat_pixels[:, 0]
            most_common_hue = Counter(hue_values).most_common(1)[0][0]

            if self.verbose:
                print(f"Detected watermark hue: {most_common_hue}")

            # Convert back to RGB for reference
            hsv_color = np.uint8([[[most_common_hue, 100, 200]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            bgr_color = tuple(reversed(rgb_color))

            if self.verbose:
                print(f"Estimated watermark color (BGR): {bgr_color}")

            self.watermark_color = bgr_color
            return bgr_color

        return None

    def detect_watermark_mask(self, image_rgb):
        """Detect watermark regions using Otsu thresholding and color analysis.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Binary mask of detected watermark regions
        """
        # Auto-detect watermark color if enabled
        if self.auto_detect_color and self.watermark_color is None:
            self.detect_watermark_color(image_rgb)

        if self.verbose:
            print("Converting image to grayscale...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        if self.verbose:
            print("Applying Otsu thresholding...")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert to get darker regions (watermarks typically darker)
        binary_inv = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        if self.verbose:
            print("Applying morphological operations...")

        # Apply morphological operations
        opened = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Additional color-based detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]

        # Low saturation regions (text, watermarks)
        color_mask = s_channel < 50

        # Combine thresholding and color detection
        combined_mask = cv2.bitwise_or(mask, color_mask.astype(np.uint8) * 255)

        if self.verbose:
            detected_pixels = np.count_nonzero(combined_mask)
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            percentage = (detected_pixels / total_pixels) * 100
            print(f"Detected watermark coverage: {percentage:.2f}%")

        return combined_mask

    def refine_mask(self, mask, min_area=100):
        """Refine the detected mask by removing small noise.

        Args:
            mask: Binary mask to refine
            min_area: Minimum area for connected components

        Returns:
            Refined mask
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        refined = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined[labels == i] = 255

        return refined
