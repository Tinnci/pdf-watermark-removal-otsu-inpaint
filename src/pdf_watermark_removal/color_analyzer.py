"""Color detection and visualization for watermark identification."""

import cv2
import numpy as np
from enum import Enum


class ColorType(Enum):
    """Color classification types."""

    BACKGROUND = "background"
    WATERMARK = "watermark"
    TEXT = "text"
    NOISE = "noise"


class ColorAnalyzer:
    """Analyzes and detects watermark colors in images."""

    def __init__(self, verbose=False):
        """Initialize color analyzer.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    @staticmethod
    def _classify_color(gray_val, coverage):
        """Intelligently classify color type and calculate confidence.

        Args:
            gray_val: Grayscale value (0-255)
            coverage: Coverage percentage (0-100)

        Returns:
            Tuple (color_type, confidence) where confidence is 0-100
        """
        gray_val = int(gray_val)

        # Background detection: very light (240-255) and high coverage (>60%)
        if 240 <= gray_val <= 255 and coverage > 60:
            return ColorType.BACKGROUND, 0

        # Watermark detection: mid-high grayscale (180-240) and moderate coverage (2-15%)
        if 180 <= gray_val <= 240 and 2 <= coverage <= 15:
            # Confidence peaks around gray_val=210 and coverage=8%
            # But the model should be more lenient
            gray_factor = 1 - abs(gray_val - 210) / 40  # peaks at 210, range 170-250
            coverage_factor = 1 - abs(coverage - 8) / 8  # peaks at 8%, range 0-16%
            base_confidence = (gray_factor * 0.5 + coverage_factor * 0.5) * 100
            # Add bonus for typical watermark coverage range
            if 3 <= coverage <= 10:
                base_confidence = min(100, base_confidence + 30)
            return ColorType.WATERMARK, max(20, min(100, base_confidence))

        # Text detection: dark (0-80) and low coverage (<5%)
        if 0 <= gray_val <= 80 and coverage < 5:
            return ColorType.TEXT, 0

        # Noise for everything else
        return ColorType.NOISE, 0

    def analyze_watermark_color(self, image_rgb):
        """Intelligently analyze and recommend watermark color.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            List of color dicts sorted by confidence, watermark first
        """
        if self.verbose:
            print("Analyzing watermark color distribution...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        unique_grays, counts = np.unique(gray, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]

        total_pixels = gray.shape[0] * gray.shape[1]

        colors_info = []
        for i, idx in enumerate(sorted_idx[:20]):  # Check top 20 colors
            gray_val = unique_grays[idx]
            count = counts[idx]
            coverage = (count / total_pixels) * 100

            rgb_color = (int(gray_val), int(gray_val), int(gray_val))

            # Classify color with confidence
            color_type, confidence = self._classify_color(gray_val, coverage)

            colors_info.append(
                {
                    "index": i,
                    "rgb": rgb_color,
                    "bgr": tuple(reversed(rgb_color)),
                    "gray": int(gray_val),
                    "count": count,
                    "coverage": coverage,
                    "color_type": color_type,
                    "confidence": confidence,
                }
            )

        # Sort by confidence (descending), then by color type priority
        type_priority = {
            ColorType.WATERMARK: 3,
            ColorType.TEXT: 2,
            ColorType.NOISE: 1,
            ColorType.BACKGROUND: 0,
        }

        colors_info.sort(
            key=lambda x: (x["confidence"], type_priority.get(x["color_type"], 0)),
            reverse=True,
        )

        # Mark the best watermark as recommended
        for color in colors_info:
            if color["color_type"] == ColorType.WATERMARK:
                color["is_recommended"] = True
                break

        return colors_info

    def get_dominant_colors(self, image_rgb, num_colors=5):
        """Get dominant non-document colors (potential watermarks).

        Args:
            image_rgb: Input image in RGB format
            num_colors: Number of colors to return

        Returns:
            List of color dictionaries
        """
        return (
            self.analyze_watermark_color(image_rgb)[:num_colors]
            if self.analyze_watermark_color(image_rgb)
            else []
        )

    def create_color_mask(self, image_rgb, color_rgb, tolerance=20):
        """Create a mask for pixels matching the given color.

        Args:
            image_rgb: Input image
            color_rgb: Target color (R, G, B)
            tolerance: Color tolerance threshold

        Returns:
            Binary mask of matching pixels
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        target_gray = int(np.mean(color_rgb))

        # Pixels within tolerance of target gray
        mask = np.abs(gray.astype(int) - target_gray) < tolerance

        return (mask * 255).astype(np.uint8)
