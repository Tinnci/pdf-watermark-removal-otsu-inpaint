"""Color detection and visualization for watermark identification."""

import cv2
import numpy as np


class ColorAnalyzer:
    """Analyzes and detects watermark colors in images."""

    def __init__(self, verbose=False):
        """Initialize color analyzer.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    def analyze_watermark_color(self, image_rgb):
        """Intelligently analyze and recommend watermark color.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            List of color dicts with recommended color first
        """
        if self.verbose:
            print("Analyzing watermark color distribution...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        unique_grays, counts = np.unique(gray, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]

        total_pixels = gray.shape[0] * gray.shape[1]

        colors_info = []
        for i, idx in enumerate(sorted_idx[:5]):
            gray_val = unique_grays[idx]
            count = counts[idx]
            coverage = (count / total_pixels) * 100

            rgb_color = (gray_val, gray_val, gray_val)

            colors_info.append(
                {
                    "index": i,
                    "rgb": rgb_color,
                    "bgr": tuple(reversed(rgb_color)),
                    "gray": gray_val,
                    "count": count,
                    "coverage": coverage,
                }
            )

        if colors_info:
            coverage = colors_info[0]["coverage"]
            if coverage > 40:
                confidence = 95
            elif coverage > 30:
                confidence = 85
            elif coverage > 20:
                confidence = 75
            else:
                confidence = 65

            colors_info[0]["confidence"] = confidence
            colors_info[0]["is_recommended"] = True

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
