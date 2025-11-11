"""Color detection and visualization for watermark identification."""

import cv2
import numpy as np
from collections import Counter


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
            Dict with recommended color and alternatives
        """
        if self.verbose:
            print("Analyzing watermark color distribution...")

        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Get low-saturation pixels (text, watermarks, not photos)
        low_sat_mask = s_channel < 80
        
        if np.count_nonzero(low_sat_mask) == 0:
            if self.verbose:
                print("No low-saturation pixels found")
            return None

        # Analyze grayscale distribution of low-saturation pixels
        low_sat_grays = gray[low_sat_mask]
        unique_grays, counts = np.unique(low_sat_grays, return_counts=True)

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]

        # Calculate statistics
        total_low_sat_pixels = len(low_sat_grays)
        total_pixels = gray.shape[0] * gray.shape[1]

        colors_info = []
        for i, idx in enumerate(sorted_indices[:5]):  # Top 5 colors
            gray_val = unique_grays[idx]
            count = counts[idx]
            percentage = (count / total_low_sat_pixels) * 100
            coverage = (count / total_pixels) * 100

            # Create RGB from grayscale
            rgb_color = (gray_val, gray_val, gray_val)

            colors_info.append({
                'index': i,
                'rgb': rgb_color,
                'bgr': tuple(reversed(rgb_color)),
                'gray': gray_val,
                'count': count,
                'percentage': percentage,
                'coverage': coverage,
            })

        # Calculate confidence based on dominant color
        if colors_info:
            dominant_percentage = colors_info[0]['percentage']
            # High confidence if dominant color is >30% or very dominant
            if dominant_percentage > 40:
                confidence = 95
            elif dominant_percentage > 30:
                confidence = 85
            elif dominant_percentage > 20:
                confidence = 75
            else:
                confidence = 65

            colors_info[0]['confidence'] = confidence
            colors_info[0]['is_recommended'] = True

        return colors_info

    def get_dominant_colors(self, image_rgb, num_colors=5):
        """Get dominant non-document colors (potential watermarks).

        Args:
            image_rgb: Input image in RGB format
            num_colors: Number of colors to return

        Returns:
            List of color dictionaries
        """
        return self.analyze_watermark_color(image_rgb)[:num_colors] if self.analyze_watermark_color(image_rgb) else []

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

