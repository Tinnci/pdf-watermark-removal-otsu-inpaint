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

    def get_dominant_colors(self, image_rgb, num_colors=5):
        """Get dominant non-document colors (potential watermarks).

        Args:
            image_rgb: Input image in RGB format
            num_colors: Number of colors to return

        Returns:
            List of tuples [(RGB, percentage), ...]
        """
        if self.verbose:
            print("Analyzing image colors...")

        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Get low-saturation regions (text, watermarks, not photos)
        # Low saturation = grayscale/neutral colors
        low_sat_mask = s_channel < 80

        if np.count_nonzero(low_sat_mask) == 0:
            if self.verbose:
                print("No low-saturation pixels found")
            return []

        # Get pixels in low saturation regions
        low_sat_pixels = image_rgb[low_sat_mask]

        # Convert to grayscale to find brightness levels
        low_sat_gray = cv2.cvtColor(low_sat_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()

        # Find distinct brightness levels (likely different text/watermark colors)
        unique_grays, counts = np.unique(low_sat_gray, return_counts=True)

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]

        colors_with_counts = []
        for idx in sorted_indices[:num_colors]:
            gray_val = unique_grays[idx]
            count = counts[idx]
            percentage = (count / len(low_sat_gray)) * 100

            # Convert grayscale back to RGB
            rgb_color = (gray_val, gray_val, gray_val)

            colors_with_counts.append({
                'rgb': rgb_color,
                'bgr': tuple(reversed(rgb_color)),
                'gray': gray_val,
                'count': count,
                'percentage': percentage,
                'hue_range': self._get_hue_range(image_rgb, low_sat_mask, gray_val)
            })

        return colors_with_counts

    def _get_hue_range(self, image_rgb, mask, target_gray):
        """Get hue range for a specific grayscale value.

        Args:
            image_rgb: Input image
            mask: Mask of low-saturation pixels
            target_gray: Target grayscale value

        Returns:
            Tuple of (min_hue, max_hue) or None
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Find pixels close to target gray value
        gray_match = np.abs(gray.astype(int) - target_gray) < 10
        combined_mask = mask & gray_match

        if np.count_nonzero(combined_mask) == 0:
            return None

        hues = hsv[combined_mask, 0]
        return (int(np.min(hues)), int(np.max(hues)))

    def visualize_colors(self, colors_data, width=400, height=100):
        """Create a visualization of detected colors.

        Args:
            colors_data: List of color dictionaries
            width: Width of visualization
            height: Height of visualization

        Returns:
            BGR image showing the colors
        """
        if not colors_data:
            return None

        # Create image with color boxes
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        color_width = width // len(colors_data)

        for i, color_data in enumerate(colors_data):
            bgr = color_data['bgr']
            start_x = i * color_width
            end_x = (i + 1) * color_width if i < len(colors_data) - 1 else width

            cv2.rectangle(img, (start_x, 0), (end_x, height), bgr, -1)

            # Add text with percentage
            percentage = color_data['percentage']
            text = f"{percentage:.1f}%"
            cv2.putText(img, text, (start_x + 5, height // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return img

    def create_color_mask(self, image_rgb, color_rgb, tolerance=20):
        """Create a mask for pixels matching the given color.

        Args:
            image_rgb: Input image
            color_rgb: Target color (R, G, B)
            tolerance: Color tolerance threshold

        Returns:
            Binary mask of matching pixels
        """
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Convert target color to HSV
        color_hsv = cv2.cvtColor(np.uint8([[[color_rgb[0], color_rgb[1], color_rgb[2]]]]),
                                cv2.COLOR_RGB2HSV)[0][0]

        # Create mask based on grayscale value (since watermarks are typically neutral)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        target_gray = int(np.mean(color_rgb))

        # Pixels within tolerance of target gray
        mask = np.abs(gray.astype(int) - target_gray) < tolerance

        return (mask * 255).astype(np.uint8)
