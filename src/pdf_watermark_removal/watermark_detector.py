"""Watermark detection using Otsu threshold segmentation."""

import cv2
import numpy as np


class WatermarkDetector:
    """Detects watermarks using Otsu threshold segmentation."""

    def __init__(self, kernel_size=3, verbose=False):
        """Initialize the watermark detector.

        Args:
            kernel_size: Size of morphological kernel
            verbose: Enable verbose logging
        """
        self.kernel_size = kernel_size
        self.verbose = verbose

    def detect_watermark_mask(self, image_rgb):
        """Detect watermark regions using Otsu thresholding.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Binary mask of detected watermark regions
        """
        if self.verbose:
            print("Converting image to grayscale...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        if self.verbose:
            print("Applying Otsu thresholding...")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        if self.verbose:
            print("Applying morphological operations...")

        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        if self.verbose:
            detected_pixels = np.count_nonzero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            percentage = (detected_pixels / total_pixels) * 100
            print(f"Detected watermark coverage: {percentage:.2f}%")

        return mask

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
