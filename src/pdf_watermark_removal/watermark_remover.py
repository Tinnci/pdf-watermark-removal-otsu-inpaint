"""Watermark removal using OpenCV inpaint."""

import cv2
import numpy as np
from .watermark_detector import WatermarkDetector


class WatermarkRemover:
    """Removes watermarks from images using inpainting."""

    def __init__(self, kernel_size=3, inpaint_radius=2, verbose=False, auto_detect_color=True):
        """Initialize the watermark remover.

        Args:
            kernel_size: Size of morphological kernel for detection
            inpaint_radius: Radius for inpainting algorithm
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
        """
        self.detector = WatermarkDetector(kernel_size=kernel_size, verbose=verbose, auto_detect_color=auto_detect_color)
        self.inpaint_radius = inpaint_radius
        self.verbose = verbose

    def remove_watermark(self, image_rgb):
        """Remove watermark from an image.

        Args:
            image_rgb: Input image in RGB format (0-255)

        Returns:
            Image with watermark removed (RGB format)
        """
        if self.verbose:
            print("Detecting watermark regions...")

        mask = self.detector.detect_watermark_mask(image_rgb)
        mask = self.detector.refine_mask(mask)

        if self.verbose:
            print(f"Applying inpainting with radius {self.inpaint_radius}...")

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        restored = cv2.inpaint(bgr, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        result = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

        return result

    def remove_watermark_multi_pass(self, image_rgb, passes=2):
        """Remove watermark using multiple passes for better results.

        Args:
            image_rgb: Input image in RGB format
            passes: Number of removal passes

        Returns:
            Image with watermark removed
        """
        result = image_rgb.copy()

        for i in range(passes):
            if self.verbose:
                print(f"Pass {i + 1}/{passes}")

            result = self.remove_watermark(result)

        return result
