"""Watermark removal using OpenCV inpaint."""

import cv2
import numpy as np
from .watermark_detector import WatermarkDetector


class WatermarkRemover:
    """Removes watermarks from images using inpainting."""

    def __init__(
        self,
        kernel_size=3,
        inpaint_radius=2,
        verbose=False,
        auto_detect_color=True,
        watermark_color=None,
    ):
        """Initialize the watermark remover.

        Args:
            kernel_size: Size of morphological kernel for detection
            inpaint_radius: Radius for inpainting algorithm
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
            watermark_color: Watermark color (R, G, B) or None
        """
        self.detector = WatermarkDetector(
            kernel_size=kernel_size,
            verbose=verbose,
            auto_detect_color=auto_detect_color,
            watermark_color=watermark_color,
        )
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

        # Calculate dynamic inpaint radius based on watermark coverage
        watermark_coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        dynamic_radius = max(2, int(self.inpaint_radius + watermark_coverage * 5))

        if self.verbose:
            coverage_pct = watermark_coverage * 100
            print(
                f"Watermark coverage: {coverage_pct:.2f}%, "
                f"dynamic radius: {dynamic_radius}"
            )

        # Apply inpainting directly on RGB format
        # Modern OpenCV supports RGB without conversion
        restored = cv2.inpaint(
            image_rgb.astype(np.uint8), mask, dynamic_radius, cv2.INPAINT_TELEA
        )

        # Post-processing: preserve edge information from original
        if watermark_coverage > 0.02:  # Only if meaningful watermark detected
            edges = cv2.Canny(image_rgb, 100, 200)
            kernel = np.ones((3, 3), np.uint8)
            edge_mask = cv2.dilate(edges, kernel, iterations=1)

            # Blend: use original image where edges are strong
            restored = np.where(edge_mask[:, :, None] > 0, image_rgb, restored).astype(
                np.uint8
            )

        return restored

    def remove_watermark_multi_pass(self, image_rgb, passes=2):
        """Remove watermark using multiple passes with progressive mask expansion.

        Uses a smarter approach: instead of reprocessing the entire image multiple
        times (which causes over-smoothing), it expands the mask progressively and
        applies inpainting once per pass with updated parameters.

        Args:
            image_rgb: Input image in RGB format
            passes: Number of removal passes

        Returns:
            Image with watermark removed
        """
        result = image_rgb.copy()

        for pass_num in range(passes):
            if self.verbose:
                print(f"Pass {pass_num + 1}/{passes}")

            # Detect mask from current result
            mask = self.detector.detect_watermark_mask(result)
            mask = self.detector.refine_mask(mask)

            if np.count_nonzero(mask) == 0:
                if self.verbose:
                    print("No watermark detected, stopping")
                break

            # Slightly expand mask for subsequent passes to catch remaining traces
            if pass_num > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Calculate coverage-based radius
            watermark_coverage = np.count_nonzero(mask) / (
                mask.shape[0] * mask.shape[1]
            )
            inpaint_radius = max(2, int(self.inpaint_radius + watermark_coverage * 5))

            if self.verbose:
                coverage_pct = watermark_coverage * 100
                print(
                    f"Watermark coverage: {coverage_pct:.2f}%, radius: {inpaint_radius}"
                )

            # Apply inpainting
            result = cv2.inpaint(
                result.astype(np.uint8), mask, inpaint_radius, cv2.INPAINT_TELEA
            )

        return result
