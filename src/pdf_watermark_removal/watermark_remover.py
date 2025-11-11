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
        inpaint_strength=1.0,
        verbose=False,
        auto_detect_color=True,
        watermark_color=None,
        protect_text=True,
    ):
        """Initialize the watermark remover.

        Args:
            kernel_size: Size of morphological kernel for detection
            inpaint_radius: Radius for inpainting algorithm
            inpaint_strength: Strength of inpainting (0.5=light, 1.0=medium, 1.5=strong)
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
            watermark_color: Watermark color (R, G, B) or None
            protect_text: Protect dark text from being removed
        """
        self.detector = WatermarkDetector(
            kernel_size=kernel_size,
            verbose=verbose,
            auto_detect_color=auto_detect_color,
            watermark_color=watermark_color,
            protect_text=protect_text,
        )
        self.inpaint_radius = inpaint_radius
        self.inpaint_strength = inpaint_strength
        self.verbose = verbose

    def apply_inpaint_strength(self, original, inpainted, mask, strength):
        """Apply inpaint strength by blending original and inpainted images.

        Args:
            original: Original image
            inpainted: Inpainted result
            mask: Binary mask of watermark regions
            strength: Blending strength (0=original, 1.0=full inpaint)

        Returns:
            Blended result
        """
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Blend based on strength: result = original * (1 - strength * mask) + inpainted * strength * mask
        blend_factor = mask_normalized[:, :, np.newaxis] * strength
        result = original.astype(np.float32) * (1 - blend_factor) + \
                 inpainted.astype(np.float32) * blend_factor
        
        return result.astype(np.uint8)

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

        # Skip processing if no watermark detected
        if np.count_nonzero(mask) == 0:
            if self.verbose:
                print("No watermark detected, returning original image")
            return image_rgb.astype(np.uint8)

        if self.verbose:
            print(f"Applying inpainting with radius {self.inpaint_radius}...")

        # Calculate dynamic inpaint radius based on watermark coverage and strength
        watermark_coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        dynamic_radius = max(
            2, int(self.inpaint_radius + watermark_coverage * 10 * self.inpaint_strength)
        )

        if self.verbose:
            coverage_pct = watermark_coverage * 100
            print(
                f"Watermark coverage: {coverage_pct:.2f}%, "
                f"inpaint strength: {self.inpaint_strength}, "
                f"dynamic radius: {dynamic_radius}"
            )

        # Convert RGB to BGR for OpenCV inpainting (best practice for color accuracy)
        image_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Apply inpainting using TELEA algorithm
        restored_bgr = cv2.inpaint(image_bgr, mask, dynamic_radius, cv2.INPAINT_TELEA)

        # Convert back to RGB
        restored = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply strength blending if not at maximum
        if self.inpaint_strength < 1.5:
            restored = self.apply_inpaint_strength(image_rgb, restored, mask, self.inpaint_strength)

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

            # Calculate coverage-based radius with strength adjustment
            watermark_coverage = np.count_nonzero(mask) / (
                mask.shape[0] * mask.shape[1]
            )
            inpaint_radius = max(
                2, int(self.inpaint_radius + watermark_coverage * 10 * self.inpaint_strength)
            )

            if self.verbose:
                coverage_pct = watermark_coverage * 100
                print(
                    f"Watermark coverage: {coverage_pct:.2f}%, "
                    f"strength: {self.inpaint_strength}, radius: {inpaint_radius}"
                )

            # Apply inpainting
            result_inpainted = cv2.inpaint(
                result.astype(np.uint8), mask, inpaint_radius, cv2.INPAINT_TELEA
            )
            
            # Apply strength blending
            if self.inpaint_strength < 1.5:
                result = self.apply_inpaint_strength(result, result_inpainted, mask, self.inpaint_strength)
            else:
                result = result_inpainted

        return result
