"""Watermark detection using Otsu threshold segmentation and color analysis."""

import cv2
import numpy as np


class WatermarkDetector:
    """Detects watermarks using Otsu threshold segmentation and color analysis."""

    def __init__(
        self,
        kernel_size=3,
        verbose=False,
        auto_detect_color=True,
        watermark_color=None,
        protect_text=True,
        color_tolerance=30,
    ):
        """Initialize the watermark detector.

        Args:
            kernel_size: Size of morphological kernel
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
            watermark_color: Watermark color (R, G, B) or None
            protect_text: Protect dark text from being removed
            color_tolerance: Color matching tolerance (0-255, default 30)
        """
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.auto_detect_color = auto_detect_color
        self.watermark_color = watermark_color
        self.protect_text = protect_text
        self.color_tolerance = color_tolerance

    def detect_watermark_color(self, image_rgb):
        """Detect the dominant watermark color using color analysis.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Tuple of (B, G, R) representing watermark color
        """
        if self.verbose:
            print("Analyzing image to detect watermark color...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        unique_grays, counts = np.unique(gray, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]

        total_pixels = gray.shape[0] * gray.shape[1]

        for idx in sorted_idx[:10]:
            gray_val = unique_grays[idx]
            count = counts[idx]
            coverage = (count / total_pixels) * 100

            # Watermark characteristics:
            # - Gray level: 100-250 (expanded from 150-250 for more sensitivity)
            # - Coverage: 1-20% (expanded from 2-15 to catch more watermarks)
            # - Excludes text (0-50, <5%) and background (>80%)
            if 100 <= gray_val <= 250 and 1 <= coverage <= 20:
                bgr_color = (gray_val, gray_val, gray_val)
                if self.verbose:
                    print(
                        f"Detected watermark color (BGR): {bgr_color}, "
                        f"coverage: {coverage:.1f}%"
                    )
                self.watermark_color = bgr_color
                return bgr_color

        return None

    def get_text_protect_mask(self, gray):
        """Create a mask to protect dark text regions from being removed.

        Args:
            gray: Grayscale image

        Returns:
            Binary mask protecting text areas (255 where text should be protected)
        """
        # Identify dark regions (typically text) with gray level 0-80
        _, text_protect = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Remove small noise from text protection mask
        kernel_protect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        text_protect = cv2.morphologyEx(text_protect, cv2.MORPH_OPEN, 
                                       kernel_protect, iterations=1)
        return text_protect

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
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]

        if self.verbose:
            print("Applying adaptive thresholding for better watermark detection...")

        # Use adaptive thresholding instead of simple Otsu for better results
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        if self.verbose:
            print("Applying morphological operations...")

        # Apply morphological operations to clean up the mask
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Combine with color-based detection for better watermark isolation
        if self.watermark_color is not None:
            if self.verbose:
                print("Combining with color-based watermark detection...")

            # Extract target gray value from watermark color (handle both RGB and BGR)
            if (
                isinstance(self.watermark_color, (tuple, list))
                and len(self.watermark_color) >= 3
            ):
                target_gray = int(
                    np.mean(self.watermark_color[:3])
                )  # Use first 3 components
            else:
                target_gray = self.watermark_color[0] if self.watermark_color else 200

            # Create color-based mask: pixels close to watermark color
            color_diff = np.abs(gray.astype(int) - target_gray)
            color_mask = color_diff < self.color_tolerance  # Dynamic tolerance

            # Combine both masks
            mask = cv2.bitwise_or(mask, color_mask.astype(np.uint8) * 255)

        # Always apply adaptive saturation threshold for additional refinement
        saturation_mean = np.mean(s_channel)
        saturation_threshold = max(30, int(saturation_mean * 0.6))
        color_mask = s_channel < saturation_threshold

        # Combine thresholding and color detection
        mask = cv2.bitwise_or(mask, color_mask.astype(np.uint8) * 255)

        # Protect white background: exclude very bright areas (>250 gray level)
        # These are typically document backgrounds, not watermarks
        _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        # background_mask is 255 where gray > 250, 0 elsewhere
        # Set mask to 0 where background_mask is 255 (white areas)
        mask[background_mask == 255] = 0

        # Protect dark text regions if enabled
        if self.protect_text:
            if self.verbose:
                print("Protecting dark text regions from removal...")
            text_protect_mask = self.get_text_protect_mask(gray)
            # Only keep watermark pixels that are NOT in text regions
            # text_protect_mask is 255 where text exists, 0 elsewhere
            # We need to exclude (invert) the text regions from watermark mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_protect_mask))

        if self.verbose:
            detected_pixels = np.count_nonzero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            percentage = (detected_pixels / total_pixels) * 100
            print(f"Detected watermark coverage: {percentage:.2f}%")

        return mask

    def refine_mask(self, mask, min_area=100, max_area=5000):
        """Refine the detected mask by removing small noise and text-like components.

        Args:
            mask: Binary mask to refine
            min_area: Minimum area for connected components
            max_area: Maximum area to avoid keeping large text blocks

        Returns:
            Refined mask
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        refined = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate aspect ratio to filter out text lines
            aspect_ratio = width / height if height > 0 else 0
            
            # Keep components that are: within area range AND not thin/elongated (text-like)
            # Aspect ratio < 10 filters out thin text lines which tend to be very elongated
            if min_area <= area <= max_area and aspect_ratio < 10:
                refined[labels == i] = 255

        return refined

    def preview_detection(self, image_rgb, output_path=None):
        """Generate debug preview showing watermark and text regions.

        Args:
            image_rgb: Input image in RGB format
            output_path: Optional path to save preview image

        Returns:
            Preview image with color-coded regions
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        mask = self.detect_watermark_mask(image_rgb)
        
        # Create colored preview
        preview = image_rgb.copy().astype(np.float32)
        
        # Red overlay for watermark regions (75% opacity)
        watermark_regions = mask > 0
        preview[watermark_regions] = preview[watermark_regions] * 0.25 + np.array([255, 0, 0]) * 0.75
        
        # Blue overlay for text protection regions (if enabled)
        if self.protect_text:
            text_mask = self.get_text_protect_mask(gray)
            text_regions = text_mask > 0
            preview[text_regions] = preview[text_regions] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        preview = preview.astype(np.uint8)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if self.verbose:
                print(f"Debug preview saved to {output_path}")
        
        return preview

