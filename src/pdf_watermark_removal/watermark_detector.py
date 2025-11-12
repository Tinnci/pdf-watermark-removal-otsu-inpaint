"""Watermark detection using multiple methods: traditional CV or YOLO."""

import cv2
import numpy as np

from .model_manager import ModelManager


class WatermarkDetector:
    """Detects watermarks using traditional CV or YOLOv8 segmentation."""

    def __init__(
        self,
        kernel_size=3,
        verbose=False,
        auto_detect_color=True,
        watermark_color=None,
        protect_text=True,
        color_tolerance=30,
        detection_method="traditional",
        yolo_model_path="yolov8n-seg.pt",
        yolo_conf_thres=0.25,
        yolo_device="auto",
        yolo_version="v8",
        auto_download_model=True,
    ):
        """Initialize the watermark detector.

        Args:
            kernel_size: Size of morphological kernel
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
            watermark_color: Watermark color (R, G, B) or None
            protect_text: Protect dark text from being removed
            color_tolerance: Color matching tolerance (0-255, default 30)
            detection_method: 'traditional' or 'yolo'
            yolo_model_path: Path to YOLO model
            yolo_conf_thres: YOLO confidence threshold
            yolo_device: YOLO device ('cpu', 'cuda', 'auto')
            yolo_version: 'v8', 'v12', or 'v11' (default: v8)
            auto_download_model: Automatically download YOLO model if not found
        """
        self.method = detection_method
        self.verbose = verbose
        # Store kernel_size for both methods (used in refine_mask)
        self.kernel_size = kernel_size

        if detection_method == "yolo":
            try:
                from .yolo_detector import YOLOVersion, YOLOWatermarkDetector

                # Determine version enum first
                version_enum = (
                    YOLOVersion.V8
                    if yolo_version == "v8"
                    else YOLOVersion.V12
                    if yolo_version == "v12"
                    else YOLOVersion.V11
                )

                # Auto-select model based on version if using default
                if yolo_model_path == "yolov8n-seg.pt":
                    if version_enum == YOLOVersion.V12:
                        yolo_model_path = "yolov12n-seg.pt"
                    elif version_enum == YOLOVersion.V11:
                        yolo_model_path = "yolo11x-watermark.pt"

                # Auto-download model if needed
                if auto_download_model:
                    manager = ModelManager(verbose=verbose)
                    yolo_model_path = str(manager.get_model_path(yolo_model_path))

                self.detector = YOLOWatermarkDetector(
                    model_path=yolo_model_path,
                    conf_thres=yolo_conf_thres,
                    device=yolo_device,
                    verbose=verbose,
                    version=version_enum,
                )
            except ImportError as e:
                # Provide clear guidance
                error_msg = (
                    "YOLO detection requires 'ultralytics' package. "
                    "Install with: pip install ultralytics>=8.3.0 "
                    "or pip install pdf-watermark-removal-otsu-inpaint[yolo]"
                )
                if verbose:
                    print(f"[ERROR] {error_msg}")
                    print(f"[DEBUG] Original error: {e}")
                raise ImportError(error_msg) from e
            except Exception as e:
                if verbose:
                    print(f"[ERROR] YOLO initialization failed: {e}")
                raise
        else:
            self._init_traditional(
                kernel_size,
                auto_detect_color,
                watermark_color,
                protect_text,
                color_tolerance,
            )

    def _init_traditional(
        self,
        kernel_size,
        auto_detect_color,
        watermark_color,
        protect_text,
        color_tolerance,
    ):
        """Initialize traditional detection parameters."""
        self.kernel_size = kernel_size
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
        # Identify dark regions (typically text) with gray level 0-150
        # Raised from 80 to 150 to better protect gray text while avoiding watermarks at 233
        _, text_protect = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Remove small noise from text protection mask
        kernel_protect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        text_protect = cv2.morphologyEx(
            text_protect, cv2.MORPH_OPEN, kernel_protect, iterations=1
        )
        return text_protect

    def detect_watermark_mask(self, image_rgb):
        """Detect watermark regions using selected method.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Binary mask of detected watermark regions
        """
        if self.method == "yolo":
            return self.detector.detect_watermark_mask(image_rgb)
        else:
            return self._traditional_detect_mask(image_rgb)

    def _traditional_detect_mask(self, image_rgb):
        """Traditional watermark detection using color analysis and structure validation.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            Binary mask of detected watermark regions
        """
        if self.auto_detect_color and self.watermark_color is None:
            self.detect_watermark_color(image_rgb)

        if self.verbose:
            print("Converting image to grayscale...")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        mask = np.zeros_like(gray)

        # === PRECISE COLOR-BASED MODE ===
        if self.watermark_color is not None:
            if self.verbose:
                print(
                    f"Using precise color-based detection with tolerance {self.color_tolerance}..."
                )

            # 1. Create raw color mask
            target_gray = int(np.mean(self.watermark_color[:3]))
            color_diff = np.abs(gray.astype(int) - target_gray)
            color_mask = (color_diff < self.color_tolerance).astype(np.uint8) * 255

            # 2. Protect critical areas BEFORE morphological operations
            # This prevents the mask from "bleeding" into text during refinement
            if self.verbose:
                print("Protecting text and background *before* mask refinement...")

            # Protect white background
            _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            protected_mask = cv2.bitwise_and(
                color_mask, cv2.bitwise_not(background_mask)
            )

            # Protect dark text
            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray)
                protected_mask = cv2.bitwise_and(
                    protected_mask, cv2.bitwise_not(text_protect_mask)
                )

            # 3. Refine the *protected* mask to remove noise and fill gaps
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
            )
            opened_mask = cv2.morphologyEx(
                protected_mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            closed_mask = cv2.morphologyEx(
                opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            mask = closed_mask

        # === AUTOMATIC DETECTION MODE (no color specified) ===
        else:
            if self.verbose:
                print("No color specified: Using general detection logic...")

            # Use adaptive thresholding and saturation as main detectors
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
            )
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            s_channel = hsv[:, :, 1]
            saturation_mean = np.mean(s_channel)
            saturation_threshold = max(30, int(saturation_mean * 0.6))
            saturation_mask = (s_channel < saturation_threshold).astype(np.uint8) * 255

            # Combine detectors
            combined_mask = cv2.bitwise_or(opened, saturation_mask)

            # Post-protection for auto mode
            _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            mask_no_bg = cv2.bitwise_and(
                combined_mask, cv2.bitwise_not(background_mask)
            )

            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray)
                mask = cv2.bitwise_and(mask_no_bg, cv2.bitwise_not(text_protect_mask))
            else:
                mask = mask_no_bg

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
        if self.method == "yolo":
            # YOLOv8 already provides good masks, light refinement only
            return self.detector.refine_mask(mask, kernel_size=self.kernel_size)
        else:
            return self._traditional_refine_mask(mask, min_area, max_area)

    def _traditional_refine_mask(self, mask, min_area=100, max_area=5000):
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
        preview[watermark_regions] = (
            preview[watermark_regions] * 0.25 + np.array([255, 0, 0]) * 0.75
        )

        # Blue overlay for text protection regions (if enabled)
        if self.protect_text:
            text_mask = self.get_text_protect_mask(gray)
            text_regions = text_mask > 0
            preview[text_regions] = (
                preview[text_regions] * 0.5 + np.array([0, 0, 255]) * 0.5
            )

        preview = preview.astype(np.uint8)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if self.verbose:
                print(f"Debug preview saved to {output_path}")

        return preview
