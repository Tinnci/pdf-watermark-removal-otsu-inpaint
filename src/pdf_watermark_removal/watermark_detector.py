"""Watermark detection using multiple methods: traditional CV or YOLO."""

import cv2
import numpy as np

from .model_manager import ModelManager
from .qr_detector import QRCodeDetector


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
        detect_qr_codes=False,
        qr_detection_method="opencv",
        remove_all_qr_codes=False,
        qr_code_categories_to_remove=None,
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
            detect_qr_codes: Enable QR code detection
            qr_detection_method: QR detection method ('opencv' or 'pyzbar')
            remove_all_qr_codes: Remove all detected QR codes
            qr_code_categories_to_remove: List of QR code categories to remove
        """
        self.method = detection_method
        self.verbose = verbose
        # Store kernel_size for both methods (used in refine_mask)
        self.kernel_size = kernel_size

        # QR code detection settings
        self.detect_qr_codes = detect_qr_codes
        self.qr_detection_method = qr_detection_method
        self.remove_all_qr_codes = remove_all_qr_codes
        self.qr_code_categories_to_remove = qr_code_categories_to_remove or []
        self.qr_detector = None
        self.detected_qr_codes = []  # Now accumulates across all pages
        self.current_page_qr_codes = []  # Temporary storage for current page

        # Initialize QR detector if enabled
        if self.detect_qr_codes:
            self.qr_detector = QRCodeDetector(
                method=qr_detection_method, verbose=verbose
            )

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

    def clear_qr_codes(self):
        """Clear accumulated QR codes when starting a new document."""
        self.detected_qr_codes.clear()

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

    def get_text_protect_mask(self, gray, watermark_mask=None, expand_pixels=3):
        """Create a mask to protect dark text regions from being removed.

        Args:
            gray: Grayscale image
            watermark_mask: Current watermark mask to avoid protecting actual watermarks
            expand_pixels: How many pixels to expand text protection for anti-aliased edges

        Returns:
            Binary mask protecting text areas (255 where text should be protected)
        """
        # Step 1: Identify core text regions (very dark pixels)
        # Use 140 as threshold to catch more anti-aliased text edges
        # This is lower than the original 150 to catch gray text edges
        _, core_text = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        # Step 2: Expand the text region to catch anti-aliased edges
        # Anti-aliased text edges often have gray values that overlap with watermark range
        if expand_pixels > 0:
            kernel_expand = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
            )
            expanded_text = cv2.dilate(core_text, kernel_expand, iterations=1)
        else:
            expanded_text = core_text

        # Step 3: Remove small noise from text protection mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_text = cv2.morphologyEx(
            expanded_text, cv2.MORPH_OPEN, kernel_clean, iterations=1
        )

        # Step 4: Return the cleaned text mask
        # Note: We don't refine by watermark mask here because that would remove
        # protection for anti-aliased text edges that happen to fall in the watermark
        # color range. The protection should be based purely on image characteristics.
        return cleaned_text

    def detect_qr_codes(self, image_rgb):
        """Detect QR codes in the image.

        Args:
            image_rgb: Input image in RGB format

        Returns:
            List of detected QR code information
        """
        if not self.detect_qr_codes or self.qr_detector is None:
            return []

        return self.qr_detector.detect_qr_codes(image_rgb)

    def _filter_qr_codes_for_removal(self, qr_codes):
        """
        Filters a list of detected QR codes based on removal criteria.

        Args:
            qr_codes (list): A list of QRCodeInfo objects.

        Returns:
            list: A list of QRCodeInfo objects that should be removed.
        """
        if self.remove_all_qr_codes:
            return qr_codes
        elif self.qr_code_categories_to_remove:
            return [
                qr
                for qr in qr_codes
                if qr.category in self.qr_code_categories_to_remove
            ]
        else:
            # Default: Remove advertisements, unknown codes, and website QR codes
            # Website QR codes are commonly used as watermarks
            return [
                qr for qr in qr_codes if qr.category in ["advertisement", "unknown", "website"]
            ]

    def detect_qr_codes_mask(self, image_rgb, page_num=1):
        """Create mask for QR codes that should be removed.

        Args:
            image_rgb: Input image in RGB format
            page_num: Page number (1-indexed) for tracking

        Returns:
            Binary mask of QR codes to remove, or None if no QR codes
        """
        if not self.detect_qr_codes or self.qr_detector is None:
            return None

        # Detect QR codes for current page
        qr_codes = self.qr_detector.detect_qr_codes(image_rgb, page_num)
        self.current_page_qr_codes = qr_codes  # Store current page only

        # Accumulate across pages
        self.detected_qr_codes.extend(qr_codes)

        if not qr_codes:
            return None

        # Determine which QR codes to remove using the new helper
        codes_to_remove = self._filter_qr_codes_for_removal(qr_codes)

        if not codes_to_remove:
            if self.verbose and qr_codes:
                print(
                    f"Found {len(qr_codes)} QR code(s) but none matched removal criteria"
                )
            return None

        # Create mask for QR codes to remove
        qr_mask = self.qr_detector.create_qr_mask(
            image_rgb.shape, codes_to_remove, padding=5
        )

        if self.verbose and codes_to_remove:
            print(f"Including {len(codes_to_remove)} QR codes in removal mask")

        return qr_mask

    def get_detected_qr_codes(self):
        """Get the list of detected QR codes from the last detection.

        Returns:
            List of QRCodeInfo objects
        """
        return self.detected_qr_codes

    def _categorize_qr_codes(self, qr_codes_list):
        """
        Categorizes a list of QR codes by their category.

        Args:
            qr_codes_list (list): A list of QRCodeInfo objects.

        Returns:
            dict: A dictionary where keys are QR code categories and values are their counts.
        """
        categories = {}
        for qr in qr_codes_list:
            categories[qr.category] = categories.get(qr.category, 0) + 1
        return categories

    def get_qr_removal_summary(self):
        """Get summary of QR codes detected and marked for removal.

        Returns:
            Dictionary with summary information
        """
        if not self.detected_qr_codes:
            return None

        total = len(self.detected_qr_codes)

        # Determine which codes were/will be removed using the helper
        to_remove = self._filter_qr_codes_for_removal(self.detected_qr_codes)

        # Categorize the codes to be removed
        categories = self._categorize_qr_codes(to_remove)

        return {
            "total_detected": total,
            "to_remove": len(to_remove),
            "categories": categories,
            "codes": to_remove,
        }

    def detect_watermark_mask(self, image_rgb, page_num=1, progress=None, task_id=None):
        """Unified artifact detection pipeline: Detection → Protection → Refinement → Combination.

        Args:
            image_rgb: Input image in RGB format
            page_num: Page number (1-indexed) for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates

        Returns:
            Binary mask of detected watermark regions
        """
        # STEP 1: Parallel Detection (0-30% progress)
        if self.method == "yolo":
            if progress and task_id:
                progress.update(task_id, description=f"[yellow]Page {page_num}: YOLO inference...")
            # YOLO returns already-processed masks, so handle differently
            watermark_mask = self.detector.detect_watermark_mask(image_rgb)
            # For YOLO, we still need to apply unified processing for QR codes and final combination
            combined_mask = self._unified_yolo_processing(
                image_rgb, watermark_mask, page_num, progress, task_id
            )
        else:
            if progress and task_id:
                progress.update(task_id, description=f"[yellow]Page {page_num}: Color analysis...")
            # Get raw watermark mask without protection/refinement
            watermark_mask = self._traditional_detect_mask(image_rgb, page_num, progress, task_id, raw=True)

            # Detect QR codes in parallel (if enabled)
            qr_mask = None
            if self.detect_qr_codes:
                if progress and task_id:
                    progress.update(task_id, advance=10, description=f"[yellow]Page {page_num}: Detecting QR codes...")
                qr_mask = self.detect_qr_codes_mask(image_rgb, page_num)

            # STEP 2-4: Unified Protection, Refinement, and Combination
            combined_mask = self._unified_protection_and_refinement(
                image_rgb, watermark_mask, qr_mask, page_num, progress, task_id
            )

        # Always complete progress tracking
        if progress and task_id:
            progress.update(task_id, description=f"[green]✓ Page {page_num}: Detection complete")
            progress.update(task_id, completed=100)

        return combined_mask

    def _precise_color_based_detection(self, image_rgb, gray_image, raw=False):
        """
        Performs watermark detection using a precise color-based approach.

        Args:
            image_rgb: Input image in RGB format.
            gray_image: Grayscale version of the input image.
            raw: If True, return raw mask without protection/refinement

        Returns:
            Binary mask of detected watermark regions.
        """
        if self.verbose:
            print(
                f"Using precise color-based detection with tolerance {self.color_tolerance}..."
            )

        # 1. Create raw color mask
        target_gray = int(np.mean(self.watermark_color[:3]))
        color_diff = np.abs(gray_image.astype(int) - target_gray)
        raw_mask = (color_diff < self.color_tolerance).astype(np.uint8) * 255

        # If raw mode, return unprocessed mask
        if raw:
            return raw_mask

        # 2. PROTECT FIRST - Apply protection before any morphological operations
        if self.verbose:
            print("Protecting text and background *before* mask refinement...")

        # Start with the raw mask
        protected_mask = raw_mask.copy()

        # Protect white background
        _, background_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
        protected_mask = cv2.bitwise_and(
            protected_mask, cv2.bitwise_not(background_mask)
        )

        # Protect dark text ONLY if it doesn't match watermark color
        # If watermark color is similar to text (100-150 range), skip text protection
        # to avoid removing the actual watermark we want to detect
        if self.protect_text and not (100 <= target_gray <= 150):
            text_protect_mask = self.get_text_protect_mask(gray_image, raw_mask)
            protected_mask = cv2.bitwise_and(
                protected_mask, cv2.bitwise_not(text_protect_mask)
            )

        # 3. REFINE SECOND - Apply morphological operations to the protected mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        opened_mask = cv2.morphologyEx(
            protected_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        closed_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        return closed_mask

    def _automatic_detection_mode(self, image_rgb, gray_image, raw=False):
        """
        Performs watermark detection using a general automatic detection logic.

        Args:
            image_rgb: Input image in RGB format.
            gray_image: Grayscale version of the input image.
            raw: If True, return raw mask without protection/refinement

        Returns:
            Binary mask of detected watermark regions.
        """
        if self.verbose:
            print("No color specified: Using general detection logic...")

        # 1. Create raw detectors (without morphological operations)
        binary = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        saturation_mean = np.mean(s_channel)
        saturation_threshold = max(30, int(saturation_mean * 0.6))
        saturation_mask = (s_channel < saturation_threshold).astype(np.uint8) * 255

        # 2. Combine raw detectors
        combined_mask = cv2.bitwise_or(binary, saturation_mask)

        # If raw mode, return unprocessed mask
        if raw:
            return combined_mask

        # 3. PROTECT FIRST - Create safe zones before any refinement
        if self.verbose:
            print("Protecting text and background *before* mask refinement...")

        # Protect white background
        _, background_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
        protected_mask = cv2.bitwise_and(
            combined_mask, cv2.bitwise_not(background_mask)
        )

        # Protect dark text
        if self.protect_text:
            text_protect_mask = self.get_text_protect_mask(gray_image, protected_mask)
            protected_mask = cv2.bitwise_and(
                protected_mask, cv2.bitwise_not(text_protect_mask)
            )

        # 4. REFINE SECOND - Apply morphological operations to protected mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

        # Remove noise first (opening)
        opened_mask = cv2.morphologyEx(
            protected_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )

        # Fill gaps in watermark regions (closing)
        closed_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        return closed_mask

    def _traditional_detect_mask(self, image_rgb, page_num=1, progress=None, task_id=None, raw=False):
        """Traditional watermark detection using color analysis and structure validation.

        Args:
            image_rgb: Input image in RGB format
            page_num: Page number (1-indexed) for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates
            raw: If True, return raw mask without protection/refinement

        Returns:
            Binary mask of detected watermark regions
        """
        if self.auto_detect_color and self.watermark_color is None:
            if progress and task_id and not raw:
                progress.update(task_id, description=f"[yellow]Page {page_num}: Auto-detecting color...")
            self.detect_watermark_color(image_rgb)

        if self.verbose:
            print("Converting image to grayscale...")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        if progress and task_id and not raw:
            progress.update(task_id, description=f"[yellow]Page {page_num}: Analyzing structure...")

        if self.watermark_color is not None:
            mask = self._precise_color_based_detection(image_rgb, gray, raw=raw)
        else:
            mask = self._automatic_detection_mode(image_rgb, gray, raw=raw)

        if self.verbose:
            detected_pixels = np.count_nonzero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            percentage = (detected_pixels / total_pixels) * 100
            print(f"Detected watermark coverage: {percentage:.2f}%")

        return mask

    def _unified_protection_and_refinement(self, image_rgb, watermark_mask, qr_mask=None, page_num=1, progress=None, task_id=None):
        """Apply unified protection and refinement to both watermark and QR masks.

        Args:
            image_rgb: Input image in RGB format
            watermark_mask: Raw watermark mask
            qr_mask: Raw QR code mask (optional)
            page_num: Page number for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates

        Returns:
            Combined mask with unified processing
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # STEP 1: Apply protections (30-60% progress)
        if progress and task_id:
            progress.update(task_id, advance=10, description=f"[yellow]Page {page_num}: Applying protections...")

        # Apply background protection to both masks
        _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        protected_watermark_mask = cv2.bitwise_and(watermark_mask, cv2.bitwise_not(background_mask))

        # Apply text protection ONLY to watermark regions (not QR codes)
        if self.protect_text:
            # Don't apply text protection if watermark color is in text-like range (100-150)
            target_gray = None
            if self.watermark_color is not None:
                target_gray = int(np.mean(self.watermark_color[:3]))

            # Only protect text if watermark color is not text-like
            if target_gray is None or not (100 <= target_gray <= 150):
                text_protect_mask = self.get_text_protect_mask(gray, protected_watermark_mask)
                protected_watermark_mask = cv2.bitwise_and(
                    protected_watermark_mask, cv2.bitwise_not(text_protect_mask)
                )

        # QR codes do NOT get background protection (they're typically on white backgrounds)
        protected_qr_mask = None
        if qr_mask is not None:
            protected_qr_mask = qr_mask

        # STEP 2: Unified refinement (60-90% progress)
        if progress and task_id:
            progress.update(task_id, advance=10, description=f"[yellow]Page {page_num}: Refining masks...")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))

        # Apply same morphological operations to both masks
        refined_watermark_mask = cv2.morphologyEx(
            protected_watermark_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        refined_watermark_mask = cv2.morphologyEx(
            refined_watermark_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        refined_qr_mask = None
        if protected_qr_mask is not None:
            refined_qr_mask = cv2.morphologyEx(
                protected_qr_mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            refined_qr_mask = cv2.morphologyEx(
                refined_qr_mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )

        # STEP 3: Final combination (90-100% progress)
        if progress and task_id:
            progress.update(task_id, advance=5, description=f"[yellow]Page {page_num}: Combining masks...")

        # Merge masks: QR codes take priority over watermarks
        combined_mask = refined_watermark_mask.copy()
        if refined_qr_mask is not None:
            # Dilate QR mask slightly for complete removal using RECT kernel for better coverage
            qr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            qr_mask_dilated = cv2.dilate(refined_qr_mask, qr_kernel, iterations=1)
            combined_mask = cv2.bitwise_or(combined_mask, qr_mask_dilated)

        return combined_mask

    def _unified_yolo_processing(self, image_rgb, yolo_mask, page_num=1, progress=None, task_id=None):
        """Handle YOLO masks with unified QR processing.

        YOLO masks are already processed, so we only need to:
        1. Detect QR codes if enabled
        2. Apply minimal unified processing (background protection, combination)

        Args:
            image_rgb: Input image in RGB format
            yolo_mask: Processed YOLO watermark mask
            page_num: Page number for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates

        Returns:
            Combined mask with QR codes added
        """
        # Detect QR codes if enabled
        qr_mask = None
        if self.detect_qr_codes:
            if progress and task_id:
                progress.update(task_id, advance=10, description=f"[yellow]Page {page_num}: Detecting QR codes...")
            qr_mask = self.detect_qr_codes_mask(image_rgb, page_num)

        # Apply minimal unified processing
        if progress and task_id:
            progress.update(task_id, advance=10, description=f"[yellow]Page {page_num}: Finalizing mask...")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Apply background protection to both masks
        _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        protected_yolo_mask = cv2.bitwise_and(yolo_mask, cv2.bitwise_not(background_mask))

        combined_mask = protected_yolo_mask.copy()
        if qr_mask is not None:
            # QR masks do NOT get background protection (they're typically on white backgrounds)
            protected_qr_mask = qr_mask
            # Dilate QR mask slightly for complete removal
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            qr_mask_dilated = cv2.dilate(protected_qr_mask, kernel, iterations=1)
            combined_mask = cv2.bitwise_or(combined_mask, qr_mask_dilated)

        return combined_mask

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
