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
        """Initialize the watermark remover.

        Args:
            kernel_size: Size of morphological kernel for detection
            inpaint_radius: Radius for inpainting algorithm
            inpaint_strength: Strength of inpainting (0.5=light, 1.0=medium, 1.5=strong)
            verbose: Enable verbose logging
            auto_detect_color: Automatically detect watermark color
            watermark_color: Watermark color (R, G, B) or None
            protect_text: Protect dark text from being removed
            color_tolerance: Color matching tolerance (0-255, default 30)
            detection_method: 'traditional' or 'yolo'
            yolo_model_path: Path to YOLO model
            yolo_conf_thres: YOLO confidence threshold
            yolo_device: YOLO device ('cpu', 'cuda', 'auto')
            yolo_version: 'v8', 'v11', or 'v11' (default: v8)
            auto_download_model: Automatically download YOLO model if not found
            detect_qr_codes: Enable QR code detection
            qr_detection_method: QR detection method ('opencv' or 'pyzbar')
            remove_all_qr_codes: Remove all detected QR codes
            qr_code_categories_to_remove: List of QR code categories to remove
        """
        self.detector = WatermarkDetector(
            kernel_size=kernel_size,
            verbose=verbose,
            auto_detect_color=auto_detect_color,
            watermark_color=watermark_color,
            protect_text=protect_text,
            color_tolerance=color_tolerance,
            detection_method=detection_method,
            yolo_model_path=yolo_model_path,
            yolo_conf_thres=yolo_conf_thres,
            yolo_device=yolo_device,
            yolo_version=yolo_version,
            auto_download_model=auto_download_model,
            detect_qr_codes=detect_qr_codes,
            qr_detection_method=qr_detection_method,
            remove_all_qr_codes=remove_all_qr_codes,
            qr_code_categories_to_remove=qr_code_categories_to_remove,
        )
        self.inpaint_radius = inpaint_radius
        self.inpaint_strength = inpaint_strength
        self.verbose = verbose
        self.last_stats = {}  # Track last removal statistics

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
        result = (
            original.astype(np.float32) * (1 - blend_factor)
            + inpainted.astype(np.float32) * blend_factor
        )

        return result.astype(np.uint8)

    def get_strength_info(self):
        """Get current strength configuration and last statistics.

        Returns:
            dict with strength info
        """
        return {
            "strength": self.inpaint_strength,
            "radius": self.inpaint_radius,
            "blend_mode": "100%"
            if self.inpaint_strength >= 1.5
            else f"{self.inpaint_strength * 100:.0f}%",
            "last_coverage": self.last_stats.get("coverage", 0.0),
            "last_radius": self.last_stats.get("dynamic_radius", 0),
        }

    def _detect_and_refine_mask(self, image_rgb, page_num=1, progress=None, task_id=None):
        """Detect and refine watermark mask.
        
        Dynamically adjusts refinement parameters when QR codes are present
        to prevent them from being filtered out due to size constraints.
        """
        mask = self.detector.detect_watermark_mask(image_rgb, page_num, progress, task_id)
        
        # If QR codes were detected on this page, increase max_area to prevent filtering them out.
        # QR codes are large components that exceed typical max_area (5000), so we set it to
        # the full image area to ensure they're preserved in the refined mask.
        has_qr_codes_on_page = self.detector.current_page_qr_codes
        
        if self.detector.method == 'traditional' and has_qr_codes_on_page:
            # Use full image area as max_area to preserve QR codes
            image_area = mask.shape[0] * mask.shape[1]
            return self.detector.refine_mask(mask, max_area=image_area)
        else:
            # Default refinement for YOLO or traditional without QR codes
            return self.detector.refine_mask(mask)

    def _calculate_dynamic_radius(self, mask):
        """Calculate dynamic inpaint radius based on coverage and strength."""
        watermark_coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        dynamic_radius = max(
            2,
            int(self.inpaint_radius + watermark_coverage * 10 * self.inpaint_strength),
        )
        return watermark_coverage, dynamic_radius

    def _record_removal_stats(self, watermark_coverage, dynamic_radius):
        """Record removal statistics for later reference."""
        self.last_stats = {
            "coverage": watermark_coverage * 100,
            "dynamic_radius": dynamic_radius,
            "strength": self.inpaint_strength,
        }

    def _apply_inpainting(self, image_rgb, mask, dynamic_radius):
        """Apply inpainting algorithm to remove watermark."""
        image_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(image_bgr, mask, dynamic_radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    def _apply_final_blending(self, image_rgb, restored, mask):
        """Apply strength blending if not at maximum strength."""
        if self.inpaint_strength < 1.5:
            return self.apply_inpaint_strength(
                image_rgb, restored, mask, self.inpaint_strength
            )
        return restored

    def remove_watermark(self, image_rgb, page_num=1, progress=None, task_id=None):
        """Remove watermark from an image.

        Args:
            image_rgb: Input image in RGB format (0-255)
            page_num: Page number (1-indexed) for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates

        Returns:
            Image with watermark removed (RGB format)
        """
        if self.verbose:
            print("Detecting watermark regions...")

        if progress and task_id:
            progress.update(task_id, description=f"[yellow]Page {page_num}: Detecting regions...")

        mask = self._detect_and_refine_mask(image_rgb, page_num, progress, task_id)

        if np.count_nonzero(mask) == 0:
            if self.verbose:
                print("No watermark detected, returning original image")
            if progress and task_id:
                progress.update(task_id, description=f"[green]✓ Page {page_num}: No watermark found")
                progress.update(task_id, completed=100)
            return image_rgb.astype(np.uint8)

        if progress and task_id:
            progress.update(task_id, description=f"[yellow]Page {page_num}: Applying inpainting...")

        if self.verbose:
            print(f"Applying inpainting with radius {self.inpaint_radius}...")

        watermark_coverage, dynamic_radius = self._calculate_dynamic_radius(mask)
        self._record_removal_stats(watermark_coverage, dynamic_radius)

        if self.verbose:
            coverage_pct = watermark_coverage * 100
            print(
                f"Watermark coverage: {coverage_pct:.2f}%, "
                f"inpaint strength: {self.inpaint_strength}, "
                f"dynamic radius: {dynamic_radius}"
            )

        restored = self._apply_inpainting(image_rgb, mask, dynamic_radius)
        result = self._apply_final_blending(image_rgb, restored, mask)

        if progress and task_id:
            progress.update(task_id, description=f"[green]✓ Page {page_num}: Complete")
            progress.update(task_id, completed=100)

        return result

    def _process_single_pass(self, result, pass_num, passes, page_num=1, progress=None, task_id=None):
        """Process a single removal pass."""
        if self.verbose:
            print(f"Pass {pass_num + 1}/{passes}")

        mask = self._detect_and_refine_mask(result, page_num, progress, task_id)

        if np.count_nonzero(mask) == 0:
            if self.verbose:
                print("No watermark detected, stopping")
            return result, False

        if pass_num > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)

        watermark_coverage, inpaint_radius = self._calculate_dynamic_radius(mask)

        if self.verbose:
            coverage_pct = watermark_coverage * 100
            print(
                f"Watermark coverage: {coverage_pct:.2f}%, "
                f"strength: {self.inpaint_strength}, radius: {inpaint_radius}"
            )

        result_inpainted = cv2.inpaint(
            result.astype(np.uint8), mask, inpaint_radius, cv2.INPAINT_TELEA
        )

        result = self._apply_final_blending(result, result_inpainted, mask)
        return result, True

    def remove_watermark_multi_pass(self, image_rgb, passes=2, page_num=1, progress=None, task_id=None):
        """Remove watermark using multiple passes with progressive mask expansion.

        Uses a smarter approach: instead of reprocessing the entire image multiple
        times (which causes over-smoothing), it expands the mask progressively and
        applies inpainting once per pass with updated parameters.

        Args:
            image_rgb: Input image in RGB format
            passes: Number of removal passes
            page_num: Page number (1-indexed) for tracking
            progress: Rich progress instance for updates
            task_id: Progress task ID for updates

        Returns:
            Image with watermark removed
        """
        result = image_rgb.copy()

        for pass_num in range(passes):
            if progress and task_id:
                progress.update(
                    task_id,
                    description=f"[yellow]Page {page_num}: Pass {pass_num+1}/{passes}..."
                )

            result, has_watermark = self._process_single_pass(
                result, pass_num, passes, page_num, progress, task_id
            )
            if not has_watermark:
                break

        return result
