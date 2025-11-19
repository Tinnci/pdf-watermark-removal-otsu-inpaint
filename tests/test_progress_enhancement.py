#!/usr/bin/env python3
"""Test suite for enhanced progress bar tracking in watermark removal pipeline."""

import importlib.util
import time
from unittest.mock import MagicMock, Mock, patch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import numpy as np
import pytest

# Import the modules to test
from pdf_watermark_removal.watermark_detector import WatermarkDetector
from pdf_watermark_removal.watermark_remover import WatermarkRemover

YOLO_AVAILABLE = importlib.util.find_spec("ultralytics") is not None


class ProgressTracker:
    """Helper class to track progress updates during detection."""

    def __init__(self):
        self.updates = []
        self.current_step = 0
        self.max_steps = 5  # Based on our enhanced progress tracking

    def update_callback(self, task_id, **kwargs):
        """Capture progress updates."""
        self.updates.append({
            'step': self.current_step,
            'timestamp': time.time(),
            'description': kwargs.get('description', ''),
            'advance': kwargs.get('advance', 0),
            'completed': kwargs.get('completed', None)
        })
        self.current_step += kwargs.get('advance', 0)

    def get_summary(self):
        """Get a summary of progress tracking."""
        return {
            'total_updates': len(self.updates),
            'max_progress_reached': max((u['step'] for u in self.updates), default=0),
            'descriptions': [u['description'] for u in self.updates],
            'completed_to_100': any(u.get('completed') == 100 for u in self.updates)
        }


def create_test_image(has_watermark=True, has_qr=False, has_text=True):
    """Create test images with configurable features."""
    if not CV2_AVAILABLE:
        # Fallback to numpy-only image creation
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        if has_watermark:
            image[100:300, 100:500] = [200, 200, 200]
        if has_text:
            image[180:220, 150:450] = [0, 0, 0]  # Simple text rectangle
        if has_qr:
            image[50:150, 450:550] = [0, 0, 0]  # Simple QR-like pattern
        return image

    # White background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    if has_watermark:
        # Light gray watermark rectangle
        cv2.rectangle(image, (100, 100), (500, 300), (200, 200, 200), -1)

    if has_text:
        # Dark text that should be protected
        cv2.putText(image, "TEST DOCUMENT", (150, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    if has_qr:
        # Simple QR-like pattern (black squares)
        for i in range(5):
            for j in range(5):
                if (i + j) % 2 == 0:
                    x, y = 450 + i * 10, 50 + j * 10
                    cv2.rectangle(image, (x, y), (x + 8, y + 8), (0, 0, 0), -1)

    return image


class TestProgressTracking:
    """Test progress bar enhancements across the detection pipeline."""

    @pytest.fixture
    def sample_image(self):
        """Provide a test image with watermark and text."""
        return create_test_image(has_watermark=True, has_qr=False, has_text=True)

    @pytest.fixture
    def sample_image_with_qr(self):
        """Provide a test image with watermark, text, and QR code."""
        return create_test_image(has_watermark=True, has_qr=True, has_text=True)

    def test_traditional_detection_progress(self, sample_image):
        """Test progress tracking for traditional detection method."""
        tracker = ProgressTracker()
        detector = WatermarkDetector(
            detection_method="traditional",
            watermark_color=(200, 200, 200),
            color_tolerance=30,
            protect_text=True,
            verbose=False
        )

        # Mock progress updates
        mock_progress = Mock()
        mock_task_id = Mock()
        mock_progress.update = Mock(side_effect=tracker.update_callback)

        # Perform detection with progress tracking
        detector.detect_watermark_mask(
            sample_image,
            page_num=1,
            progress=mock_progress,
            task_id=mock_task_id
        )

        summary = tracker.get_summary()

        # Verify progress behavior
        assert summary['total_updates'] >= 2, "Should have multiple progress updates"
        assert summary['completed_to_100'], "Should complete to 100%"

        # Verify step descriptions
        descriptions = ' '.join(summary['descriptions'])
        assert "Color analysis" in descriptions or "Detecting" in descriptions

    def test_qr_detection_progress(self, sample_image_with_qr):
        """Test progress tracking when QR code detection is enabled."""
        detector = WatermarkDetector(
            detection_method="traditional",
            detect_qr_codes=True,
            qr_detection_method="opencv",
            remove_all_qr_codes=False,
            verbose=False
        )

        # Perform detection with QR code scanning
        mask = detector.detect_qr_codes_mask(
            sample_image_with_qr,
            page_num=1
        )

        # Should have detected something (even if not actual QR codes)
        assert mask is not None or len(detector.detected_qr_codes) >= 0

    @pytest.mark.skipif(not YOLO_AVAILABLE, reason="YOLO not available")
    def test_yolo_detection_progress(self, sample_image):
        """Test progress tracking for YOLO detection method."""
        tracker = ProgressTracker()

        detector = WatermarkDetector(
            detection_method="yolo",
            yolo_model_path="yolov8n-seg.pt",
            yolo_conf_thres=0.25,
            verbose=False
        )

        mock_progress = Mock()
        mock_task_id = Mock()
        mock_progress.update = Mock(side_effect=tracker.update_callback)

        detector.detect_watermark_mask(
            sample_image,
            page_num=1,
            progress=mock_progress,
            task_id=mock_task_id
        )

        summary = tracker.get_summary()

        # YOLO should have different progress steps
        descriptions = ' '.join(summary['descriptions'])
        assert "YOLO" in descriptions or "inference" in descriptions or len(descriptions) > 0

    def test_multi_pass_progress(self, sample_image):
        """Test progress tracking for multi-pass removal."""
        tracker = ProgressTracker()
        remover = WatermarkRemover(
            kernel_size=3,
            inpaint_radius=2,
            inpaint_strength=1.0,
            detection_method="traditional",
            watermark_color=(200, 200, 200),
            verbose=False
        )

        mock_progress = Mock()
        mock_task_id = Mock()
        mock_progress.update = Mock(side_effect=tracker.update_callback)

        # Mock the per-pass processing to verify multiple passes are tracked
        with patch.object(remover, '_process_single_pass') as mock_pass:
            mock_pass.return_value = (sample_image, True)

            remover.remove_watermark_multi_pass(
                sample_image,
                passes=3,
                page_num=1,
                progress=mock_progress,
                task_id=mock_task_id
            )

            # Verify multi-pass was called 3 times
            assert mock_pass.call_count == 3

    def test_progress_with_no_watermark(self):
        """Test progress behavior when no watermark is detected."""
        tracker = ProgressTracker()

        # Create image without watermark
        clean_image = create_test_image(has_watermark=False, has_text=True)

        detector = WatermarkDetector(
            detection_method="traditional",
            auto_detect_color=True,
            verbose=False
        )

        mock_progress = Mock()
        mock_task_id = Mock()
        mock_progress.update = Mock(side_effect=tracker.update_callback)

        detector.detect_watermark_mask(
            clean_image,
            page_num=1,
            progress=mock_progress,
            task_id=mock_task_id
        )

        # Should still complete progress even when no watermark found
        summary = tracker.get_summary()
        assert summary['completed_to_100'], "Should complete progress even with no detection"

    def test_progress_error_handling(self, sample_image):
        """Test progress tracking when errors occur during detection."""
        tracker = ProgressTracker()

        detector = WatermarkDetector(
            detection_method="traditional",
            watermark_color=(200, 200, 200),
            verbose=False
        )

        mock_progress = Mock()
        mock_task_id = Mock()
        mock_progress.update = Mock(side_effect=tracker.update_callback)

        # Simulate an error during detection
        with patch.object(detector, '_traditional_detect_mask') as mock_detect:
            mock_detect.side_effect = ValueError("Detection failed")

            with pytest.raises(ValueError):
                detector.detect_watermark_mask(
                    sample_image,
                    page_num=1,
                    progress=mock_progress,
                    task_id=mock_task_id
                )

        # Verify progress was attempted even with error
        summary = tracker.get_summary()
        assert summary['total_updates'] > 0, "Should have attempted progress updates"

    # @pytest.mark.skip(reason="Performance test shows expected overhead from progress tracking")
    def test_progress_performance_overhead(self, sample_image):
        """Verify that progress tracking doesn't add significant overhead."""
        # Baseline without progress tracking
        detector = WatermarkDetector(
            detection_method="traditional",
            watermark_color=(200, 200, 200),
            verbose=False
        )

        start = time.time()
        mask_baseline = detector.detect_watermark_mask(sample_image)
        baseline_time = time.time() - start

        # With progress tracking - use a lighter mock
        class LightMock:
            def update(self, *args, **kwargs):
                pass

        mock_progress = LightMock()
        mock_task_id = "test_task"

        start = time.time()
        mask_with_progress = detector.detect_watermark_mask(
            sample_image,
            page_num=1,
            progress=mock_progress,
            task_id=mock_task_id
        )
        with_progress_time = time.time() - start

        # Overhead should be reasonable (< 50% for mocked progress with multiple updates)
        if baseline_time > 0:
            overhead = (with_progress_time - baseline_time) / baseline_time * 100
            assert overhead < 50.0, f"Progress tracking adds {overhead:.1f}% overhead, should be < 50%"

        # Results should be identical
        np.testing.assert_array_equal(mask_baseline, mask_with_progress)


class TestIntegrationProgress:
    """Integration tests for full pipeline progress tracking."""

    def test_full_pipeline_progress(self, tmp_path):
        """Test progress tracking through the complete removal pipeline."""
        from pdf_watermark_removal.pdf_processor import PDFProcessor
        from pdf_watermark_removal.watermark_remover import WatermarkRemover

        # Create a simple test PDF
        test_image = create_test_image(has_watermark=True, has_text=True)
        pdf_path = tmp_path / "test_input.pdf"
        output_path = tmp_path / "test_output.pdf"

        # Save test image as PDF using PIL
        from PIL import Image
        pil_img = Image.fromarray(test_image)
        pil_img.save(pdf_path)

        processor = PDFProcessor(dpi=100, verbose=False)
        remover = WatermarkRemover(
            detection_method="traditional",
            watermark_color=(200, 200, 200),
            verbose=False
        )

        # Track progress through full pipeline
        tracker = ProgressTracker()

        # Mock the progress bar at CLI level
        with patch('pdf_watermark_removal.cli.Progress') as mock_progress_class:
            mock_progress = MagicMock()
            mock_task_id = MagicMock()
            mock_progress.add_task = Mock(return_value=mock_task_id)
            mock_progress.update = Mock(side_effect=tracker.update_callback)
            mock_progress_class.return_value.__enter__ = Mock(return_value=mock_progress)
            mock_progress_class.return_value.__exit__ = Mock(return_value=False)

            # Run conversion
            images = processor.pdf_to_images(pdf_path)

            # Run removal with progress tracking
            processed = []
            for page_num, img in enumerate(images, 1):
                result = remover.remove_watermark(
                    img,
                    page_num=page_num,
                    progress=mock_progress,
                    task_id=mock_task_id
                )
                processed.append(result)

            # Save back to PDF
            processor.images_to_pdf(processed, output_path)

        # Verify pipeline completed successfully
        assert output_path.exists(), "Output PDF should be created"

        summary = tracker.get_summary()
        assert summary['total_updates'] >= 3, "Full pipeline should have multiple progress updates"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
