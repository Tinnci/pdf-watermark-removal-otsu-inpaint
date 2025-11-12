#!/usr/bin/env python3
"""Debug script to understand the protection issue."""

import cv2
import numpy as np

from src.pdf_watermark_removal.watermark_detector import WatermarkDetector


def create_test_image_with_text():
    """Create a test image with watermark and text to test protection order."""
    # Create a white background
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add light gray watermark (simulating the issue)
    watermark_color = (200, 200, 200)  # Light gray
    cv2.rectangle(image, (50, 50), (350, 150), watermark_color, -1)

    # Add dark text that should be protected
    text_color = (0, 0, 0)  # Pure black
    cv2.putText(
        image, "PROTECT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3
    )
    cv2.putText(image, "TEXT", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

    return image


def _create_color_mask(gray_image, target_gray, color_tolerance):
    """Creates a raw color mask based on a target grayscale value and tolerance."""
    color_diff = np.abs(gray_image.astype(int) - target_gray)
    color_mask = (color_diff < color_tolerance).astype(np.uint8) * 255
    return color_mask


def _create_text_protection_mask(gray_image, threshold, kernel_size):
    """Creates a mask to protect dark text areas."""
    _, text_protect = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel_protect = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    text_protect = cv2.morphologyEx(
        text_protect, cv2.MORPH_OPEN, kernel_protect, iterations=1
    )
    return text_protect


def _apply_protection(color_mask, text_protect_mask):
    """Applies text protection to the color mask."""
    return cv2.bitwise_and(color_mask, cv2.bitwise_not(text_protect_mask))


def _apply_background_protection(protected_mask, gray_image, threshold):
    """Applies background protection to remove very light areas from the mask."""
    _, background_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(protected_mask, cv2.bitwise_not(background_mask))


def _create_debug_visualization(
    original_image, color_mask, text_protect_mask, protected_mask, final_protected_mask
):
    """Creates a comparison image for visualizing each step of the protection process."""
    step1 = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    step2 = cv2.cvtColor(text_protect_mask, cv2.COLOR_GRAY2BGR)
    step3 = cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2BGR)
    step4 = cv2.cvtColor(final_protected_mask, cv2.COLOR_GRAY2BGR)

    comparison = np.vstack(
        [
            np.hstack([original_image, step1]),  # Original vs Raw color mask
            np.hstack([step2, step3]),  # Text protection vs After text protection
            np.hstack([step4, step1]),  # Final vs Raw (to see difference)
        ]
    )
    cv2.imwrite("debug_protection_steps.png", comparison)
    print("   - Saved visualization to debug_protection_steps.png")


def debug_protection_steps():
    """Debug each step of the protection process."""
    print("=== DEBUGGING PROTECTION STEPS ===")

    # Create test image
    image = create_test_image_with_text()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print(f"Image shape: {image.shape}")
    print(f"Gray shape: {gray.shape}")
    print(f"Gray range: {gray.min()} - {gray.max()}")

    # Step 1: Create raw color mask
    target_gray = 200  # Our watermark color
    color_tolerance = 30
    color_mask = _create_color_mask(gray, target_gray, color_tolerance)

    print("\n1. Raw color mask:")
    print(f"   - Non-zero pixels: {np.count_nonzero(color_mask)}")
    print(f"   - Target gray: {target_gray}")
    print(f"   - Tolerance: ±{color_tolerance}")
    print(
        f"   - Range: {target_gray - color_tolerance} - {target_gray + color_tolerance}"
    )

    # Step 2: Create text protection mask
    text_threshold = 150
    text_kernel_size = (2, 2)
    text_protect = _create_text_protection_mask(gray, text_threshold, text_kernel_size)

    print("\n2. Text protection mask:")
    print(f"   - Threshold: {text_threshold} (protects 0-{text_threshold})")
    print(f"   - Non-zero pixels: {np.count_nonzero(text_protect)}")
    print(
        f"   - Text pixels that should be protected: {np.count_nonzero(text_protect)}"
    )

    # Check specific text area
    text_area = gray[80:120, 100:300]  # Approximate text region
    print(
        f"   - Text area gray values: min={text_area.min()}, max={text_area.max()}, mean={text_area.mean():.1f}"
    )

    # Step 3: Apply protection
    protected_mask = _apply_protection(color_mask, text_protect)

    print("\n3. After text protection:")
    print(f"   - Non-zero pixels: {np.count_nonzero(protected_mask)}")
    print(
        f"   - Text pixels removed: {np.count_nonzero(color_mask) - np.count_nonzero(protected_mask)}"
    )

    # Step 4: Background protection
    background_threshold = 250
    final_protected = _apply_background_protection(
        protected_mask, gray, background_threshold
    )

    print("\n4. After background protection:")
    print(f"   - Non-zero pixels: {np.count_nonzero(final_protected)}")
    print(
        f"   - Background pixels removed: {np.count_nonzero(protected_mask) - np.count_nonzero(final_protected)}"
    )

    # Visualize each step
    print("\n5. Creating visualization...")
    _create_debug_visualization(
        image, color_mask, text_protect, protected_mask, final_protected
    )

    # Test with actual detector
    print("\n=== TESTING WITH ACTUAL DETECTOR ===")
    detector = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=30,
        protect_text=True,
        verbose=True,
    )

    final_mask = detector.detect_watermark_mask(image)
    print(f"Final detector result: {np.count_nonzero(final_mask)} pixels")

    # Compare our manual calculation with detector result
    print("\nComparison:")
    print(f"Manual calculation: {np.count_nonzero(final_protected)}")
    print(f"Detector result: {np.count_nonzero(final_mask)}")
    print(f"Match: {'YES' if np.array_equal(final_protected, final_mask) else 'NO'}")

    return final_mask


def test_text_color_ranges():
    """Test different text colors to see what gets protected."""
    print("\n=== TESTING TEXT COLOR RANGES ===")

    # Create test image with different gray levels
    image = np.ones((100, 256, 3), dtype=np.uint8) * 255

    # Add vertical stripes of different gray levels
    for gray_val in range(0, 256, 10):
        color = (gray_val, gray_val, gray_val)
        cv2.rectangle(image, (gray_val, 20), (gray_val + 5, 80), color, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create text protection mask
    _, text_protect = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel_protect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_protect = cv2.morphologyEx(
        text_protect, cv2.MORPH_OPEN, kernel_protect, iterations=1
    )

    print("Gray level -> Protection status:")
    for gray_val in range(0, 256, 20):
        # Check a small area around each gray level
        area = text_protect[30:70, gray_val : gray_val + 5]
        protected = np.count_nonzero(area) > 0
        print(f"  {gray_val:3d}: {'PROTECTED' if protected else 'NOT PROTECTED'}")

    cv2.imwrite(
        "text_color_test.png",
        np.hstack([image, cv2.cvtColor(text_protect, cv2.COLOR_GRAY2BGR)]),
    )


if __name__ == "__main__":
    print("DEBUGGING TEXT PROTECTION ISSUE")
    print("=" * 50)

    final_mask = debug_protection_steps()
    test_text_color_ranges()

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("Check the generated PNG files for visual analysis")
