#!/usr/bin/env python3
"""Debug the actual modified detector step by step."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pdf_watermark_removal.watermark_detector import WatermarkDetector


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


def debug_detector_internals():
    """Debug by replicating the exact detector logic."""
    print("=== DEBUGGING DETECTOR INTERNALS ===")

    image = create_test_image_with_text()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print(f"Image shape: {image.shape}")
    print(f"Gray range: {gray.min()} - {gray.max()}")

    # Replicate the exact detector parameters
    kernel_size = 3
    color_tolerance = 30
    watermark_color = (200, 200, 200)
    protect_text = True
    verbose = True  # noqa: F841

    print("\nDetector parameters:")
    print(f"  Watermark color: {watermark_color}")
    print(f"  Color tolerance: {color_tolerance}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Protect text: {protect_text}")

    # Step 1: Create raw color mask (exactly as in detector)
    target_gray = int(np.mean(watermark_color))
    color_diff = np.abs(gray.astype(int) - target_gray)
    raw_mask = (color_diff < color_tolerance).astype(np.uint8) * 255

    print("\nStep 1 - Raw color mask:")
    print(f"  Target gray: {target_gray}")
    print(f"  Raw mask pixels: {np.count_nonzero(raw_mask)}")

    # Step 2: Text protection FIRST
    if protect_text:
        print("\nStep 2a - Text protection:")

        # Create text protection mask (from get_text_protect_mask)
        _, core_text = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        print(f"  Core text pixels: {np.count_nonzero(core_text)}")

        # Expand for anti-aliased edges
        expand_pixels = 2
        kernel_expand = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
        )
        expanded_text = cv2.dilate(core_text, kernel_expand, iterations=1)
        print(f"  Expanded text pixels: {np.count_nonzero(expanded_text)}")

        # Remove noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_text = cv2.morphologyEx(
            expanded_text, cv2.MORPH_OPEN, kernel_clean, iterations=1
        )
        print(f"  Cleaned text pixels: {np.count_nonzero(cleaned_text)}")

        # Refine with watermark mask (remove areas already in watermark)
        text_protect_mask = cv2.bitwise_and(cleaned_text, cv2.bitwise_not(raw_mask))
        print(f"  Final text protection pixels: {np.count_nonzero(text_protect_mask)}")

        # Apply text protection
        protected_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(text_protect_mask))
        print(f"  After text protection: {np.count_nonzero(protected_mask)}")
        print(
            f"  Text pixels removed: {np.count_nonzero(raw_mask) - np.count_nonzero(protected_mask)}"
        )
    else:
        protected_mask = raw_mask.copy()

    # Step 3: Background protection SECOND
    print("\nStep 2b - Background protection:")
    _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    final_protected = cv2.bitwise_and(protected_mask, cv2.bitwise_not(background_mask))
    print(f"  Background mask pixels: {np.count_nonzero(background_mask)}")
    print(f"  After background protection: {np.count_nonzero(final_protected)}")

    # Step 4: Morphological operations
    print("\nStep 3 - Morphological operations:")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    opened_mask = cv2.morphologyEx(
        final_protected, cv2.MORPH_OPEN, kernel, iterations=1
    )
    print(f"  After opening: {np.count_nonzero(opened_mask)}")

    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    print(f"  After closing: {np.count_nonzero(closed_mask)}")

    # Check specific areas
    text_area = closed_mask[80:120, 100:300]
    text_pixels = np.count_nonzero(text_area)
    print(f"\nFinal text area check: {text_pixels} pixels in text region")

    return closed_mask


def test_with_actual_detector():
    """Test with the actual modified detector."""
    print("\n=== TESTING WITH ACTUAL DETECTOR ===")

    image = create_test_image_with_text()

    # Create detector with exact same parameters
    detector = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=30,
        kernel_size=3,
        protect_text=True,
        verbose=True,
    )

    mask = detector.detect_watermark_mask(image)

    print(f"Detector result: {np.count_nonzero(mask)} pixels")

    # Compare with our manual calculation
    manual_mask = debug_detector_internals()
    print(f"Manual calculation: {np.count_nonzero(manual_mask)} pixels")

    # Check if they match
    match = np.array_equal(mask, manual_mask)
    print(f"Results match: {'YES' if match else 'NO'}")

    if not match:
        print("Differences found - analyzing...")
        diff = cv2.bitwise_xor(mask, manual_mask)
        diff_pixels = np.count_nonzero(diff)
        print(f"Difference pixels: {diff_pixels}")

        # Save visualization
        viz = np.hstack(
            [
                image,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(manual_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR),
            ]
        )
        cv2.imwrite("detector_vs_manual.png", viz)
        print("Saved comparison to detector_vs_manual.png")

    return mask


def test_text_extraction():
    """Test if we can properly extract text from the image."""
    print("\n=== TESTING TEXT EXTRACTION ===")

    image = create_test_image_with_text()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Try different thresholds for text extraction
    thresholds = [100, 120, 140, 150, 160, 180]

    for thresh in thresholds:
        _, text_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        text_pixels = np.count_nonzero(text_mask)

        # Check text area specifically
        text_area = text_mask[80:120, 100:300]
        area_pixels = np.count_nonzero(text_area)

        print(
            f"Threshold {thresh:3d}: Total={text_pixels:5d}, TextArea={area_pixels:4d}"
        )

    # Try with expansion
    print("\nWith expansion (3 pixels):")
    _, core_text = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    expanded = cv2.dilate(core_text, kernel, iterations=1)

    text_area_expanded = expanded[80:120, 100:300]
    print(f"Expanded text area: {np.count_nonzero(text_area_expanded)} pixels")


if __name__ == "__main__":
    print("DETAILED DEBUG OF TEXT PROTECTION ISSUE")
    print("=" * 60)

    manual_result = debug_detector_internals()
    actual_result = test_with_actual_detector()
    test_text_extraction()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")

    # Final assessment
    text_area_manual = manual_result[80:120, 100:300]
    text_area_actual = actual_result[80:120, 100:300]

    manual_text_pixels = np.count_nonzero(text_area_manual)
    actual_text_pixels = np.count_nonzero(text_area_actual)

    print("\nFINAL ASSESSMENT:")
    print(f"Manual calculation text pixels: {manual_text_pixels}")
    print(f"Actual detector text pixels: {actual_text_pixels}")
    print(
        f"Text protection working: {'NO' if actual_text_pixels > 1000 else 'PARTIAL'}"
    )
