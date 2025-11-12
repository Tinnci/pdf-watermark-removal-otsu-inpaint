#!/usr/bin/env python3
"""Step-by-step debug of the protection process."""

import cv2
import numpy as np


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


def manual_protection_process():
    """Manually step through the protection process exactly as implemented."""
    print("=== MANUAL STEP-BY-STEP PROTECTION ===")

    image = create_test_image_with_text()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print(f"Image: {image.shape}, Gray: {gray.shape}")
    print(f"Gray range: {gray.min()} - {gray.max()}")

    # Step 1: Create raw color mask (exactly as in code)
    target_gray = int(np.mean([200, 200, 200]))  # watermark_color
    color_tolerance = 30
    color_diff = np.abs(gray.astype(int) - target_gray)
    color_mask = (color_diff < color_tolerance).astype(np.uint8) * 255

    print("\nStep 1 - Raw color mask:")
    print(f"Target gray: {target_gray}, tolerance: {color_tolerance}")
    print(f"Color mask pixels: {np.count_nonzero(color_mask)}")

    # Step 2: Protect white background
    _, background_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    protected_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(background_mask))

    print("\nStep 2 - After background protection:")
    print(f"Background mask pixels: {np.count_nonzero(background_mask)}")
    print(f"Protected mask pixels: {np.count_nonzero(protected_mask)}")

    # Step 3: Protect dark text (using the NEW improved method)
    print("\nStep 3 - Text protection:")

    # Step 3a: Core text detection
    _, core_text = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    print(f"Core text pixels: {np.count_nonzero(core_text)}")

    # Step 3b: Expand for anti-aliased edges
    expand_pixels = 2
    kernel_expand = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
    )
    expanded_text = cv2.dilate(core_text, kernel_expand, iterations=1)
    print(f"Expanded text pixels: {np.count_nonzero(expanded_text)}")

    # Step 3c: Remove noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_text = cv2.morphologyEx(
        expanded_text, cv2.MORPH_OPEN, kernel_clean, iterations=1
    )
    print(f"Cleaned text pixels: {np.count_nonzero(cleaned_text)}")

    # Step 3d: Refine with watermark mask (this is the key step!)
    text_protect_mask = cv2.bitwise_and(cleaned_text, cv2.bitwise_not(protected_mask))
    print(f"Final text protection pixels: {np.count_nonzero(text_protect_mask)}")

    # Step 3e: Apply protection
    final_protected = cv2.bitwise_and(
        protected_mask, cv2.bitwise_not(text_protect_mask)
    )
    print(f"After text protection: {np.count_nonzero(final_protected)}")
    print(
        f"Text pixels removed: {np.count_nonzero(protected_mask) - np.count_nonzero(final_protected)}"
    )

    # Step 4: Morphological operations (opening then closing)
    print("\nStep 4 - Morphological operations:")
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    opened_mask = cv2.morphologyEx(
        final_protected, cv2.MORPH_OPEN, kernel, iterations=1
    )
    print(f"After opening: {np.count_nonzero(opened_mask)}")

    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    print(f"After closing: {np.count_nonzero(closed_mask)}")

    # Check text area specifically
    text_area = closed_mask[80:120, 100:300]
    text_pixels = np.count_nonzero(text_area)
    print(f"\nText area check: {text_pixels} pixels in text region")

    # Visualize each step
    visualization = np.vstack(
        [
            np.hstack([image, cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)]),
            np.hstack(
                [
                    cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(text_protect_mask, cv2.COLOR_GRAY2BGR),
                ]
            ),
            np.hstack(
                [
                    cv2.cvtColor(final_protected, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2BGR),
                ]
            ),
        ]
    )

    cv2.imwrite("manual_protection_steps.png", visualization)
    print("\nSaved visualization to manual_protection_steps.png")

    return closed_mask


def test_different_text_colors():
    """Test with different text colors to see what works."""
    print("\n=== TESTING DIFFERENT TEXT COLORS ===")

    # Create base image
    base_image = np.ones((100, 600, 3), dtype=np.uint8) * 255

    # Add watermark
    cv2.rectangle(base_image, (50, 30), (550, 70), (200, 200, 200), -1)

    # Add text with different colors
    text_colors = [
        (0, 0, 0),  # Pure black
        (20, 20, 20),  # Very dark gray
        (50, 50, 50),  # Dark gray
        (80, 80, 80),  # Medium-dark gray
        (100, 100, 100),  # Medium gray
        (150, 150, 150),  # Light gray (should not be protected)
    ]

    results = []
    for i, color in enumerate(text_colors):
        image = base_image.copy()
        gray_val = color[0]
        cv2.putText(
            image,
            f"Text{gray_val}",
            (50 + i * 80, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Test protection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, core_text = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Check text area
        text_area = core_text[35:75, 50 + i * 80 : 50 + (i + 1) * 80]
        protected = np.count_nonzero(text_area) > 0

        results.append((gray_val, protected))
        print(
            f"Gray level {gray_val:3d}: {'PROTECTED' if protected else 'NOT PROTECTED'}"
        )

    # Create visualization
    viz_image = np.ones((150, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(viz_image, (50, 60), (550, 90), (200, 200, 200), -1)

    for i, (gray_val, protected) in enumerate(results):
        color = (gray_val, gray_val, gray_val)
        status = "PROTECTED" if protected else "NOT PROT"
        cv2.putText(
            viz_image,
            f"{gray_val}",
            (50 + i * 80, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.putText(
            viz_image,
            status,
            (50 + i * 80, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255) if not protected else (0, 255, 0),
            1,
        )

    cv2.imwrite("text_color_protection_test.png", viz_image)


if __name__ == "__main__":
    print("STEP-BY-STEP DEBUG OF TEXT PROTECTION")
    print("=" * 60)

    final_mask = manual_protection_process()
    test_different_text_colors()

    print("\n" + "=" * 60)
    print("Check the generated PNG files for detailed analysis")
