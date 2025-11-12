#!/usr/bin/env python3
"""Fix the text protection to handle anti-aliased edges."""

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


def improved_text_protection(gray, watermark_mask, gray_threshold=150, expand_pixels=2):
    """Improved text protection that handles anti-aliased edges.

    Args:
        gray: Grayscale image
        watermark_mask: Current watermark mask (to avoid protecting actual watermarks)
        gray_threshold: Gray level below which we consider pixels as text
        expand_pixels: How many pixels to expand the text protection

    Returns:
        Binary mask protecting text areas
    """
    # Step 1: Identify core text regions (very dark pixels)
    _, core_text = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Expand the text region to catch anti-aliased edges
    if expand_pixels > 0:
        kernel_expand = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
        )
        expanded_text = cv2.dilate(core_text, kernel_expand, iterations=1)
    else:
        expanded_text = core_text

    # Step 3: Remove small noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_text = cv2.morphologyEx(
        expanded_text, cv2.MORPH_OPEN, kernel_clean, iterations=1
    )

    # Step 4: Refine by removing areas that are already in the watermark mask
    # (This prevents protecting actual watermarks that happen to be dark)
    refined_text = cv2.bitwise_and(cleaned_text, cv2.bitwise_not(watermark_mask))

    return refined_text


def test_improved_protection():
    """Test the improved text protection."""
    print("Testing improved text protection...")

    image = create_test_image_with_text()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create initial watermark mask
    target_gray = 200
    color_tolerance = 30
    color_diff = np.abs(gray.astype(int) - target_gray)
    watermark_mask = (color_diff < color_tolerance).astype(np.uint8) * 255

    print(f"Initial watermark mask: {np.count_nonzero(watermark_mask)} pixels")

    # Apply improved text protection
    text_protection = improved_text_protection(
        gray, watermark_mask, gray_threshold=150, expand_pixels=3
    )

    print(f"Text protection mask: {np.count_nonzero(text_protection)} pixels")

    # Apply protection
    protected_mask = cv2.bitwise_and(watermark_mask, cv2.bitwise_not(text_protection))

    print(f"After text protection: {np.count_nonzero(protected_mask)} pixels")
    print(
        f"Text pixels removed: {np.count_nonzero(watermark_mask) - np.count_nonzero(protected_mask)}"
    )

    # Visualize
    visualization = np.hstack(
        [
            image,
            cv2.cvtColor(watermark_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(text_protection, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2BGR),
        ]
    )

    cv2.imwrite("improved_text_protection.png", visualization)
    print("Saved visualization to improved_text_protection.png")

    return protected_mask


def modify_watermark_detector():
    """Modify the WatermarkDetector to use improved text protection."""

    # Read the current file
    with open("src/pdf_watermark_removal/watermark_detector.py") as f:
        content = f.read()

    # Find the get_text_protect_mask method and replace it
    old_method = '''    def get_text_protect_mask(self, gray):
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
        return text_protect'''

    new_method = '''    def get_text_protect_mask(self, gray, watermark_mask=None, expand_pixels=2):
        """Create a mask to protect dark text regions from being removed.

        Args:
            gray: Grayscale image
            watermark_mask: Current watermark mask to avoid protecting actual watermarks
            expand_pixels: How many pixels to expand text protection for anti-aliased edges

        Returns:
            Binary mask protecting text areas (255 where text should be protected)
        """
        # Step 1: Identify core text regions (very dark pixels)
        # Use 150 as threshold to catch text and dark regions while avoiding most watermarks
        _, core_text = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Step 2: Expand the text region to catch anti-aliased edges
        # Anti-aliased text edges often have gray values that overlap with watermark range
        if expand_pixels > 0:
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixels*2+1, expand_pixels*2+1))
            expanded_text = cv2.dilate(core_text, kernel_expand, iterations=1)
        else:
            expanded_text = core_text

        # Step 3: Remove small noise from text protection mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_text = cv2.morphologyEx(expanded_text, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        # Step 4: If watermark mask is provided, refine by removing areas already in watermark
        # This prevents protecting actual watermarks that happen to be dark
        if watermark_mask is not None:
            refined_text = cv2.bitwise_and(cleaned_text, cv2.bitwise_not(watermark_mask))
            return refined_text

        return cleaned_text'''

    # Replace the method
    if old_method in content:
        content = content.replace(old_method, new_method)
        print("Successfully updated get_text_protect_mask method")
    else:
        print("Could not find the old method to replace")
        return False

    # Now update the calls to get_text_protect_mask to pass the watermark mask
    # In color-based mode
    old_call_color = """            # Protect dark text
            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray)
                protected_mask = cv2.bitwise_and(
                    protected_mask, cv2.bitwise_not(text_protect_mask)
                )"""

    new_call_color = """            # Protect dark text
            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray, protected_mask)
                protected_mask = cv2.bitwise_and(
                    protected_mask, cv2.bitwise_not(text_protect_mask)
                )"""

    # In automatic mode
    old_call_auto = """            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray)
                mask = cv2.bitwise_and(mask_no_bg, cv2.bitwise_not(text_protect_mask))"""

    new_call_auto = """            if self.protect_text:
                text_protect_mask = self.get_text_protect_mask(gray, mask_no_bg)
                mask = cv2.bitwise_and(mask_no_bg, cv2.bitwise_not(text_protect_mask))"""

    content = content.replace(old_call_color, new_call_color)
    content = content.replace(old_call_auto, new_call_auto)

    # Write back the modified content
    with open("src/pdf_watermark_removal/watermark_detector.py", "w") as f:
        f.write(content)

    print("Successfully modified WatermarkDetector")
    return True


def test_modified_detector():
    """Test the modified detector."""
    print("\n=== TESTING MODIFIED DETECTOR ===")

    image = create_test_image_with_text()

    # Test with modified detector
    detector = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=30,
        protect_text=True,
        verbose=True,
    )

    mask = detector.detect_watermark_mask(image)

    # Check text protection
    text_area = mask[80:120, 100:300]
    text_pixels = np.count_nonzero(text_area)

    print(f"Modified detector result: {np.count_nonzero(mask)} pixels")
    print(f"Text pixels in mask: {text_pixels}")
    print(
        f"Text protection: {'PASS' if text_pixels < 100 else 'FAIL'}"
    )  # Allow some edge pixels

    return text_pixels < 100


if __name__ == "__main__":
    print("TESTING IMPROVED TEXT PROTECTION")
    print("=" * 50)

    # Test the improved protection concept
    improved_mask = test_improved_protection()

    # Modify the actual detector
    success = modify_watermark_detector()

    if success:
        # Test the modified detector
        test_passed = test_modified_detector()
        print(f"\nModified detector test: {'PASS' if test_passed else 'FAIL'}")
    else:
        print("Failed to modify detector")

    print("\n" + "=" * 50)
