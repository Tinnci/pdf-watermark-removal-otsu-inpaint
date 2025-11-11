#!/usr/bin/env python3
"""Example usage of the PDF watermark removal library."""

from pdf_watermark_removal.pdf_processor import PDFProcessor
from pdf_watermark_removal.watermark_remover import WatermarkRemover


def example_basic_usage(input_pdf, output_pdf):
    """Basic example of watermark removal."""
    print("Example: Basic watermark removal")
    print(f"Input: {input_pdf}")
    print(f"Output: {output_pdf}")

    processor = PDFProcessor(verbose=True)
    remover = WatermarkRemover(verbose=True)

    print("\n1. Converting PDF to images...")
    images = processor.pdf_to_images(input_pdf)

    print(f"\n2. Removing watermarks from {len(images)} pages...")
    processed_images = []
    for i, img in enumerate(images):
        print(f"  Page {i + 1}/{len(images)}")
        processed = remover.remove_watermark(img)
        processed_images.append(processed)

    print("\n3. Converting images back to PDF...")
    processor.images_to_pdf(processed_images, output_pdf)

    print(f"\n✓ Done! Output saved to: {output_pdf}")


def example_selective_pages(input_pdf, output_pdf, pages):
    """Example of processing only specific pages."""
    print(f"Example: Removing watermarks from pages {pages}")

    processor = PDFProcessor(verbose=True)
    remover = WatermarkRemover(verbose=True)

    images = processor.pdf_to_images(input_pdf, pages=pages)
    processed_images = [remover.remove_watermark(img) for img in images]
    processor.images_to_pdf(processed_images, output_pdf)

    print(f"✓ Done! Output saved to: {output_pdf}")


def example_multi_pass_removal(input_pdf, output_pdf, passes=2):
    """Example of multi-pass removal for stubborn watermarks."""
    print(f"Example: Multi-pass watermark removal ({passes} passes)")

    processor = PDFProcessor(verbose=True)
    remover = WatermarkRemover(verbose=True)

    images = processor.pdf_to_images(input_pdf)
    processed_images = [
        remover.remove_watermark_multi_pass(img, passes=passes) for img in images
    ]
    processor.images_to_pdf(processed_images, output_pdf)

    print(f"✓ Done! Output saved to: {output_pdf}")


def example_custom_parameters(input_pdf, output_pdf, kernel_size=5, inpaint_radius=3):
    """Example with custom parameters."""
    print(f"Example: Custom parameters (kernel_size={kernel_size}, inpaint_radius={inpaint_radius})")

    processor = PDFProcessor(dpi=300, verbose=True)
    remover = WatermarkRemover(
        kernel_size=kernel_size,
        inpaint_radius=inpaint_radius,
        verbose=True,
    )

    images = processor.pdf_to_images(input_pdf)
    processed_images = [remover.remove_watermark(img) for img in images]
    processor.images_to_pdf(processed_images, output_pdf)

    print(f"✓ Done! Output saved to: {output_pdf}")


if __name__ == "__main__":
    print("PDF Watermark Removal - Examples\n")
    print("These examples show how to use the library programmatically.")
    print("Update the file paths and run the examples as needed.\n")

    import sys

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        print(f"Using input: {input_file}")
        print(f"Using output: {output_file}\n")

        try:
            example_basic_usage(input_file, output_file)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python examples.py <input_pdf> <output_pdf>")
        print("\nExample:")
        print("  python examples.py input.pdf output.pdf")
