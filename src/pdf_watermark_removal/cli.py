"""Command-line interface for PDF watermark removal."""

import sys
from pathlib import Path

import click

from .pdf_processor import PDFProcessor
from .watermark_remover import WatermarkRemover
from .color_selector import ColorSelector


def parse_pages(pages_str):
    """Parse page specification string.

    Args:
        pages_str: String like "1,3,5" or "1-5" or None

    Returns:
        List of page numbers or None
    """
    if pages_str is None:
        return None

    pages = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))

    return sorted(set(pages))


def parse_color(color_str):
    """Parse color from string format 'R,G,B'.

    Args:
        color_str: Color string like "128,128,128"

    Returns:
        Tuple (R, G, B) or None
    """
    if not color_str:
        return None

    try:
        parts = [int(x.strip()) for x in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError("Color must have 3 components")
        if not all(0 <= p <= 255 for p in parts):
            raise ValueError("Color values must be 0-255")
        return tuple(parts)
    except (ValueError, AttributeError):
        return None


@click.command()
@click.argument("input_pdf", type=click.Path(exists=True))
@click.argument("output_pdf", type=click.Path())
@click.option(
    "--kernel-size",
    default=3,
    type=int,
    help="Morphological kernel size for watermark detection",
)
@click.option(
    "--inpaint-radius",
    default=2,
    type=int,
    help="Radius for inpainting algorithm",
)
@click.option(
    "--pages",
    default=None,
    type=str,
    help="Pages to process (e.g., '1,3,5' or '1-5'). Process all pages if not specified.",
)
@click.option(
    "--multi-pass",
    default=1,
    type=int,
    help="Number of removal passes",
)
@click.option(
    "--dpi",
    default=150,
    type=int,
    help="DPI for PDF to image conversion",
)
@click.option(
    "--color",
    default=None,
    type=str,
    help="Watermark color as 'R,G,B' (e.g., '128,128,128'). Interactive if not specified.",
)
@click.option(
    "--auto-color",
    is_flag=True,
    default=False,
    help="Skip interactive color selection, use automatic detection",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(input_pdf, output_pdf, kernel_size, inpaint_radius, pages, multi_pass, dpi, color, auto_color, verbose):
    """Remove watermarks from PDF using Otsu threshold and inpaint.

    INPUT_PDF: Path to input PDF file
    OUTPUT_PDF: Path to output PDF file
    
    By default, processes all pages. Use --pages to process specific pages.
    By default, offers interactive color selection. Use --auto-color to skip.
    """
    try:
        if verbose:
            click.echo("PDF Watermark Removal Tool")
            click.echo(f"Input: {input_pdf}")
            click.echo(f"Output: {output_pdf}")
            click.echo(f"Kernel size: {kernel_size}")
            click.echo(f"Inpaint radius: {inpaint_radius}")
            click.echo(f"Multi-pass: {multi_pass}")
            click.echo(f"DPI: {dpi}")
            click.echo()

        pages_list = parse_pages(pages)

        # Parse color if provided
        watermark_color = parse_color(color) if color else None

        processor = PDFProcessor(dpi=dpi, verbose=verbose)

        # Interactive color selection for first page only
        if not auto_color and not watermark_color:
            with click.progressbar(
                length=1, label="Loading first page", show_pos=False
            ) as bar:
                first_page_images = processor.pdf_to_images(input_pdf, pages=[1])
                bar.update(1)

            if first_page_images:
                selector = ColorSelector(verbose=verbose)
                watermark_color = selector.get_color_for_detection(
                    first_page_images[0],
                    auto_detect=False
                )

        # Initialize remover with detected/selected color
        remover = WatermarkRemover(
            kernel_size=kernel_size,
            inpaint_radius=inpaint_radius,
            verbose=verbose,
            auto_detect_color=watermark_color is None,
            watermark_color=watermark_color,
        )

        # Convert all pages
        if verbose:
            click.echo("Step 1: Converting PDF to images...")
        with click.progressbar(
            length=1, label="Loading PDF", show_pos=False
        ) as bar:
            images = processor.pdf_to_images(input_pdf, pages=pages_list)
            bar.update(1)

        if verbose:
            if pages_list:
                click.echo(f"Processing {len(pages_list)} specified pages")
            else:
                click.echo(f"Processing all {len(images)} pages")
            click.echo("Step 2: Removing watermarks...")

        # Process images with progress bar
        processed_images = []
        with click.progressbar(
            length=len(images), label="Removing watermarks", show_pos=True
        ) as bar:
            for i, img in enumerate(images):
                if multi_pass > 1:
                    processed = remover.remove_watermark_multi_pass(img, passes=multi_pass)
                else:
                    processed = remover.remove_watermark(img)
                processed_images.append(processed)
                bar.update(1)

        if verbose:
            click.echo("Step 3: Converting images back to PDF...")
        with click.progressbar(
            length=1, label="Saving PDF", show_pos=False
        ) as bar:
            processor.images_to_pdf(processed_images, output_pdf)
            bar.update(1)

        click.echo(f"\n✓ Watermark removal completed successfully!")
        click.echo(f"Output saved to: {output_pdf}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
