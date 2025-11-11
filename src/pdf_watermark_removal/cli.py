"""Command-line interface for PDF watermark removal."""

import sys
from pathlib import Path

import click

from .pdf_processor import PDFProcessor
from .watermark_remover import WatermarkRemover


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
    help="Pages to process (e.g., '1,3,5' or '1-5')",
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
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(input_pdf, output_pdf, kernel_size, inpaint_radius, pages, multi_pass, dpi, verbose):
    """Remove watermarks from PDF using Otsu threshold and inpaint.

    INPUT_PDF: Path to input PDF file
    OUTPUT_PDF: Path to output PDF file
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

        processor = PDFProcessor(dpi=dpi, verbose=verbose)
        remover = WatermarkRemover(
            kernel_size=kernel_size,
            inpaint_radius=inpaint_radius,
            verbose=verbose,
        )

        with click.progressbar(
            length=100, label="Processing", show_pos=True
        ) as bar:
            bar.update(10)

            if verbose:
                click.echo("Step 1: Converting PDF to images...")
            images = processor.pdf_to_images(input_pdf, pages=pages_list)
            bar.update(30)

            if verbose:
                click.echo("Step 2: Removing watermarks...")
            processed_images = []
            for i, img in enumerate(images):
                if verbose:
                    click.echo(f"  Processing image {i + 1}/{len(images)}...")
                if multi_pass > 1:
                    processed = remover.remove_watermark_multi_pass(img, passes=multi_pass)
                else:
                    processed = remover.remove_watermark(img)
                processed_images.append(processed)

            bar.update(50)

            if verbose:
                click.echo("Step 3: Converting images back to PDF...")
            processor.images_to_pdf(processed_images, output_pdf)
            bar.update(100)

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
