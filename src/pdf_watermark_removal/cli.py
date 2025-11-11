"""Command-line interface for PDF watermark removal."""

import sys
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .pdf_processor import PDFProcessor
from .watermark_remover import WatermarkRemover
from .color_selector import ColorSelector
from .stats import ProcessingStats
from .i18n import set_language, t


console = Console()


def parse_pages(pages_str):
    """Parse page specification string.

    Args:
        pages_str: String like "1,3,5" or "1-5" or None

    Returns:
        List of page numbers (1-indexed) or None
    """
    if pages_str is None:
        return None

    pages = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                pages.extend(range(int(start), int(end) + 1))
            except ValueError:
                raise ValueError(f"Invalid page range: {part}")
        else:
            try:
                pages.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")

    return sorted(set(pages)) if pages else None


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
    "--inpaint-strength",
    default=1.0,
    type=float,
    help="Inpainting strength (0.5=light, 1.0=medium, 1.5=strong)",
)
@click.option(
    "--pages",
    default=None,
    type=str,
    help="Pages to process ('1,3,5' or '1-5'). All pages if not set.",
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
    help="Watermark color 'R,G,B' (e.g., '128,128,128'). Interactive if not set.",
)
@click.option(
    "--auto-color",
    is_flag=True,
    default=False,
    help="Skip interactive color selection, use automatic detection",
)
@click.option(
    "--protect-text",
    is_flag=True,
    default=True,
    help="Protect dark text from being removed",
)
@click.option(
    "--color-tolerance",
    default=30,
    type=int,
    help="Color matching tolerance (0-255, lower=stricter)",
)
@click.option(
    "--debug-mask",
    is_flag=True,
    default=False,
    help="Save debug preview of watermark detection",
)
@click.option(
    "--lang",
    default=None,
    type=str,
    help="Language (zh_CN, en_US). Auto-detect if not specified.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(
    input_pdf,
    output_pdf,
    kernel_size,
    inpaint_radius,
    inpaint_strength,
    pages,
    multi_pass,
    dpi,
    color,
    auto_color,
    protect_text,
    color_tolerance,
    debug_mask,
    lang,
    verbose,
):
    """Remove watermarks from PDF using Otsu threshold and inpaint."""
    try:
        # Set language
        if lang:
            set_language(lang)

        # Initialize stats
        stats = ProcessingStats(verbose=verbose)

        # Display header
        console.print(
            Panel(
                f"[bold cyan]{t('title')}[/bold cyan]\n"
                f"[yellow]Input:[/yellow]  {input_pdf}\n"
                f"[yellow]Output:[/yellow] {output_pdf}",
                title="[bold]Configuration[/bold]",
                border_style="cyan",
            )
        )

        if verbose:
            console.print("\n[bold blue]Verbose Mode Enabled[/bold blue]")

        pages_list = parse_pages(pages)

        # Parse color if provided
        watermark_color = parse_color(color) if color else None

        processor = PDFProcessor(dpi=dpi, verbose=verbose)

        # Interactive color selection for first page only
        if not auto_color and not watermark_color:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(f"[cyan]{t('loading_pdf')}...", total=None)
                first_page_images = processor.pdf_to_images(input_pdf, pages=[1])
                progress.stop_task(task)

            if first_page_images:
                selector = ColorSelector(verbose=verbose)
                watermark_color = selector.get_color_for_detection(
                    first_page_images[0], auto_detect=False
                )

        # Initialize remover with detected/selected color
        remover = WatermarkRemover(
            kernel_size=kernel_size,
            inpaint_radius=inpaint_radius,
            inpaint_strength=inpaint_strength,
            verbose=verbose,
            auto_detect_color=watermark_color is None,
            watermark_color=watermark_color,
            protect_text=protect_text,
            color_tolerance=color_tolerance,
        )

        # Convert all pages
        msg = "\n[bold]Step 1:[/bold] [yellow]Converting PDF to images...[/yellow]"
        console.print(msg)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Loading PDF", total=1)
            images = processor.pdf_to_images(input_pdf, pages=pages_list)
            progress.update(task, completed=1)

        page_info = (
            f"all {len(images)}" if not pages_list else f"{len(pages_list)} specified"
        )
        console.print(f"[green]Loaded {page_info} pages[/green]\n")

        # Debug mode: preview first page detection
        if debug_mask and images:
            console.print("[bold yellow]Debug Mode: Generating detection preview...[/bold yellow]")
            preview = remover.detector.preview_detection(
                images[0], output_path="debug_watermark_mask.png"
            )
            console.print("[green]✓ Saved debug preview to: debug_watermark_mask.png[/green]\n")

        # Set page dimensions for accurate statistics
        if images:
            page_height, page_width = images[0].shape[:2]
            stats.set_page_size(page_width, page_height)

        if verbose:
            if pages_list:
                console.print(
                    f"[blue]Processing {len(pages_list)} specified pages[/blue]"
                )
            else:
                console.print(f"[blue]Processing all {len(images)} pages[/blue]")

        console.print("[bold]Step 2:[/bold] [yellow]Removing watermarks...[/yellow]")

        # Process images with progress bar
        processed_images = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("-"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing pages", total=len(images))

            for i, img in enumerate(images):
                if multi_pass > 1:
                    processed = remover.remove_watermark_multi_pass(
                        img, passes=multi_pass
                    )
                else:
                    processed = remover.remove_watermark(img)
                processed_images.append(processed)
                progress.update(task, advance=1)

        console.print("[green]Watermark removal completed[/green]\n")

        console.print(
            "[bold]Step 3:[/bold] [yellow]Converting images back to PDF...[/yellow]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"[cyan]{t('saving_pdf')}", total=None)
            processor.images_to_pdf(processed_images, output_pdf)
            progress.stop_task(task)

        # Update stats
        stats.pages_processed = len(processed_images)
        if watermark_color:
            stats.set_watermark_color(watermark_color, coverage=100.0)
        output_size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        stats.set_output(output_pdf, output_size_mb)

        # Display statistics
        stats.display_summary(i18n_t=t)

    except FileNotFoundError as e:
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)
    except ImportError as e:
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback

            traceback.print_exc()
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
