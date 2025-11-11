"""Command-line interface for PDF watermark removal."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.style import Style

from .pdf_processor import PDFProcessor
from .watermark_remover import WatermarkRemover
from .color_selector import ColorSelector


console = Console()


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
        # Display header
        console.print(Panel(
            "[bold cyan]PDF Watermark Removal Tool[/bold cyan]\n"
            f"[yellow]Input:[/yellow]  {input_pdf}\n"
            f"[yellow]Output:[/yellow] {output_pdf}",
            title="[bold]Configuration[/bold]",
            border_style="cyan"
        ))

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
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Loading first page...", total=None)
                first_page_images = processor.pdf_to_images(input_pdf, pages=[1])
                progress.stop_task(task)

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
        console.print("\n[bold]Step 1:[/bold] [yellow]Converting PDF to images...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Loading PDF", total=1)
            images = processor.pdf_to_images(input_pdf, pages=pages_list)
            progress.update(task, completed=1)

        page_info = f"all {len(images)}" if not pages_list else f"{len(pages_list)} specified"
        console.print(f"[green]Loaded {page_info} pages[/green]\n")

        if verbose:
            if pages_list:
                console.print(f"[blue]Processing {len(pages_list)} specified pages[/blue]")
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
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("[cyan]Processing pages", total=len(images))
            
            for i, img in enumerate(images):
                if multi_pass > 1:
                    processed = remover.remove_watermark_multi_pass(img, passes=multi_pass)
                else:
                    processed = remover.remove_watermark(img)
                processed_images.append(processed)
                progress.update(task, advance=1)

        console.print("[green]Watermark removal completed[/green]\n")

        console.print("[bold]Step 3:[/bold] [yellow]Converting images back to PDF...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Saving PDF", total=None)
            processor.images_to_pdf(processed_images, output_pdf)
            progress.stop_task(task)

        # Success message
        console.print(Panel(
            f"[green]Watermark removal completed successfully![/green]\n"
            f"[cyan]Output saved to:[/cyan] [bold yellow]{output_pdf}[/bold yellow]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        ))

    except FileNotFoundError as e:
        console.print(Panel(f"[red]{e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
        sys.exit(1)
    except ImportError as e:
        console.print(Panel(f"[red]{e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        console.print(Panel(f"[red]{e}[/red]", title="[bold red]Error[/bold red]", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
