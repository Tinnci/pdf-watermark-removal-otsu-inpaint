"""
Command-line interface for PDF watermark removal.

This module provides the main entry point for the PDF watermark removal tool,
handling argument parsing, configuration, and orchestrating the watermark
detection and removal process.
"""

import os
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .color_selector import ColorSelector
from .document_classifier import DocumentClassifier, get_optimal_parameters
from .i18n import set_language, t
from .pdf_processor import PDFProcessor
from .stats import ProcessingStats
from .watermark_remover import WatermarkRemover

console = Console()


def _check_ultralytics_available(verbose):
    """
    Checks if the `ultralytics` library is installed and available.

    Args:
        verbose (bool): If True, prints a success message when ultralytics is found.

    Returns:
        bool: True if ultralytics is installed, False otherwise. If not installed,
              it prints instructions for installation.
    """
    try:
        from ultralytics import YOLO  # noqa: F401

        if verbose:
            console.print("[dim]✓ Ultralytics available[/dim]")
        return True
    except ImportError:
        console.print(
            Panel(
                "[red]YOLO detection requires additional dependencies:[/red]\n\n"
                "[yellow]  pip install pdf-watermark-removal-otsu-inpaint[yolo][/yellow]\n\n"
                "[dim]Or install manually:[/dim]\n"
                "[yellow]  pip install ultralytics>=8.3.0[/yellow]",
                title="[bold red]✗ YOLO Not Available[/bold red]",
                border_style="red",
            )
        )
        return False


def _check_yolo_model_file(yolo_model):
    """
    Checks if the specified YOLO model file exists or is a known model name.
    If the model is not found locally and is not a known auto-downloadable model,
    it prints a warning message.

    Args:
        yolo_model (str): The path to the YOLO model file or a known model name.
    """
    model_path = Path(yolo_model)
    is_model_name = yolo_model in ("yolov8n-seg.pt", "yolo12n-seg.pt")

    if not is_model_name and not model_path.exists():
        console.print(
            Panel(
                f"[red]YOLO model file not found: {yolo_model}[/red]\n\n"
                "[dim]Default models will be auto-downloaded on first use:[/dim]\n"
                "[yellow]  yolov8n-seg.pt (6.2 MB)[/yellow]\n"
                "[yellow]  yolo12n-seg.pt (5.1 MB)[/yellow]\n\n"
                "[dim]Or provide a custom model path.[/dim]",
                title="[bold yellow]⚠ Model Will Be Auto-Downloaded[/bold yellow]",
                border_style="yellow",
            )
        )


def validate_yolo_setup(detection_method, yolo_model, verbose):
    """Validate YOLO dependencies and model availability.

    Args:
        detection_method: Detection method ('traditional' or 'yolo')
        yolo_model: Path to YOLO model file
        verbose: Verbose logging

    Raises:
        click.Abort: If YOLO is requested but not available or misconfigured
    """
    if detection_method != "yolo":
        return

    if not _check_ultralytics_available(verbose):
        raise click.Abort() from None

    _check_yolo_model_file(yolo_model)


def _parse_page_range(part):
    """
    Parses a page range string (e.g., '1-5') into a list of page numbers.

    Args:
        part (str): A string representing a page range (e.g., "1-5").

    Returns:
        list[int]: A list of 1-indexed page numbers within the specified range.

    Raises:
        ValueError: If the input string is not a valid page range format.
    """
    try:
        start, end = part.split("-")
        return list(range(int(start), int(end) + 1))
    except ValueError as err:
        raise ValueError(f"Invalid page range: {part}") from err


def _parse_single_page(part):
    """
    Parses a single page number string into an integer.

    Args:
        part (str): A string representing a single page number (e.g., "3").

    Returns:
        list[int]: A list containing the single 1-indexed page number.

    Raises:
        ValueError: If the input string is not a valid integer.
    """
    try:
        return [int(part)]
    except ValueError as err:
        raise ValueError(f"Invalid page number: {part}") from err


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
            pages.extend(_parse_page_range(part))
        else:
            pages.extend(_parse_single_page(part))

    return sorted(set(pages)) if pages else None


def parse_color(color_str):
    """
    Parses a color string in 'R,G,B' format into a tuple of integers.

    Args:
        color_str (str): A string representing an RGB color (e.g., "128,128,128").

    Returns:
        tuple[int, int, int] or None: A tuple (R, G, B) if the string is valid,
                                     otherwise None. Each component is an integer
                                     between 0 and 255.
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
@click.argument("input_pdf", type=click.Path(exists=True), required=False)
@click.argument("output_pdf", type=click.Path(), required=False)
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
    "--skip-errors",
    is_flag=True,
    default=False,
    help="Skip pages with errors instead of failing",
)
@click.option(
    "--show-strength",
    is_flag=True,
    default=False,
    help="Display strength parameters in progress feedback",
)
@click.option(
    "--no-auto-classify",
    is_flag=True,
    default=False,
    help="Disable automatic document type detection",
)
@click.option(
    "--detection-method",
    default="traditional",
    type=click.Choice(["traditional", "yolo"]),
    help="Detection method: 'traditional'=fast+lightweight (default), 'yolo'=accurate+slower (requires ultralytics)",
)
@click.option(
    "--yolo-model",
    default="yolov8n-seg.pt",
    type=click.Path(exists=False),
    help="YOLO model: 'yolov8n-seg.pt' (fast), 'yolov12n-seg.pt' (balanced), 'yolo11x-watermark.pt' (specialized), or custom path",
)
@click.option(
    "--yolo-conf",
    default=0.25,
    type=float,
    help="YOLO confidence threshold 0-1 (lower=more detections). For YOLO detection only.",
)
@click.option(
    "--yolo-device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device for YOLO inference",
)
@click.option(
    "--yolo-version",
    default="v8",
    type=click.Choice(["v8", "v12", "v11"]),
    help="YOLO version for detection (v8=fast, v12=accurate, v11=specialized watermark detection)",
)
@click.option(
    "--preset",
    default=None,
    type=click.Choice(["electronic-color"]),
    help="Preset mode: 'electronic-color' for precise color removal on electronic documents (requires --color)",
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
@click.option(
    "--list-models",
    is_flag=True,
    help="List available YOLO models and exit",
)
@click.option(
    "--detect-qr-codes",
    is_flag=True,
    default=False,
    help="Enable QR code detection and removal",
)
@click.option(
    "--qr-detection-method",
    default="opencv",
    type=click.Choice(["opencv", "pyzbar"]),
    help="QR code detection method: 'opencv'=built-in (default), 'pyzbar'=alternative library",
)
@click.option(
    "--remove-all-qr-codes",
    is_flag=True,
    default=False,
    help="Remove all detected QR codes (treat as watermarks)",
)
@click.option(
    "--qr-categories-to-remove",
    default=None,
    type=str,
    help="Comma-separated list of QR code categories to remove (e.g., 'advertisement,unknown')",
)
@click.option(
    "--qr-preset",
    default=None,
    type=click.Choice(["aggressive", "conservative", "ads_only", "interactive"]),
    help="QR code removal preset: 'aggressive'=remove all, 'conservative'=remove ads+unknown, 'ads_only'=remove only ads, 'interactive'=prompt user",
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
    skip_errors,
    show_strength,
    no_auto_classify,
    detection_method,
    yolo_model,
    yolo_conf,
    yolo_device,
    yolo_version,
    preset,
    lang,
    verbose,
    list_models,
    detect_qr_codes,
    qr_detection_method,
    remove_all_qr_codes,
    qr_categories_to_remove,
    qr_preset,
):
    """Remove watermarks from PDF using Otsu threshold and inpaint."""
    try:
        # --- Preset Mode Validation ---
        # Checks if the 'electronic-color' preset is used and ensures that
        # the '--color' option is also provided, as it's a requirement for this preset.
        # It also forces the detection method to 'traditional' if the preset is active.
        if preset == "electronic-color":
            if not color:
                console.print(
                    Panel(
                        "[red]Preset 'electronic-color' requires --color to be specified.[/red]\n\n"
                        "[yellow]Example:[/yellow]\n"
                        "  pdf-watermark-removal input.pdf output.pdf --preset electronic-color --color '200,200,200'\n\n"
                        "[dim]This preset is optimized for precise watermark color removal on electronic documents.[/dim]",
                        title="[bold red]Missing Required Parameter[/bold red]",
                        border_style="red",
                    )
                )
                raise click.Abort()

            # Force traditional detection method for color-based preset
            if detection_method != "traditional":
                if verbose:
                    console.print(
                        "[yellow]⚠ Preset 'electronic-color' requires traditional detection method. Switching...[/yellow]"
                    )
                detection_method = "traditional"

        # --- Model Listing ---
        # If the --list-models flag is set, it lists available YOLO models and exits.
        if list_models:
            from .model_manager import ModelManager

            manager = ModelManager(verbose=True)
            manager.list_available_models()
            return

        # --- Argument Validation and Setup ---
        # Ensures that input and output PDF paths are provided, unless only listing models.
        # Sets the language for internationalization if specified.
        # Validates the YOLO setup (dependencies and model file) if YOLO detection is enabled.
        if not input_pdf or not output_pdf:
            console.print(
                "[red]Error: INPUT_PDF and OUTPUT_PDF are required unless using --list-models[/red]"
            )
            raise click.MissingParameter("input_pdf or output_pdf")

        # Set language
        if lang:
            set_language(lang)

        # Validate YOLO setup before processing
        validate_yolo_setup(detection_method, yolo_model, verbose)

        # Initialize stats
        stats = ProcessingStats(verbose=verbose)

        # --- Configuration Display ---
        # Constructs and prints a panel summarizing the processing configuration,
        # including input/output files and detection method details.
        config_text = f"[bold cyan]{t('title')}[/bold cyan]\n"
        config_text += f"[yellow]Input:[/yellow]  {input_pdf}\n"
        config_text += f"[yellow]Output:[/yellow] {output_pdf}"

        # Add detection method info
        if detection_method == "yolo":
            version_display = f"YOLO{yolo_version.upper()}-seg"
            if yolo_version == "v12":
                accuracy = "Higher accuracy"
                actual_model = "yolov12n-seg.pt"  # Fixed: yolov12 not yolo12
            elif yolo_version == "v11":
                accuracy = "Specialized watermark detection"
                actual_model = "yolo11x-watermark.pt"
            else:
                accuracy = "Fast baseline"
                actual_model = "yolov8n-seg.pt"

            # Use actual model if default was provided
            display_model = (
                actual_model if yolo_model == "yolov8n-seg.pt" else yolo_model
            )

            config_text += (
                f"\n[yellow]Detection:[/yellow] {version_display} ({accuracy})"
            )
            config_text += (
                f"\n[dim]  Model: {display_model} | Confidence: {yolo_conf} | "
                f"Device: {yolo_device}[/dim]"
            )
        else:
            config_text += (
                "\n[yellow]Detection:[/yellow] Traditional CV (fast, no dependencies)"
            )

        console.print(
            Panel(
                config_text,
                title="[bold]Configuration[/bold]",
                border_style="cyan",
            )
        )

        if verbose:
            console.print("\n[bold blue]Verbose Mode Enabled[/bold blue]")

        # --- Argument Parsing ---
        # Parses the specified page range, watermark color, and QR code categories
        # from the command-line arguments.
        pages_list = parse_pages(pages)

        # Parse color if provided
        watermark_color = parse_color(color) if color else None

        # Parse QR code categories to remove
        qr_categories_list = None
        if qr_categories_to_remove:
            qr_categories_list = [
                cat.strip() for cat in qr_categories_to_remove.split(",")
            ]

        # Auto-detect and prompt for QR code scanning will happen after processor is created

        # --- QR Code Preset Handling ---
        # Applies predefined settings for QR code removal based on the chosen preset
        # (e.g., 'aggressive', 'conservative', 'ads_only').
        if qr_preset and detect_qr_codes:
            if qr_preset == "aggressive":
                remove_all_qr_codes = True
            elif qr_preset == "conservative":
                qr_categories_list = ["advertisement", "unknown"]
            elif qr_preset == "ads_only":
                qr_categories_list = ["advertisement"]
            # "interactive" will be handled later

        processor = PDFProcessor(dpi=dpi, verbose=verbose)

        # --- Automatic QR Code Detection and User Prompt ---
        # If QR code detection is not explicitly enabled, this block attempts to
        # detect QR codes on the first page. If found, it prompts the user to
        # enable QR code scanning and removal for the current session.
        if not detect_qr_codes:
            # Quick QR code detection on first page to see if there might be QR codes
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Analyzing document for QR codes...", total=None
                )
                first_page_for_scan = processor.pdf_to_images(input_pdf, pages=[1])
                progress.stop_task(task)

            if first_page_for_scan:
                from .qr_detector import QRCodeDetector

                temp_detector = QRCodeDetector(method="opencv", verbose=False)
                potential_qr_codes = temp_detector.detect_qr_codes(
                    first_page_for_scan[0]
                )

                if potential_qr_codes:
                    # Found potential QR codes - ask user if they want to enable detection
                    try:
                        enable_qr = click.confirm(
                            f"\n[Auto-detected] Found {len(potential_qr_codes)} potential QR code(s) in document. "
                            "Enable QR code scanning and removal?",
                            default=True,
                        )
                        if enable_qr:
                            detect_qr_codes = True
                            # Store the loaded image for later use
                            first_page_images = first_page_for_scan
                            console.print(
                                "[green][OK] QR code detection enabled for this session![/green]"
                            )
                        else:
                            console.print(
                                "[dim]QR code detection skipped. Run with --detect-qr-codes to enable later.[/dim]"
                            )
                    except (EOFError, click.Abort):
                        pass

        # --- Interactive Color Selection ---
        # If using the traditional detection method and no color is specified
        # (neither via --color nor --auto-color), this block initiates an
        # interactive color selection process using the first page of the PDF.
        # It also handles the activation of an interactive 'electronic-color' preset.
        use_interactive_preset = False
        # Note: first_page_images might already be loaded from QR auto-detection

        if detection_method == "traditional" and not auto_color and not watermark_color:
            # Only load first page if not already loaded for QR detection
            if first_page_images is None:
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
                color_result = selector.get_color_for_detection(
                    first_page_images[0], auto_detect=False
                )

                # Check if user chose preset mode interactively
                if isinstance(color_result, dict) and color_result.get("use_preset"):
                    watermark_color = color_result["color"]
                    use_interactive_preset = True
                    console.print(
                        "[bold green]✓ Electronic-color preset activated![/bold green]"
                    )
                else:
                    watermark_color = color_result

        # Load first page for QR code detection if needed (and not already loaded)
        if detect_qr_codes and qr_preset == "interactive" and first_page_images is None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Loading first page for QR analysis...", total=None
                )
                first_page_images = processor.pdf_to_images(input_pdf, pages=[1])
                progress.stop_task(task)

        # --- Interactive QR Code Selection ---
        # If QR code detection is enabled and the 'interactive' QR preset is chosen,
        # this block allows the user to interactively select which detected QR codes
        # should be removed from the document.
        if detect_qr_codes and qr_preset == "interactive" and first_page_images:
            from .qr_detector import QRCodeDetector
            from .qr_selector import QRCodeSelector

            # Create temporary detector for interactive selection
            temp_qr_detector = QRCodeDetector(
                method=qr_detection_method, verbose=verbose
            )
            qr_codes = temp_qr_detector.detect_qr_codes(first_page_images[0])

            if qr_codes:
                qr_selector = QRCodeSelector(verbose=verbose)
                codes_to_remove = qr_selector.select_qr_codes_to_remove(
                    qr_codes, first_page_images[0]
                )

                if codes_to_remove:
                    # Update the categories to remove based on user selection
                    selected_categories = {qr.category for qr in codes_to_remove}
                    qr_categories_list = list(selected_categories)
                    console.print(
                        f"[bold green]✓ QR code selection complete: {len(codes_to_remove)} codes selected for removal[/bold green]"
                    )
                else:
                    console.print("[dim]No QR codes selected for removal[/dim]")
            else:
                console.print("[dim]No QR codes detected in first page[/dim]")

        # --- Initialize Watermark Remover ---
        # Sets up the WatermarkRemover instance with all the configured parameters,
        # including kernel size, inpainting settings, color detection, text protection,
        # YOLO parameters (if applicable), and QR code detection/removal settings.
        remover = WatermarkRemover(
            kernel_size=kernel_size,
            inpaint_radius=inpaint_radius,
            inpaint_strength=inpaint_strength,
            verbose=verbose,
            auto_detect_color=watermark_color is None,
            watermark_color=watermark_color,
            protect_text=protect_text,
            color_tolerance=color_tolerance,
            # YOLO parameters
            detection_method=detection_method,
            yolo_model_path=yolo_model,
            yolo_conf_thres=yolo_conf,
            yolo_device=yolo_device,
            yolo_version=yolo_version,
            auto_download_model=True,
            # QR code parameters
            detect_qr_codes=detect_qr_codes,
            qr_detection_method=qr_detection_method,
            remove_all_qr_codes=remove_all_qr_codes,
            qr_code_categories_to_remove=qr_categories_list,
        )

        # --- Display Strength Configuration ---
        # If requested via the --show-strength flag, this block retrieves and
        # displays detailed information about the inpainting strength parameters.
        if show_strength:
            strength_info = remover.get_strength_info()
            strength_table = Panel(
                f"[cyan]Inpaint Strength:[/cyan] [green]{strength_info['strength']:.1f}[/green]\n"
                f"[cyan]Blend Mode:[/cyan] [green]{strength_info['blend_mode']}[/green]\n"
                f"[cyan]Base Radius:[/cyan] [green]{inpaint_radius}[/green]\n"
                f"[dim]Note: Dynamic radius will be calculated per-page based on watermark coverage[/dim]",
                title="[bold]Strength Configuration[/bold]",
                border_style="cyan",
            )
            console.print(strength_table)

        # --- Step 1: Convert PDF to Images ---
        # Converts the input PDF document into a series of images, one for each page.
        # This step is crucial as watermark removal operations are performed on images.
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

        # --- Auto-classify Document Type and Optimize Parameters ---
        # This section automatically detects the document type (e.g., electronic, scanned)
        # and suggests or applies optimal processing parameters based on the classification.
        # It also handles the auto-suggestion of the 'electronic-color' preset.
        auto_classify = not no_auto_classify  # Invert the flag
        classification = None

        # Auto-detect and prompt for electronic-color preset for suitable documents
        auto_preset_suggested = False
        if auto_classify and images and not preset and not use_interactive_preset:
            classifier = DocumentClassifier(verbose=verbose)
            classification = classifier.classify(images[0])

            # Suggest electronic-color preset for electronic documents with high confidence
            if (
                classification.doc_type.value == "electronic"
                and classification.confidence >= 75
            ):
                try:
                    suggest_preset = click.confirm(
                        "\nElectronic document detected with high confidence. "
                        "Use 'electronic-color' preset mode for precise watermark removal?",
                        default=True,
                    )
                    if suggest_preset:
                        preset = "electronic-color"
                        auto_preset_suggested = True
                        console.print(
                            "[green]✓ Electronic-color preset activated for this session![/green]"
                        )
                except (EOFError, click.Abort):
                    pass

        # Apply preset parameters if specified via CLI flag OR interactively chosen
        if preset == "electronic-color" or use_interactive_preset:
            preset_params = get_optimal_parameters(None, preset_mode="electronic-color")

            console.print(
                "[dim]🎯 Preset mode: ELECTRONIC-COLOR "
                "(precise color removal) → Optimized[/dim]"
            )

            # Apply preset parameters (these override defaults but not user-specified values)
            applied_params = []

            if color_tolerance == 30:  # Default value
                color_tolerance = preset_params["color_tolerance"]
                applied_params.append(f"color_tolerance={color_tolerance}")

            if inpaint_strength == 1.0:  # Default value
                inpaint_strength = preset_params["inpaint_strength"]
                applied_params.append(f"inpaint_strength={inpaint_strength}")

            if kernel_size == 3:  # Default value
                kernel_size = preset_params["kernel_size"]
                applied_params.append(f"kernel_size={kernel_size}")

            if multi_pass == 1:  # Default value
                multi_pass = preset_params["multi_pass"]
                applied_params.append(f"multi_pass={multi_pass}")

            if dpi == 150:  # Default value
                dpi = preset_params["dpi"]
                applied_params.append(f"dpi={dpi}")

            if applied_params and verbose:
                console.print(f"[dim]   └─ Applied: {', '.join(applied_params)}[/dim]")

            # Disable auto-classify when using preset (unless it was auto-suggested)
            if not auto_preset_suggested:
                auto_classify = False

        if auto_classify and images and not auto_preset_suggested:
            classifier = DocumentClassifier(verbose=verbose)
            classification = classifier.classify(images[0])
            auto_params = get_optimal_parameters(classification.doc_type)

            # Compact one-line summary for visibility without blocking flow
            console.print(
                f"[dim]📊 Document analysis: {classification.doc_type.value.upper()} "
                f"({classification.confidence:.0f}% confidence) "
                f"→ Auto-optimized[/dim]"
            )

            # Apply auto parameters only if user didn't override (check default values)
            applied_params = []

            if color_tolerance == 30:  # Default value
                color_tolerance = auto_params["color_tolerance"]
                applied_params.append(f"color_tolerance={color_tolerance}")

            if inpaint_strength == 1.0:  # Default value
                inpaint_strength = auto_params["inpaint_strength"]
                applied_params.append(f"inpaint_strength={inpaint_strength}")

            if kernel_size == 3:  # Default value
                kernel_size = auto_params["kernel_size"]
                applied_params.append(f"kernel_size={kernel_size}")

            if multi_pass == 1:  # Default value
                multi_pass = auto_params["multi_pass"]
                applied_params.append(f"multi_pass={multi_pass}")

            if dpi == 150:  # Default value
                dpi = auto_params["dpi"]
                applied_params.append(f"dpi={dpi}")

            if applied_params and verbose:
                console.print(f"[dim]   └─ Applied: {', '.join(applied_params)}[/dim]")

        # --- Debug Mode: Detection Preview ---
        # If the --debug-mask flag is enabled, this block generates and saves
        # a preview image of the watermark detection mask for the first page.
        if debug_mask and images:
            console.print(
                "[bold yellow]Debug Mode: Generating detection preview...[/bold yellow]"
            )
            remover.detector.preview_detection(
                images[0], output_path="debug_watermark_mask.png"
            )
            console.print(
                "[green]✓ Saved debug preview to: debug_watermark_mask.png[/green]\n"
            )

        # --- Set Page Dimensions for Statistics ---
        # If images were successfully loaded, this block sets the dimensions
        # of the first page in the statistics tracker for accurate calculations.
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

        # --- Step 2: Watermark Removal Processing Loop ---
        # Iterates through each page image, applies the watermark detection and
        # removal algorithms, and tracks progress and statistics.
        # Handles multi-pass removal, debug information, and error skipping.
        processed_images = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("-"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task("[cyan]Overall Progress", total=len(images))

            for page_idx, img in enumerate(images):
                page_num = page_idx + 1
                page_task = progress.add_task(
                    f"[yellow]Page {page_num}/{len(images)}", total=100
                )

                try:
                    # Watermark detection and removal
                    progress.update(
                        page_task, description=f"[yellow]Page {page_num}: Processing..."
                    )
                    progress.update(page_task, completed=0)

                    if multi_pass > 1:
                        processed = remover.remove_watermark_multi_pass(
                            img, passes=multi_pass
                        )
                    else:
                        processed = remover.remove_watermark(img)

                    progress.update(page_task, completed=100)

                    # Build completion message with optional strength details
                    status_msg = f"[green]✓ Page {page_num}"
                    if show_strength:
                        stats_info = remover.last_stats
                        status_msg += (
                            f" [dim]| cov:{stats_info['coverage']:.1f}% "
                            f"| str:{stats_info['strength']:.1f} "
                            f"| rad:{stats_info['dynamic_radius']}[/dim]"
                        )

                    # Add QR code removal feedback
                    if detect_qr_codes:
                        qr_summary = remover.detector.get_qr_removal_summary()
                        if qr_summary and qr_summary["to_remove"] > 0:
                            status_msg += f" [cyan]| QR:{qr_summary['to_remove']}"
                            if qr_summary.get("categories"):
                                cats = ", ".join(
                                    f"{k}={v}"
                                    for k, v in qr_summary["categories"].items()
                                )
                                status_msg += f" ({cats})"
                            status_msg += "[/cyan]"

                    status_msg += "[/green]"

                    progress.update(page_task, description=status_msg)
                    processed_images.append(processed)

                    # Record page statistics
                    mask = remover.detector.detect_watermark_mask(img)
                    coverage = (
                        np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100
                    )
                    stats.add_page_stat(page_num, coverage, status="success")

                    # Track QR code statistics if enabled
                    if detect_qr_codes:
                        qr_codes = remover.detector.get_detected_qr_codes()
                        if qr_codes:
                            stats.add_qr_detection(qr_codes)

                except Exception as e:
                    error_msg = f"[red]Page {page_num}: {str(e)[:50]}[/red]"
                    progress.update(page_task, description=error_msg)

                    if skip_errors:
                        console.print(
                            f"[yellow]⚠ Skipped page {page_num}: {str(e)[:80]}[/yellow]"
                        )
                        processed_images.append(img)  # Keep original
                        stats.add_page_stat(page_num, 0.0, status="skipped")
                    else:
                        if verbose:
                            console.print(
                                f"[red]Error processing page {page_num}: {e}[/red]"
                            )
                        raise

                finally:
                    progress.update(main_task, advance=1)
                    progress.remove_task(page_task)

        console.print("[green]Watermark removal completed[/green]\n")

        # --- Step 3: Convert Images Back to PDF ---
        # Takes the processed images and combines them into a new PDF document,
        # which is then saved to the specified output path.
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

        # --- Update and Display Statistics ---
        # Finalizes the processing statistics, including output file size,
        # and then displays a summary of the entire operation.
        # This also includes a detailed summary of QR code detection and removal.
        stats.pages_processed = len(processed_images)
        if watermark_color:
            stats.set_watermark_color(watermark_color, coverage=100.0)
        output_size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        stats.set_output(output_pdf, output_size_mb)

        # Display QR code summary if any were processed
        if detect_qr_codes:
            qr_codes = remover.detector.get_detected_qr_codes()
            if qr_codes:
                console.print("\n[bold cyan]QR Code Detection Summary:[/bold cyan]")

                # Determine which codes were removed
                if remove_all_qr_codes:
                    codes_to_remove = qr_codes
                elif qr_categories_list:
                    codes_to_remove = [
                        qr for qr in qr_codes if qr.category in qr_categories_list
                    ]
                else:
                    codes_to_remove = [
                        qr
                        for qr in qr_codes
                        if qr.category in ["advertisement", "unknown"]
                    ]

                # Build summary
                categories_found = {}
                categories_removed = {}
                for qr in qr_codes:
                    categories_found[qr.category] = (
                        categories_found.get(qr.category, 0) + 1
                    )

                for qr in codes_to_remove:
                    categories_removed[qr.category] = (
                        categories_removed.get(qr.category, 0) + 1
                    )

                # Display summary
                console.print(
                    f"  [cyan]Detected:[/cyan] {len(qr_codes)} total | "
                    f"[green]Removed:[/green] {len(codes_to_remove)}"
                )

                if categories_found:
                    found_str = ", ".join(
                        f"{cat}:{count}" for cat, count in categories_found.items()
                    )
                    console.print(f"  [dim]Found categories:[/dim] {found_str}")

                if categories_removed:
                    removed_str = ", ".join(
                        f"{cat}:{count}" for cat, count in categories_removed.items()
                    )
                    console.print(f"  [dim]Removed categories:[/dim] {removed_str}\n")

        # Set final QR code statistics
        if detect_qr_codes:
            # Aggregate QR code stats from all pages
            total_detected = 0
            total_removed = 0
            categories_combined = {}

            # Get QR codes from the detector (last processed page)
            qr_codes = remover.detector.get_detected_qr_codes()
            if qr_codes:
                total_detected = len(qr_codes)
                # Count removed QR codes based on categories
                if remove_all_qr_codes:
                    total_removed = total_detected
                elif qr_categories_list:
                    total_removed = len(
                        [qr for qr in qr_codes if qr.category in qr_categories_list]
                    )
                else:
                    # Default conservative removal
                    total_removed = len(
                        [
                            qr
                            for qr in qr_codes
                            if qr.category in ["advertisement", "unknown"]
                        ]
                    )

                # Count categories
                for qr in qr_codes:
                    category = qr.category
                    categories_combined[category] = (
                        categories_combined.get(category, 0) + 1
                    )

            stats.set_qr_stats(total_detected, total_removed, categories_combined)

        # Display statistics
        stats.display_summary(i18n_t=t)

    # --- Error Handling ---
    # Catches various exceptions that might occur during the process,
    # prints an informative error message to the console, and exits with a non-zero status code.
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
