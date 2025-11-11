"""Processing statistics and result feedback."""

import time
from datetime import timedelta

from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text


class ProcessingStats:
    """Track and display processing statistics."""

    def __init__(self, verbose=False):
        """Initialize statistics tracker.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.console = Console()
        self.start_time = time.time()
        self.pages_processed = 0
        self.watermark_color = None
        self.watermark_coverage = 0.0
        self.output_file = None
        self.output_size_mb = 0.0
        self.page_width = 2000  # Default A4 at 150 DPI: ~1654px
        self.page_height = 2825  # Default A4 at 150 DPI: ~2339px
        self.page_stats = []  # Track per-page statistics

    def set_watermark_color(self, color_rgb, coverage=0.0):
        """Set detected watermark color.

        Args:
            color_rgb: Tuple (R, G, B)
            coverage: Coverage percentage of total pixels
        """
        self.watermark_color = color_rgb
        self.watermark_coverage = coverage

    def add_page(self):
        """Increment processed pages counter."""
        self.pages_processed += 1

    def add_page_stat(self, page_num, watermark_coverage, status="success"):
        """Add per-page statistics.

        Args:
            page_num: Page number (1-indexed)
            watermark_coverage: Watermark coverage percentage (0-100)
            status: Processing status ('success', 'partial', 'failed')
        """
        self.page_stats.append(
            {
                "page": page_num,
                "coverage": watermark_coverage,
                "status": status,
            }
        )

    def set_output(self, output_file, file_size_mb):
        """Set output file information.

        Args:
            output_file: Path to output PDF
            file_size_mb: File size in MB
        """
        self.output_file = output_file
        self.output_size_mb = file_size_mb

    def set_page_size(self, width, height):
        """Set page dimensions for accurate pixel calculations.

        Args:
            width: Page width in pixels
            height: Page height in pixels
        """
        self.page_width = width
        self.page_height = height

    def get_elapsed_time(self):
        """Get formatted elapsed time.

        Returns:
            str: Formatted time (HH:MM:SS)
        """
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))

    def display_summary(self, i18n_t=None):
        """Display processing summary panel with detailed statistics.

        Args:
            i18n_t: Translation function
        """
        if i18n_t is None:

            def default_t(key, **kw):
                return key

            i18n_t = default_t

        # Calculate pixels removed (based on actual page size)
        total_pixels = self.page_width * self.page_height * self.pages_processed
        pixels_removed = int(self.watermark_coverage / 100 * total_pixels)

        # Create main summary table
        table = Table(show_header=False, padding=(0, 1))
        table.add_column("Label", style="cyan", width=25)
        table.add_column("Value", style="green")

        table.add_row(
            f"[bold]{i18n_t('pages_processed')}:[/bold]", f"{self.pages_processed}"
        )

        if self.watermark_color:
            table.add_row(
                f"[bold]{i18n_t('watermark_detection')}:[/bold]",
                f"RGB{self.watermark_color}",
            )

        table.add_row(
            f"[bold]{i18n_t('coverage')}:[/bold]", f"{self.watermark_coverage:.1f}%"
        )

        table.add_row(
            f"[bold]{i18n_t('pixels_removed')}:[/bold]", f"{pixels_removed:,}"
        )

        table.add_row(
            f"[bold]{i18n_t('time_elapsed')}:[/bold]", self.get_elapsed_time()
        )

        if self.output_file:
            table.add_row(
                f"[bold]{i18n_t('output_saved')}:[/bold]",
                f"{self.output_file} ({self.output_size_mb:.1f} MB)",
            )

        # Display main summary
        self.console.print(
            Panel(
                table,
                title="[bold green]Processing Complete[/bold green]",
                border_style="green",
            )
        )

        # Display per-page statistics if available
        if self.page_stats:
            self._display_page_statistics()

    def _display_page_statistics(self):
        """Display detailed per-page statistics in a table."""
        # Show summary of page statistics
        page_table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
        page_table.add_column("Page", style="cyan", width=6)
        page_table.add_column("Coverage %", style="blue", width=12)
        page_table.add_column("Status", style="yellow", width=10)

        success_count = 0
        partial_count = 0
        failed_count = 0
        skipped_count = 0

        for stat in self.page_stats:
            page_num = stat["page"]
            coverage = stat["coverage"]
            status = stat["status"]

            # Count statuses
            if status == "success":
                success_count += 1
                status_display = "[green]✓ Success[/green]"
            elif status == "partial":
                partial_count += 1
                status_display = "[yellow]⚠ Partial[/yellow]"
            elif status == "skipped":
                skipped_count += 1
                status_display = "[cyan]⊘ Skipped[/cyan]"
            else:
                failed_count += 1
                status_display = "[red]✗ Failed[/red]"

            page_table.add_row(f"{page_num}", f"{coverage:.1f}%", status_display)

        # Display page stats if we have a reasonable number
        if len(self.page_stats) <= 50:
            self.console.print("\n[bold]Per-Page Statistics:[/bold]")
            self.console.print(page_table)
        else:
            # For large PDFs, show summary only
            self.console.print("\n[bold]Processing Summary by Status:[/bold]")
            summary_table = Table(show_header=False, padding=(0, 1))
            summary_table.add_column("", style="cyan")
            summary_table.add_column("", style="green")
            summary_table.add_row("[green]✓ Successful[/green]", str(success_count))
            summary_table.add_row("[yellow]⚠ Partial[/yellow]", str(partial_count))
            summary_table.add_row("[cyan]⊘ Skipped[/cyan]", str(skipped_count))
            summary_table.add_row("[red]✗ Failed[/red]", str(failed_count))
            self.console.print(summary_table)


class ColorPreview:
    """Generate visual color previews."""

    @staticmethod
    def _rgb_to_hex(color_rgb):
        """Convert RGB tuple to hex color.

        Args:
            color_rgb: Tuple (R, G, B) or numpy uint8

        Returns:
            Tuple (hex_str, r, g, b) - hex color code and integer values
        """
        try:
            r = (
                int(color_rgb[0])
                if hasattr(color_rgb[0], "__int__")
                else int(color_rgb[0])
            )
            g = (
                int(color_rgb[1])
                if hasattr(color_rgb[1], "__int__")
                else int(color_rgb[1])
            )
            b = (
                int(color_rgb[2])
                if hasattr(color_rgb[2], "__int__")
                else int(color_rgb[2])
            )
        except (TypeError, ValueError, IndexError):
            r, g, b = 128, 128, 128

        return f"#{r:02x}{g:02x}{b:02x}", r, g, b

    @staticmethod
    def create_color_block(color_rgb, width=30):
        """Create a colored block using Unicode characters.

        Args:
            color_rgb: Tuple (R, G, B)
            width: Width in characters

        Returns:
            Rich Text object with colored block
        """
        hex_color, r, g, b = ColorPreview._rgb_to_hex(color_rgb)
        block_char = "█" * width

        try:
            # Use Style with bgcolor for consistent rendering
            return Text(block_char, style=Style(bgcolor=hex_color, color="black"))
        except Exception:
            # Fallback to simpler styling
            return Text(block_char, style="on white")

    @staticmethod
    def create_comparison(watermark_color):
        """Create a color comparison display with real colors.

        Args:
            watermark_color: Tuple (R, G, B)

        Returns:
            str: Rich-formatted comparison panel
        """
        hex_color, r, g, b = ColorPreview._rgb_to_hex(watermark_color)

        # Determine if color is dark or light for text contrast
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_color = "white" if luminance < 0.5 else "black"

        # Create color blocks
        watermark_block = Text("█" * 25, style=f"on {hex_color}")
        document_block = Text("█" * 25, style="on white")

        # Create sample text with contrast
        contrast_white = Text("On White Background", style=f"{text_color} on white")
        contrast_light = Text("Contrast Preview", style=f"{text_color} on #f0f0f0")

        return f"""
[bold cyan]Color Preview:[/bold cyan]

[bold]Hex Code:[/bold] {hex_color}
[bold]RGB:[/bold] RGB({r}, {g}, {b})

[bold]Document Background:[/bold]
{document_block}

[bold]Watermark Color:[/bold]
{watermark_block}

[bold]Text Contrast:[/bold]
{contrast_white}
{contrast_light}
"""

    @staticmethod
    def create_color_table(colors, i18n_t=None):
        """Create a table with real color previews and type info.

        Args:
            colors: List of color dicts with 'rgb', 'coverage', 'color_type' keys
            i18n_t: Translation function

        Returns:
            Rich Table object
        """
        if i18n_t is None:

            def default_t(key, **kw):
                return key

            i18n_t = default_t

        table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
        table.add_column("#", style="cyan", width=4)
        table.add_column("Preview", width=25)
        table.add_column("RGB", style="green", width=18)
        table.add_column("Coverage", style="blue", width=12)
        table.add_column("Type", style="yellow", width=12)

        for i, color_data in enumerate(colors[:10]):
            rgb = color_data.get("rgb", (128, 128, 128))
            coverage = color_data.get("coverage", 0.0)
            color_type = color_data.get("color_type", "unknown")

            # Safely convert to int
            try:
                r = int(rgb[0]) if hasattr(rgb[0], "__int__") else int(rgb[0])
                g = int(rgb[1]) if hasattr(rgb[1], "__int__") else int(rgb[1])
                b = int(rgb[2]) if hasattr(rgb[2], "__int__") else int(rgb[2])
            except (TypeError, ValueError, IndexError):
                r, g, b = 128, 128, 128

            # Create colored block
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            try:
                block = Text("  " * 10 + "  ", style=f"on {hex_color}")
            except Exception:
                block = Text("█" * 20)

            # Format type
            type_label = (
                color_type.value.upper()
                if hasattr(color_type, "value")
                else str(color_type).upper()
            )

            table.add_row(
                str(i), block, f"RGB({r},{g},{b})", f"{coverage:.1f}%", type_label
            )

        return table
