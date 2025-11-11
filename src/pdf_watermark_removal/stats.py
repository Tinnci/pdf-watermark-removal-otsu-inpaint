"""Processing statistics and result feedback."""

import time
from datetime import timedelta
from rich.console import Console
from rich.panel import Panel
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
    
    def set_output(self, output_file, file_size_mb):
        """Set output file information.
        
        Args:
            output_file: Path to output PDF
            file_size_mb: File size in MB
        """
        self.output_file = output_file
        self.output_size_mb = file_size_mb
    
    def get_elapsed_time(self):
        """Get formatted elapsed time.
        
        Returns:
            str: Formatted time (HH:MM:SS)
        """
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
    def display_summary(self, i18n_t=None):
        """Display processing summary panel.
        
        Args:
            i18n_t: Translation function
        """
        if i18n_t is None:
            i18n_t = lambda x, **kw: x
        
        # Calculate pixels removed (rough estimate)
        pixels_removed = int(self.watermark_coverage * 8000000)  # Typical page pixels
        
        # Create summary table
        table = Table(show_header=False, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row(
            f"[bold]{i18n_t('pages_processed')}:[/bold]",
            f"{self.pages_processed}"
        )
        
        if self.watermark_color:
            table.add_row(
                f"[bold]{i18n_t('watermark_detection')}:[/bold]",
                f"RGB{self.watermark_color}"
            )
        
        table.add_row(
            f"[bold]{i18n_t('coverage')}:[/bold]",
            f"{self.watermark_coverage:.1f}%"
        )
        
        table.add_row(
            f"[bold]{i18n_t('pixels_removed')}:[/bold]",
            f"{pixels_removed:,}"
        )
        
        table.add_row(
            f"[bold]{i18n_t('time_elapsed')}:[/bold]",
            self.get_elapsed_time()
        )
        
        if self.output_file:
            table.add_row(
                f"[bold]{i18n_t('output_saved')}:[/bold]",
                f"{self.output_file} ({self.output_size_mb:.1f} MB)"
            )
        
        # Display in panel
        self.console.print(Panel(
            table,
            title="[bold green]Processing Complete[/bold green]",
            border_style="green"
        ))


class ColorPreview:
    """Generate visual color previews."""
    
    @staticmethod
    def _rgb_to_hex(color_rgb):
        """Convert RGB tuple to hex color.
        
        Args:
            color_rgb: Tuple (R, G, B) or numpy uint8
            
        Returns:
            str: Hex color code (e.g., "#e9e9e9")
        """
        try:
            r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
        except (TypeError, ValueError):
            r, g, b = 128, 128, 128
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def create_color_block(color_rgb, width=30):
        """Create a colored block using Unicode characters.
        
        Args:
            color_rgb: Tuple (R, G, B)
            width: Width in characters
            
        Returns:
            Rich Text object with colored block
        """
        hex_color = ColorPreview._rgb_to_hex(color_rgb)
        block_char = "█" * width
        
        try:
            # Try to use RGB color directly
            return Text(block_char, style=f"on {hex_color}")
        except:
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
        hex_color = ColorPreview._rgb_to_hex(watermark_color)
        r, g, b = int(watermark_color[0]), int(watermark_color[1]), int(watermark_color[2])
        
        # Determine if color is dark or light for text contrast
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_color = "white" if luminance < 0.5 else "black"
        
        # Create color blocks
        watermark_block = Text("█" * 25, style=f"on {hex_color}")
        document_block = Text("█" * 25, style="on white")
        
        # Create sample text with contrast
        sample_text = Text("Sample Watermark", style=f"{hex_color}")
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
        """Create a table with real color previews.
        
        Args:
            colors: List of color dicts with 'rgb', 'coverage' keys
            i18n_t: Translation function
            
        Returns:
            Rich Table object
        """
        if i18n_t is None:
            i18n_t = lambda x, **kw: x
        
        table = Table(title="Color Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="cyan", width=8)
        table.add_column("Color Block", width=35)
        table.add_column("RGB Value", style="green")
        table.add_column("Coverage", style="blue")
        
        for i, color_data in enumerate(colors[:10]):
            rgb = color_data.get('rgb', (128, 128, 128))
            coverage = color_data.get('coverage', 0.0)
            
            # Create colored block
            hex_color = ColorPreview._rgb_to_hex(rgb)
            try:
                block = Text("█" * 20, style=f"on {hex_color}")
            except:
                block = Text("█" * 20)
            
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            table.add_row(
                str(i),
                block,
                f"({r}, {g}, {b})",
                f"{coverage:.1f}%"
            )
        
        return table

