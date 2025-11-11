"""Processing statistics and result feedback."""

import time
from datetime import timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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
    def create_preview(color_rgb, width=50, height=8):
        """Create a colored preview box.
        
        Args:
            color_rgb: Tuple (R, G, B)
            width: Width in characters
            height: Height in lines
            
        Returns:
            str: Rich-formatted color box
        """
        # Create color box with background
        rgb = color_rgb
        if isinstance(rgb, tuple) and len(rgb) == 3:
            # Handle numpy uint8
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        else:
            r, g, b = rgb
        
        # Build the box
        line = " " * width
        return f"[on rgb({r},{g},{b})]{line}[/on rgb({r},{g},{b})]"
    
    @staticmethod
    def create_comparison(watermark_color, contrast_level="High"):
        """Create a color comparison display.
        
        Args:
            watermark_color: Tuple (R, G, B)
            contrast_level: Contrast level string
            
        Returns:
            str: Rich-formatted comparison
        """
        comparison = f"""
[bold cyan]Color Preview:[/bold cyan]

Document Background:
{ColorPreview.create_preview((255, 255, 255))}

Watermark Color:
{ColorPreview.create_preview(watermark_color)}

Contrast: [bold green]{contrast_level}[/bold green]
"""
        return comparison
