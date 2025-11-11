"""Interactive CLI utilities for watermark color selection using rich."""

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from .color_analyzer import ColorAnalyzer


class ColorSelector:
    """Interactive color selection for watermark detection."""

    def __init__(self, verbose=False):
        """Initialize color selector.

        Args:
            verbose: Enable verbose logging
        """
        self.analyzer = ColorAnalyzer(verbose=verbose)
        self.verbose = verbose
        self.console = Console()

    def select_watermark_color_interactive(self, image_rgb, coarse=True):
        """Interactively select watermark color from image.

        Args:
            image_rgb: First page image for analysis
            coarse: If True, show 3 colors; if False, show 10 colors

        Returns:
            Tuple (R, G, B) of selected color or None
        """
        num_colors = 3 if coarse else 10

        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]WATERMARK COLOR DETECTION[/bold cyan]")
        self.console.print("=" * 60)

        # Analyze colors
        self.console.print(f"\n[yellow]Analyzing {num_colors} most common colors in the document...[/yellow]")
        colors = self.analyzer.get_dominant_colors(image_rgb, num_colors=num_colors)

        if not colors:
            self.console.print("[red]No colors detected. Using automatic detection.[/red]")
            return None

        # Display options
        self.console.print("\n[bold]Detected colors (likely watermark or text):[/bold]\n")
        self._display_colors_table(colors)

        # Get user input
        while True:
            try:
                choice = click.prompt(
                    "\nSelect color number (0-indexed) or 'a' for automatic",
                    type=str,
                    default='a'
                ).strip().lower()

                if choice == 'a' or choice == '':
                    self.console.print("[green]Using automatic color detection...[/green]")
                    return None

                choice_idx = int(choice)
                if 0 <= choice_idx < len(colors):
                    selected = colors[choice_idx]
                    
                    # Display selected color with preview
                    self._display_selected_color(selected)

                    # Ask if user wants finer control
                    if coarse and choice_idx > 0:
                        refine = click.confirm(
                            "\nShow more color options for finer selection?",
                            default=False
                        )
                        if refine:
                            return self.select_watermark_color_interactive(image_rgb, coarse=False)

                    return selected['rgb']
                else:
                    self.console.print(f"[red]Invalid choice. Please enter 0-{len(colors)-1} or 'a'[/red]")
            except ValueError:
                self.console.print("[red]Invalid input. Please enter a number or 'a'[/red]")

    def _display_colors_table(self, colors):
        """Display color options with rich table visualization.

        Args:
            colors: List of color dictionaries
        """
        table = Table(title="Color Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Index", style="cyan", width=8)
        table.add_column("Color Preview", width=30)
        table.add_column("RGB Value", style="green")
        table.add_column("Gray Level", style="yellow")
        table.add_column("Percentage", style="blue")

        for i, color_data in enumerate(colors):
            rgb = color_data['rgb']
            percentage = color_data['percentage']
            gray_val = color_data['gray']

            # Create visual bar using Unicode blocks
            bar_length = min(int(percentage / 2), 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Color the bar based on the actual color
            bar_style = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

            table.add_row(
                str(i),
                f"[{bar_style}]{bar}[/{bar_style}]",
                f"RGB{rgb}",
                str(gray_val),
                f"{percentage:.1f}%"
            )

        self.console.print(table)

    def _display_selected_color(self, color_data):
        """Display selected color with detailed information.

        Args:
            color_data: Selected color dictionary
        """
        rgb = color_data['rgb']
        percentage = color_data['percentage']
        gray_val = color_data['gray']

        # Create a colored panel for preview
        preview_text = "   " * 10  # Large colored block
        color_style = f"rgb({rgb[0]},{rgb[1]},{rgb[2]}) on rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        panel_content = f"""
[{color_style}]{preview_text}[/{color_style}]

RGB Value: [green]{rgb}[/green]
Gray Level: [yellow]{gray_val}[/yellow]
Percentage in document: [blue]{percentage:.2f}%[/blue]
        """

        self.console.print(Panel(panel_content, title="[bold]Selected Watermark Color[/bold]", border_style="green"))

    def get_color_for_detection(self, first_image_rgb, auto_detect=False):
        """Get watermark color for detection.

        Args:
            first_image_rgb: First page image
            auto_detect: If True, skip interactive selection

        Returns:
            Tuple (R, G, B) of watermark color or None for auto-detection
        """
        if auto_detect:
            return None

        try:
            use_interactive = click.confirm(
                "\nWould you like to interactively select the watermark color?",
                default=False
            )
            if not use_interactive:
                self.console.print("[green]Using automatic color detection...[/green]")
                return None
        except (EOFError, click.Abort):
            self.console.print("\n[green]Using automatic color detection...[/green]")
            return None

        # Ask for coarse vs fine selection
        try:
            use_coarse = click.confirm(
                "Use coarse color selection (3 main colors)?",
                default=True
            )
        except (EOFError, click.Abort):
            use_coarse = True

        return self.select_watermark_color_interactive(first_image_rgb, coarse=use_coarse)
