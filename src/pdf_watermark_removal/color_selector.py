"""Interactive CLI utilities for watermark color selection."""

import click
import numpy as np
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

    def select_watermark_color_interactive(self, image_rgb, coarse=True):
        """Interactively select watermark color from image.

        Args:
            image_rgb: First page image for analysis
            coarse: If True, show 3 colors; if False, show 10 colors

        Returns:
            Tuple (R, G, B) of selected color or None
        """
        num_colors = 3 if coarse else 10

        click.echo("\n" + "=" * 60)
        click.echo("WATERMARK COLOR DETECTION")
        click.echo("=" * 60)

        # Analyze colors
        click.echo(f"\nAnalyzing {num_colors} most common colors in the document...")
        colors = self.analyzer.get_dominant_colors(image_rgb, num_colors=num_colors)

        if not colors:
            click.echo("No colors detected. Using automatic detection.")
            return None

        # Display options
        click.echo("\nDetected colors (likely watermark or text):\n")
        self._display_colors(colors)

        # Get user input
        while True:
            try:
                choice = click.prompt(
                    "\nSelect color number (0-indexed) or 'a' for automatic",
                    type=str,
                    default='a'
                ).strip().lower()

                if choice == 'a' or choice == '':
                    click.echo("Using automatic color detection...")
                    return None

                choice_idx = int(choice)
                if 0 <= choice_idx < len(colors):
                    selected = colors[choice_idx]
                    click.echo(f"\n✓ Selected color: RGB{selected['rgb']}")
                    click.echo(f"  Percentage in document: {selected['percentage']:.2f}%")

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
                    click.echo(f"Invalid choice. Please enter 0-{len(colors)-1} or 'a'")
            except ValueError:
                click.echo("Invalid input. Please enter a number or 'a'")

    def _display_colors(self, colors):
        """Display color options with visual representation.

        Args:
            colors: List of color dictionaries
        """
        # Create color bar visualization
        click.echo("Color bars:")
        for i, color_data in enumerate(colors):
            rgb = color_data['rgb']
            percentage = color_data['percentage']
            gray_val = color_data['gray']

            # Create visual bar (using block characters)
            bar_length = min(int(percentage / 2), 25)
            bar = "█" * bar_length

            click.echo(
                f"  {i}: {bar:<25} RGB{rgb} ({gray_val:3d}) - {percentage:5.1f}%"
            )

        click.echo()

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
                click.echo("Using automatic color detection...")
                return None
        except (EOFError, click.Abort):
            click.echo("\nUsing automatic color detection...")
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
