"""Interactive CLI utilities for watermark color selection with optimized UX."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style

from .color_analyzer import ColorAnalyzer, ColorType
from .stats import ColorPreview
from .i18n import t


class ColorSelector:
    """Optimized interactive color selection with single-step flow."""

    def __init__(self, verbose=False):
        """Initialize color selector.

        Args:
            verbose: Enable verbose logging
        """
        self.analyzer = ColorAnalyzer(verbose=verbose)
        self.verbose = verbose
        self.console = Console()

    def select_watermark_color_interactive(self, image_rgb):
        """Smart one-step color selection with auto-recommendation.

        Args:
            image_rgb: First page image for analysis

        Returns:
            Tuple (R, G, B) of selected color or None for auto-detection
        """
        self.console.print(
            "\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]"
        )
        self.console.print("[bold cyan]WATERMARK COLOR DETECTION[/bold cyan]")
        self.console.print(
            "[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]"
        )

        # Analyze and recommend
        colors = self.analyzer.analyze_watermark_color(image_rgb)

        # Filter out background colors
        valid_colors = [c for c in colors if c["color_type"] != ColorType.BACKGROUND]

        if not valid_colors:
            self.console.print(
                "[red]✗ No watermark colors detected. Using automatic detection.[/red]"
            )
            return None

        # Find best watermark
        recommended = None
        for color in valid_colors:
            if color["color_type"] == ColorType.WATERMARK:
                recommended = color
                break

        # Fallback to first valid color if no watermark found
        if not recommended:
            recommended = valid_colors[0]

        self._display_recommendation(recommended, valid_colors)

        # Smart decision tree
        return self._interactive_decision(recommended, valid_colors)

    def _display_recommendation(self, recommended, all_colors):
        """Display the recommended color with confidence and alternatives.

        Args:
            recommended: Recommended color dict
            all_colors: All detected colors (already filtered)
        """
        rgb = recommended["rgb"]
        gray = recommended["gray"]
        confidence = recommended.get("confidence", 0)
        coverage = recommended["coverage"]
        color_type = recommended["color_type"]

        # Create ASCII confidence bar for Windows compatibility
        filled = int(confidence / 5)
        empty = 20 - filled
        confidence_bar = "=" * filled + "-" * empty
        confidence_color = (
            "green" if confidence >= 85 else "yellow" if confidence >= 70 else "red"
        )

        # Type indicator with emoji
        type_indicator = {
            ColorType.WATERMARK: "[yellow]💧 WATERMARK[/yellow]",
            ColorType.TEXT: "[cyan]📝 TEXT[/cyan]",
            ColorType.NOISE: "[red]⚠️  NOISE[/red]",
            ColorType.BACKGROUND: "[red]❌ BACKGROUND[/red]",
        }.get(color_type, "[gray]UNKNOWN[/gray]")

        # Format recommendation panel with real color preview
        panel_content = f"""
[bold cyan]{t("recommended_color")}:[/bold cyan]

[bold]{t("rgb_value")}:[/bold] RGB{rgb}
[bold]{t("gray_level")}:[/bold] {gray}
[bold]{t("coverage")}:[/bold] {coverage:.1f}%
[bold]Type:[/bold] {type_indicator}

[bold]{t("confidence")}:[/bold] [{confidence_color}]{confidence_bar}[/{confidence_color}] {int(confidence)}%

{ColorPreview.create_comparison(rgb)}
"""

        self.console.print(Panel(panel_content, border_style="cyan"))

        # Show alternatives if available with color table
        # Filter to show watermark and text types only
        alternatives = [
            c
            for c in all_colors[1:]
            if c["color_type"] in (ColorType.WATERMARK, ColorType.TEXT)
        ]
        if alternatives:
            self.console.print(f"\n[bold]{t('other_colors')}:[/bold]")
            color_table = ColorPreview.create_color_table(alternatives[:3], i18n_t=t)
            self.console.print(color_table)

    def _display_alternatives_table(self, alternatives):
        """Display alternative colors in a compact table.

        Args:
            alternatives: List of alternative color dicts
        """
        table = Table(show_header=True, header_style="dim magenta", padding=(0, 1))
        table.add_column("RGB", style="dim cyan")
        table.add_column("Coverage", style="dim yellow")
        table.add_column("Usage", style="dim blue")

        for color in alternatives:
            rgb = color["rgb"]
            coverage = color["coverage"]

            # Safely convert to int
            try:
                r = int(rgb[0]) if hasattr(rgb[0], "__int__") else int(rgb[0])
                g = int(rgb[1]) if hasattr(rgb[1], "__int__") else int(rgb[1])
                b = int(rgb[2]) if hasattr(rgb[2], "__int__") else int(rgb[2])
            except (TypeError, ValueError, IndexError):
                r, g, b = 128, 128, 128

            table.add_row(
                f"RGB({r},{g},{b})", f"{coverage:.1f}%", f"{coverage:.1f}% coverage"
            )

        self.console.print(table)

    def _interactive_decision(self, recommended, all_colors):
        """Smart interactive decision with minimal confirmations.

        Args:
            recommended: Recommended color
            all_colors: All detected colors

        Returns:
            Selected color or None
        """
        confidence = recommended.get("confidence", 0)

        # High confidence: Just confirm
        if confidence >= 85:
            try:
                proceed = click.confirm(
                    f"\nUse this color ({confidence}% confidence)?", default=True
                )
                if proceed:
                    self.console.print("[green][+] Using recommended color[/green]")
                    return recommended["rgb"]
            except (EOFError, click.Abort):
                self.console.print("[green]Using recommended color[/green]")
                return recommended["rgb"]

        # Medium confidence: Ask if user wants alternatives
        if confidence >= 70:
            try:
                show_alternatives = click.confirm(
                    f"\nMedium confidence ({confidence}%). Show alternatives?",
                    default=False,
                )
                if show_alternatives:
                    return self._select_from_alternatives(all_colors)
                else:
                    self.console.print("[green][+] Using recommended color[/green]")
                    return recommended["rgb"]
            except (EOFError, click.Abort):
                self.console.print("[green]Using recommended color[/green]")
                return recommended["rgb"]

        # Low confidence: Show alternatives by default
        self.console.print("\n[yellow]Low confidence - showing alternatives[/yellow]")
        return self._select_from_alternatives(all_colors)

    def _select_from_alternatives(self, colors):
        """Let user select from alternatives with visual table.

        Args:
            colors: List of color dicts (already filtered)

        Returns:
            Selected color or None
        """
        self.console.print("\n[bold]Select from available colors:[/bold]\n")

        # Display selection table with type info
        table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
        table.add_column("#", style="cyan", width=3)
        table.add_column("Preview", width=25)
        table.add_column("RGB Value", style="green", width=18)
        table.add_column("Coverage", style="yellow", width=12)
        table.add_column("Type", style="blue", width=12)

        for i, color in enumerate(colors[:10]):
            rgb = color["rgb"]
            coverage = color["coverage"]
            color_type = color["color_type"]

            # Safely convert to int
            try:
                r = int(rgb[0]) if hasattr(rgb[0], "__int__") else int(rgb[0])
                g = int(rgb[1]) if hasattr(rgb[1], "__int__") else int(rgb[1])
                b = int(rgb[2]) if hasattr(rgb[2], "__int__") else int(rgb[2])
            except (TypeError, ValueError, IndexError):
                r, g, b = 128, 128, 128

            # Create colored block with visible characters
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            try:
                block = Text("█" * 15, style=Style(bgcolor=hex_color, color="black"))
            except Exception:
                block = Text("█" * 15)

            type_label = (
                color_type.value.upper()
                if hasattr(color_type, "value")
                else str(color_type)
            )

            table.add_row(
                str(i), block, f"RGB({r},{g},{b})", f"{coverage:.1f}%", type_label
            )

        self.console.print(table)

        # Get selection
        while True:
            try:
                choice = (
                    click.prompt(
                        "\nSelect color number (or 'a' for auto)", type=str, default="a"
                    )
                    .strip()
                    .lower()
                )

                if choice == "a" or choice == "":
                    self.console.print("[green]Using automatic detection[/green]")
                    return None

                choice_idx = int(choice)
                if 0 <= choice_idx < len(colors):
                    selected = colors[choice_idx]
                    self.console.print(
                        f"[green][+] Selected RGB{selected['rgb']}[/green]"
                    )
                    return selected["rgb"]
                else:
                    self.console.print(
                        f"[red]Invalid choice. Enter 0-{len(colors) - 1} or 'a'[/red]"
                    )
            except ValueError:
                self.console.print("[red]Invalid input[/red]")

    def get_color_for_detection(self, first_image_rgb, auto_detect=False):
        """Get watermark color with optimized UX flow.

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
                "\nInteractively select watermark color?", default=True
            )
            if not use_interactive:
                self.console.print("[green]Using automatic detection[/green]")
                return None
        except (EOFError, click.Abort):
            self.console.print("\n[green]Using automatic detection[/green]")
            return None

        return self.select_watermark_color_interactive(first_image_rgb)
