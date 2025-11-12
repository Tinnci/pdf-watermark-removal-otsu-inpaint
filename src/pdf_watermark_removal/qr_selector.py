"""Interactive QR code selection and management."""

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .qr_detector import QRCodeInfo


class QRCodeSelector:
    """Handles interactive QR code selection and user preferences."""

    def __init__(self, verbose: bool = False):
        """Initialize the QR code selector.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.console = Console()

        # Default removal preferences
        self.default_removals = {
            "advertisement": True,  # Remove ads by default
            "general": False,  # Keep general QR codes by default
            "documentation": False,  # Keep documentation QR codes
            "website": False,  # Keep website links by default
            "wifi": False,  # Keep WiFi codes
            "contact": False,  # Keep contact info
            "email": False,  # Keep email codes
            "phone": False,  # Keep phone codes
            "unknown": True,  # Remove unknown codes by default (safety)
        }

    def select_qr_codes_to_remove(
        self, qr_codes: List[QRCodeInfo], image_rgb: Optional[np.ndarray] = None
    ) -> List[QRCodeInfo]:
        """Interactively select which QR codes to remove.

        Args:
            qr_codes: List of detected QR codes
            image_rgb: Optional image for preview

        Returns:
            List of QR codes selected for removal
        """
        if not qr_codes:
            if self.verbose:
                self.console.print("[dim]No QR codes detected[/dim]")
            return []

        # Group QR codes by category
        grouped = self._group_and_summarize(qr_codes)

        # Show detection summary
        self._show_detection_summary(qr_codes, grouped)

        # Ask user about QR code handling
        if not self._should_process_qr_codes():
            return []

        # Show detailed QR code information
        if image_rgb is not None:
            self._show_visual_preview(qr_codes, image_rgb)

        # Get user preferences
        removal_categories = self._get_user_preferences(grouped)

        # Filter QR codes based on user selection
        codes_to_remove = [qr for qr in qr_codes if qr.category in removal_categories]

        # Show final selection summary
        self._show_selection_summary(qr_codes, codes_to_remove)

        return codes_to_remove

    def _group_and_summarize(
        self, qr_codes: List[QRCodeInfo]
    ) -> Dict[str, List[QRCodeInfo]]:
        """Group QR codes by category and provide summary."""
        grouped = {}
        for qr in qr_codes:
            if qr.category not in grouped:
                grouped[qr.category] = []
            grouped[qr.category].append(qr)

        return grouped

    def _show_detection_summary(
        self, qr_codes: List[QRCodeInfo], grouped: Dict[str, List[QRCodeInfo]]
    ):
        """Show a summary of detected QR codes."""
        summary_table = Table(
            title=f"QR Code Detection Summary ({len(qr_codes)} found)"
        )
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_column("Types", style="yellow")
        summary_table.add_column("Default Action", style="magenta")

        for category, codes in grouped.items():
            count = len(codes)
            types = ", ".join({qr.qr_type.value for qr in codes})
            default_action = (
                "Remove" if self.default_removals.get(category, False) else "Keep"
            )
            action_color = "red" if default_action == "Remove" else "green"

            summary_table.add_row(
                category,
                str(count),
                types,
                f"[{action_color}]{default_action}[/{action_color}]",
            )

        self.console.print(
            Panel(
                summary_table,
                title="[bold]QR Code Detection[/bold]",
                border_style="blue",
            )
        )

    def _should_process_qr_codes(self) -> bool:
        """Ask user if they want to process QR codes."""
        self.console.print("\n[bold cyan]QR Code Removal Options:[/bold cyan]")
        self.console.print(
            "• [green]Keep all QR codes[/green] - Preserve all detected QR codes"
        )
        self.console.print(
            "• [yellow]Review and select[/yellow] - Choose which categories to remove"
        )
        self.console.print(
            "• [red]Remove all QR codes[/red] - Treat all QR codes as watermarks"
        )

        choice = Prompt.ask(
            "How would you like to handle QR codes?",
            choices=["keep", "review", "remove_all"],
            default="review",
        )

        return choice in ["review", "remove_all"]

    def _show_visual_preview(self, qr_codes: List[QRCodeInfo], image_rgb: np.ndarray):
        """Show visual preview of detected QR codes."""
        # Create preview image with QR codes highlighted
        preview = image_rgb.copy()

        # Draw bounding boxes and labels
        for i, qr in enumerate(qr_codes):
            x, y, width, height = qr.bbox

            # Draw bounding box
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Add label
            label = f"{i + 1}: {qr.qr_type.value} ({qr.category})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Draw label background
            cv2.rectangle(preview, (x, y - 20), (x + label_size[0], y), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(
                preview, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # Save preview image
        preview_path = "qr_detection_preview.png"
        cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        self.console.print(f"\n[dim]Visual preview saved to: {preview_path}[/dim]")
        self.console.print("[dim]Green boxes show detected QR code locations[/dim]")

    def _get_user_preferences(self, grouped: Dict[str, List[QRCodeInfo]]) -> Set[str]:
        """Get user preferences for QR code removal."""
        removal_categories = set()

        # Show detailed table with content preview
        self.console.print("\n[bold]QR Code Details:[/bold]")

        details_table = Table()
        details_table.add_column("Category", style="cyan")
        details_table.add_column("Count", style="green")
        details_table.add_column("Sample Content", style="yellow")
        details_table.add_column("Action", style="magenta")

        for category, codes in grouped.items():
            # Show sample content (first QR code in category)
            sample = (
                codes[0].content[:50] + "..."
                if len(codes[0].content) > 50
                else codes[0].content
            )

            # Default action based on category
            default_remove = self.default_removals.get(category, False)

            # Ask user about this category
            action = self._ask_category_action(
                category, len(codes), sample, default_remove
            )

            if action == "remove":
                removal_categories.add(category)

            # Add to details table
            action_text = (
                "[red]Remove[/red]" if action == "remove" else "[green]Keep[/green]"
            )
            details_table.add_row(category, str(len(codes)), sample, action_text)

        self.console.print(details_table)

        return removal_categories

    def _ask_category_action(
        self, category: str, count: int, sample: str, default_remove: bool
    ) -> str:
        """Ask user about action for a specific QR code category."""
        default_action = "remove" if default_remove else "keep"

        # For large numbers of QR codes, provide batch options
        if count > 1:
            self.console.print(
                f"\n[bold]{category.title()} QR Codes[/bold] ({count} found)"
            )
            self.console.print(f'Sample content: "{sample}"')

            action = Prompt.ask(
                f"Action for {category} QR codes?",
                choices=["remove", "keep", "review_individual"],
                default=default_action,
            )

            if action == "review_individual":
                # This would require more complex individual review
                # For now, fall back to default
                action = default_action
                self.console.print(
                    "[dim]Individual review not implemented, using category default[/dim]"
                )

        else:
            self.console.print(f"\n[bold]{category.title()} QR Code[/bold]")
            self.console.print(f'Content: "{sample}"')

            action = Prompt.ask(
                "Action for this QR code?",
                choices=["remove", "keep"],
                default=default_action,
            )

        return action

    def _show_selection_summary(
        self, all_codes: List[QRCodeInfo], codes_to_remove: List[QRCodeInfo]
    ):
        """Show final summary of QR code selection."""
        total_count = len(all_codes)
        remove_count = len(codes_to_remove)
        keep_count = total_count - remove_count

        summary_text = f"""
[bold]QR Code Removal Summary:[/bold]
• Total detected: {total_count}
• Selected for removal: [red]{remove_count}[/red]
• Preserved: [green]{keep_count}[/green]

Categories to remove:
"""

        if codes_to_remove:
            removed_categories = {qr.category for qr in codes_to_remove}
            summary_text += ", ".join(f"[red]{cat}[/red]" for cat in removed_categories)
        else:
            summary_text += "[green]None[/green]"

        self.console.print(
            Panel(
                summary_text, title="[bold]Final Selection[/bold]", border_style="blue"
            )
        )

        # Final confirmation
        if codes_to_remove:
            if not Confirm.ask("\nProceed with QR code removal?", default=True):
                return []  # User cancelled

    def get_removals_by_preset(
        self, preset: str, qr_codes: List[QRCodeInfo]
    ) -> List[QRCodeInfo]:
        """Get QR codes to remove based on preset rules.

        Args:
            preset: Preset name ('aggressive', 'conservative', 'ads_only')
            qr_codes: List of detected QR codes

        Returns:
            List of QR codes to remove
        """
        if preset == "aggressive":
            # Remove all QR codes
            return qr_codes.copy()

        elif preset == "conservative":
            # Only remove advertisements and unknown codes
            return [
                qr for qr in qr_codes if qr.category in ["advertisement", "unknown"]
            ]

        elif preset == "ads_only":
            # Only remove advertisements
            return [qr for qr in qr_codes if qr.category == "advertisement"]

        else:
            # Default: interactive selection
            return self.select_qr_codes_to_remove(qr_codes)

    def create_qr_removal_mask(
        self,
        image_shape: Tuple[int, int],
        qr_codes_to_remove: List[QRCodeInfo],
        expansion_pixels: int = 10,
    ) -> np.ndarray:
        """Create a mask for QR codes that should be removed.

        Args:
            image_shape: Shape of the image (height, width)
            qr_codes_to_remove: List of QR codes to remove
            expansion_pixels: Additional pixels to expand the mask

        Returns:
            Binary mask where QR codes to remove are marked
        """
        if not qr_codes_to_remove:
            return np.zeros(image_shape[:2], dtype=np.uint8)

        # Create combined mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for qr in qr_codes_to_remove:
            x, y, width, height = qr.bbox

            # Expand the region slightly to ensure complete removal
            x1 = max(0, x - expansion_pixels)
            y1 = max(0, y - expansion_pixels)
            x2 = min(image_shape[1], x + width + expansion_pixels)
            y2 = min(image_shape[0], y + height + expansion_pixels)

            # Fill the expanded region
            mask[y1:y2, x1:x2] = 255

        return mask
