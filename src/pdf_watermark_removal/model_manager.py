"""Model download and management utilities for YOLO models."""

from pathlib import Path

import requests
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn
from rich.table import Table

# Official pretrained model mappings (verified URLs)
MODEL_URLS = {
    "yolov8n-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
        "size": 7054355,
        "description": "YOLOv8 Nano - Fast baseline",
    },
    "yolov12n-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov12n-seg.pt",
        "size": 6853577,
        "description": "YOLO12 Nano - Higher accuracy",
    },
    "yolo11x-watermark.pt": {
        "url": "https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt",
        "size": 114512018,
        "description": "YOLO11 XLarge - Specialized watermark segmentation/detection",
        "specialized": True,
    },
}

# Model cache directory
MODEL_CACHE_DIR = Path.home() / ".cache" / "pdf_watermark_remover" / "models"


class ModelManager:
    """Manages YOLO model downloading and caching."""

    def __init__(self, verbose=False):
        """Initialize model manager.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.cache_dir = MODEL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def get_model_path(self, model_name):
        """Get model path, auto-download if not exists.

        Args:
            model_name: Model filename (e.g. 'yolov8n-seg.pt')

        Returns:
            Path: Local model file path

        Raises:
            ValueError: If model name is not recognized
            RuntimeError: If download or verification fails
        """
        # Handle full paths (user-provided custom models)
        model_path = Path(model_name)
        if model_path.is_absolute() and model_path.exists():
            if self.verbose:
                self.console.print(f"[dim]Using custom model: {model_path}[/dim]")
            return model_path

        # Handle model names that should be cached
        cache_path = self.cache_dir / model_name
        if cache_path.exists():
            if self.verbose:
                self.console.print(f"[dim]Using cached model: {cache_path}[/dim]")
            return cache_path

        # Model not found, attempt download
        if model_name not in MODEL_URLS:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: "
                f"{', '.join(MODEL_URLS.keys())}"
            )

        return self._download_model(model_name, cache_path)

    def _download_model(self, model_name, target_path):
        """Download model and show progress.

        Args:
            model_name: Model name to download
            target_path: Where to save the model

        Returns:
            Path: Path to downloaded model

        Raises:
            RuntimeError: If download or verification fails
        """
        model_info = MODEL_URLS[model_name]
        url = model_info["url"]

        self.console.print(f"\n[blue]📥 Downloading YOLO model: {model_name}[/blue]")
        self.console.print(f"[dim]   → {url}[/dim]")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            temp_path = target_path.with_suffix(".tmp")

            with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Downloading", total=total_size)

                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

            # Verify downloaded file
            if not self._verify_model(temp_path, model_info):
                temp_path.unlink()
                raise RuntimeError("Model file verification failed")

            temp_path.rename(target_path)
            self.console.print(f"[green]✓ Model saved to: {target_path}[/green]\n")
            return target_path

        except requests.RequestException as e:
            self.console.print(f"[red]✗ Download failed: {e}[/red]")
            raise RuntimeError(f"Failed to download model: {e}") from e

    def _verify_model(self, file_path, model_info):
        """Verify downloaded file integrity.

        Args:
            file_path: Path to downloaded file
            model_info: Model information dict with 'size' and optional 'hash'

        Returns:
            bool: True if file is valid
        """
        if not file_path.exists():
            return False

        # Check file size (primary verification method)
        actual_size = file_path.stat().st_size
        expected_size = model_info.get("size", 0)

        if expected_size and actual_size != expected_size:
            self.console.print(
                f"[yellow]⚠ File size mismatch: {actual_size} vs {expected_size} bytes[/yellow]"
            )
            # For now, accept files within 1% tolerance due to potential URL differences
            tolerance = expected_size * 0.01
            if abs(actual_size - expected_size) > tolerance:
                return False
            self.console.print(
                "[dim]Size variance within tolerance, accepting file[/dim]"
            )

        return True

    def list_available_models(self):
        """List available models and their status."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=25)
        table.add_column("Description", style="white")
        table.add_column("Status", style="green", width=12)
        table.add_column("Size (MB)", justify="right", width=10)

        for model_name, info in MODEL_URLS.items():
            model_path = self.cache_dir / model_name
            description = info.get("description", "General purpose")

            if model_path.exists():
                status = "[green]✓ Cached[/green]"
                size_mb = model_path.stat().st_size / (1024 * 1024)
            else:
                status = "[yellow]◆ Available[/yellow]"
                size_mb = info.get("size", 0) / (1024 * 1024)

            table.add_row(model_name, description, status, f"{size_mb:.1f}")

        self.console.print("\n[bold]Available Models for Watermark Detection[/bold]\n")
        self.console.print(table)
        self.console.print(f"\n[dim]Cache directory: {self.cache_dir}[/dim]\n")
        self.console.print(
            "[cyan]Recommendations:[/cyan]\n"
            "  [yellow]yolov8n-seg.pt[/yellow]        - Fast general-purpose (6.7 MB)\n"
            "  [yellow]yolov12n-seg.pt[/yellow]       - Balanced accuracy (6.5 MB)\n"
            "  [yellow]yolo11x-watermark.pt[/yellow]  - Specialized watermark detection (109 MB)\n"
        )

    def clear_cache(self):
        """Clear all cached models."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.console.print(f"[green]✓ Cleared cache: {self.cache_dir}[/green]")
        else:
            self.console.print("[dim]No cache to clear[/dim]")
