# Installation & Usage Guide

## UV Tool Installation

This tool can be installed and run using the `uv` package manager.

### Install the Tool

```bash
# Install from local directory
uv tool install --editable .

# Or install from PyPI (when published)
uv tool install pdf-watermark-removal-otsu-inpaint
```

### Run the Tool

Once installed, you can run it directly:

```bash
pdf-watermark-removal input.pdf output.pdf
```

### Run Without Installation (uvx)

You can also run the tool without installing it:

```bash
# From local directory
uvx --with-editable . pdf-watermark-removal input.pdf output.pdf

# From PyPI (when published)
uvx pdf-watermark-removal-otsu-inpaint input.pdf output.pdf
```

## UV Tool Options

### Basic Usage

```bash
pdf-watermark-removal input.pdf output.pdf
```

### Advanced Options

```bash
# Specify pages to process
pdf-watermark-removal input.pdf output.pdf --pages 1,3,5

# Process page range
pdf-watermark-removal input.pdf output.pdf --pages 1-5

# Multi-pass removal for stubborn watermarks
pdf-watermark-removal input.pdf output.pdf --multi-pass 2

# Custom kernel size for detection
pdf-watermark-removal input.pdf output.pdf --kernel-size 5

# Adjust inpaint radius
pdf-watermark-removal input.pdf output.pdf --inpaint-radius 3

# Higher DPI for better quality
pdf-watermark-removal input.pdf output.pdf --dpi 300

# Verbose output
pdf-watermark-removal input.pdf output.pdf --verbose
```

## Algorithm Details

### Otsu Threshold Segmentation (Otsu阈值分割)

The tool automatically detects watermark regions using Otsu's method:

1. Converts each PDF page to an image
2. Converts image to grayscale
3. Applies Otsu's automatic thresholding to create a binary image
4. Uses morphological operations (open and close) to refine the mask
5. Removes noise by filtering small connected components

### OpenCV Inpaint (OpenCV修复)

Watermarks are removed using OpenCV's inpainting algorithm:

1. Uses the detected mask to identify watermark regions
2. Applies TELEA inpainting method to reconstruct the document
3. Supports multi-pass inpainting for better results

## Requirements

- Python 3.8+
- uv package manager

The following dependencies are installed automatically:
- OpenCV (opencv-python)
- NumPy
- Pillow
- PyPDF
- Click
- PyMuPDF (for PDF rendering)

## Troubleshooting

### "PyMuPDF is required" Error

If you get this error, install PyMuPDF:

```bash
uv tool upgrade pdf-watermark-removal-otsu-inpaint --with PyMuPDF
```

### Poor Watermark Detection

Try adjusting the kernel size:

```bash
# Larger kernel for larger watermarks
pdf-watermark-removal input.pdf output.pdf --kernel-size 7

# Smaller kernel for smaller watermarks
pdf-watermark-removal input.pdf output.pdf --kernel-size 2
```

### Incomplete Watermark Removal

Use multi-pass mode:

```bash
pdf-watermark-removal input.pdf output.pdf --multi-pass 2
```

## Development

To set up the project for development:

```bash
# Clone the repository
git clone <repository-url>
cd pdf-watermark-removal-otsu-inpaint

# Install in editable mode with dev dependencies
uv tool install --editable ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
ruff check src/
```
