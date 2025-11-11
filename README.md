# PDF Watermark Removal Tool

A command-line tool to remove watermarks from PDF files using Otsu threshold segmentation and OpenCV inpaint.

## Features

- **Otsu Threshold Segmentation**: Automatically detects watermark regions using Otsu's method
- **OpenCV Inpaint**: Intelligently removes watermarks while preserving document content
- **PDF Support**: Works with multi-page PDF documents
- **Batch Processing**: Process multiple files efficiently

## Installation

### Using uv (recommended)

```bash
uv tool install pdf-watermark-removal-otsu-inpaint
```

### Using pip

```bash
pip install pdf-watermark-removal-otsu-inpaint
```

## Quick Start

### Basic Usage

```bash
pdf-watermark-removal input.pdf output.pdf
```

### With Options

```bash
pdf-watermark-removal input.pdf output.pdf --threshold 150 --kernel-size 5
```

## Command-Line Options

- `input_pdf`: Path to the input PDF file
- `output_pdf`: Path to the output PDF file
- `--threshold`: Otsu threshold value (default: auto-detected)
- `--kernel-size`: Morphological kernel size for watermark detection (default: 3)
- `--inpaint-radius`: Radius of inpainting (default: 2)
- `--pages`: Specific pages to process (e.g., "1,3,5" or "1-5")
- `--verbose`: Enable verbose output

## How It Works

1. **Convert PDF to Images**: Converts each PDF page to an image
2. **Detect Watermarks**: Uses Otsu thresholding to identify watermark regions
3. **Inpaint**: Applies OpenCV's inpainting algorithm to remove watermarks
4. **Reconstruct PDF**: Converts processed images back to PDF

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow
- PyPDF

## License

MIT
