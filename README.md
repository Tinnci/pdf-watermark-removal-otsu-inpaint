# PDF Watermark Removal Tool

A command-line UV tool to remove watermarks from PDF files using Otsu threshold segmentation and OpenCV inpaint with interactive watermark color detection.

## Features

- **Otsu Threshold Segmentation**: Automatically detects watermark regions using Otsu's method
- **OpenCV Inpaint**: Intelligently removes watermarks while preserving document content
- **Interactive Color Detection**: Visual color picker to select watermark color from the document
  - Coarse mode: Shows 3 most common colors
  - Fine mode: Shows 10 most common colors for precise selection
- **PDF Support**: Works with multi-page PDF documents
- **Full Document Processing**: Processes all pages by default
- **Progress Visualization**: Real-time progress bars for each operation
- **Flexible Parameters**: Adjust kernel size, inpaint radius, DPI, and more
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

### From local directory

```bash
cd pdf-watermark-removal-otsu-inpaint
uv tool install --editable .
```

## Quick Start

### Basic Usage (All Pages, Interactive Color Selection)

```bash
pdf-watermark-removal input.pdf output.pdf
```

### Specify Watermark Color Explicitly

```bash
pdf-watermark-removal input.pdf output.pdf --color "200,200,200"
```

The color format is R,G,B with values from 0-255.

### Skip Interactive Selection

```bash
pdf-watermark-removal input.pdf output.pdf --auto-color
```

### Process Specific Pages Only

```bash
pdf-watermark-removal input.pdf output.pdf --pages 1,3,5
pdf-watermark-removal input.pdf output.pdf --pages 1-10
```

### With Advanced Options

```bash
pdf-watermark-removal input.pdf output.pdf \
  --color "180,180,180" \
  --kernel-size 5 \
  --inpaint-radius 3 \
  --multi-pass 2 \
  --dpi 300 \
  --verbose
```

## Command-Line Options

```
INPUT_PDF                  Path to input PDF file
OUTPUT_PDF                 Path to output PDF file

OPTIONS:
  --color TEXT             Watermark color as 'R,G,B' (e.g., '128,128,128')
                          Interactive selection if not specified
  --auto-color             Skip interactive selection, use automatic detection
  --pages TEXT             Pages to process (e.g., '1,3,5' or '1-5')
                          Process all pages if not specified
  --kernel-size INTEGER    Morphological kernel size (default: 3)
  --inpaint-radius INTEGER Inpainting radius (default: 2)
  --multi-pass INTEGER     Number of removal passes (default: 1)
  --dpi INTEGER            DPI for PDF rendering (default: 150)
  -v, --verbose            Enable verbose output
  --help                   Show help message
```

## How It Works

### 1. Color Detection & Selection
- Analyzes first page to detect dominant colors
- Shows most common non-photo colors (likely watermark/text)
- User selects watermark color or confirms automatic selection
- Supports coarse (3) and fine (10) color options

### 2. Watermark Detection (Otsu Threshold)
- Converts each PDF page to image at specified DPI
- Converts image to grayscale
- Applies Otsu's automatic thresholding to create binary image
- Uses morphological operations (open and close) to refine mask
- Combines with color saturation analysis for better detection
- Filters small noise components

### 3. Watermark Removal (Inpainting)
- Uses detected mask to identify watermark regions
- Applies OpenCV's TELEA inpainting method
- Reconstructs watermarked areas using surrounding texture
- Supports multi-pass for stubborn watermarks

### 4. PDF Reconstruction
- Converts processed images back to PDF
- Preserves document layout and quality

## Algorithm Details

### Otsu阈值分割 (Otsu Threshold Segmentation)
The tool automatically detects optimal threshold using Otsu's method:
- No manual parameter tuning needed
- Separates foreground (watermark) from background (document)
- Works well with text and graphic watermarks

### OpenCV修复 (OpenCV Inpainting)
Uses TELEA algorithm for content-aware fill:
- Telea method: Fast, good for document cleanup
- Intelligent interpolation from surrounding pixels
- Preserves document structure and text

## Requirements

- Python 3.8+
- uv package manager (for tool installation)

### Automatic Dependencies
- OpenCV (opencv-python)
- NumPy
- Pillow
- PyPDF
- Click
- PyMuPDF

## Examples

### Example 1: Interactive Color Selection
```bash
$ pdf-watermark-removal contract.pdf contract_clean.pdf

Loading first page  [##################################] 100%

Would you like to interactively select the watermark color?
[y/N]: y
Use coarse color selection (3 main colors)? [Y/n]: y

============================================================
WATERMARK COLOR DETECTION
============================================================

Analyzing 3 most common colors in the document...

Detected colors (likely watermark or text):

Color bars:
  0: ██████████████████████       RGB(200, 200, 200) (200)  45.3%
  1: ████████████████             RGB(150, 150, 150) (150)  28.1%
  2: ██████████                   RGB(100, 100, 100) (100)  18.2%

Select color number (0-indexed) or 'a' for automatic [a]: 0

✓ Selected color: RGB(200, 200, 200)
  Percentage in document: 45.3%
...
✓ Watermark removal completed successfully!
Output saved to: contract_clean.pdf
```

### Example 2: Explicit Color and Multi-Pass
```bash
pdf-watermark-removal document.pdf clean.pdf \
  --color "220,220,220" \
  --multi-pass 2 \
  --verbose
```

### Example 3: High-Quality Processing
```bash
pdf-watermark-removal thesis.pdf thesis_clean.pdf \
  --dpi 300 \
  --kernel-size 5 \
  --inpaint-radius 3 \
  --auto-color
```

## Performance

Typical processing times on modern systems:
- Single page: 1-2 seconds
- 10 pages: 10-20 seconds
- 100 pages: 2-5 minutes

Factors affecting speed:
- PDF resolution (DPI)
- Page complexity
- Inpaint radius
- Multi-pass count
- System CPU/memory

## Troubleshooting

### Poor Watermark Detection

**Symptoms**: Watermark not fully detected

**Solutions**:
1. Try fine color selection: `--color "180,180,180"` with different values
2. Increase kernel size: `--kernel-size 5` or `--kernel-size 7`
3. Use multi-pass: `--multi-pass 2`

### Artifacts or Blurriness

**Symptoms**: Cleaned PDF has blurry or distorted areas

**Solutions**:
1. Reduce inpaint radius: `--inpaint-radius 1`
2. Lower DPI: `--dpi 150` (default is good for most documents)
3. Single pass: `--multi-pass 1` (default)

### Memory Issues

**Symptoms**: "Out of memory" error on large PDFs

**Solutions**:
1. Lower DPI: `--dpi 100`
2. Process specific pages: `--pages 1-50` (then 51-100, etc.)
3. Increase system available memory

## License

MIT

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [INSTALL.md](INSTALL.md) - Installation and development guide
- [UV_TOOL_GUIDE.md](UV_TOOL_GUIDE.md) - UV tool configuration details
