# PDF Watermark Removal Tool

[![PyPI version](https://badge.fury.io/py/pdf-watermark-removal-otsu-inpaint.svg)](https://pypi.org/project/pdf-watermark-removal-otsu-inpaint/)
[![Version](https://img.shields.io/badge/version-0.4.1-green.svg)](https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Tinnci-black.svg)](https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint)

A command-line tool to remove watermarks from PDF files using advanced image processing techniques including adaptive thresholding, intelligent color detection, and OpenCV inpainting. Features interactive watermark color selection and beautiful CLI with progress visualization.

## 🎯 Key Features

- **🧠 Intelligent Document Classification**: Auto-detects document type (Electronic/Scanned/Mixed)
  - Multi-dimensional visual analysis (color, text, edges, noise)
  - Automatic parameter optimization for each type
  - Confidence-scored classification
  - Can be disabled with `--no-auto-classify`

- **Intelligent Color Detection**: Automatically classifies colors as BACKGROUND, WATERMARK, TEXT, or NOISE
  - Multi-pass analysis with confidence scoring
  - Interactive visual color picker with preview
  - Smart background protection to avoid false positives
  
- **Advanced Watermark Detection**:
  - Adaptive Gaussian thresholding for better precision than traditional Otsu
  - Combined color and saturation analysis
  - Automatic background (white area) exclusion
  - Dark text protection (RGB 0-80 preserved)
  - Morphological operations for noise removal
  
- **Precision Inpainting**:
  - OpenCV TELEA algorithm with dynamic radius adjustment
  - Coverage-based parameter optimization
  - **Strength control**: `--inpaint-strength` (0.5-1.5) for fine-tuned removal
  - Progressive multi-pass removal for stubborn watermarks
  - Accurate color space handling (RGB ↔ BGR conversion)

- **Production Quality CLI**:
  - Beautiful Rich-formatted panels and progress bars
  - Multi-level progress tracking (Overall + Per-Page)
  - Internationalization support (English & Chinese)
  - Detailed logging and statistics
  - Error handling with `--skip-errors` to continue on failures

- **Flexible Processing**:
  - Batch process multiple pages
  - Select specific pages or ranges
  - Adjustable DPI for different quality needs
  - Per-page statistics and coverage reporting
  - Debug mode: `--debug-mask` generates detection preview
  - Strength visibility: `--show-strength` displays parameters

## Installation

### Using uv (recommended)

```bash
uv tool install pdf-watermark-removal-otsu-inpaint
```

### Using pip

```bash
pip install pdf-watermark-removal-otsu-inpaint
```

### With YOLO Detection Support

For using YOLO-based watermark detection (more accurate on complex watermarks):

```bash
# YOLO support (CPU only)
pip install pdf-watermark-removal-otsu-inpaint[yolo]

# YOLO with GPU acceleration (requires PyTorch and CUDA)
pip install pdf-watermark-removal-otsu-inpaint[yolo-gpu]
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

### Preset Mode: Electronic Documents with Precise Color Removal

For electronic documents (PDFs generated from software, not scanned), use the `electronic-color` preset for optimal precise color removal:

```bash
pdf-watermark-removal input.pdf output.pdf --preset electronic-color --color "200,200,200"
```

This preset:
- Uses **extremely strict color matching** (tolerance: 5) - only removes the exact color specified
- Optimized for electronic documents with discrete colors and sharp edges
- Protects black text and white backgrounds
- Uses traditional method (no YOLO required)
- Perfect for removing watermarks of a specific color without affecting other content

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
INPUT_PDF                     Path to input PDF file
OUTPUT_PDF                    Path to output PDF file

OPTIONS:
  --color TEXT                Watermark color as 'R,G,B' (e.g., '128,128,128')
                             Interactive selection if not specified
  --auto-color                Skip interactive selection, use automatic detection
  --pages TEXT                Pages to process (e.g., '1,3,5' or '1-5')
                             Process all pages if not specified
  
  --kernel-size INTEGER       Morphological kernel size (default: 3)
  --inpaint-radius INTEGER    Inpainting radius (default: 2)
  --inpaint-strength FLOAT    Inpainting strength 0.5-1.5 (default: 1.0)
                             0.5=light, 1.0=medium, 1.5=strong
  --multi-pass INTEGER        Number of removal passes (default: 1)
  --dpi INTEGER               DPI for PDF rendering (default: 150)
  
  --color-tolerance INTEGER   Color matching tolerance 0-255 (default: 30)
                             Lower=stricter matching
  --protect-text              Protect dark text from removal (default: True)
  
  --preset                    Preset mode: 'electronic-color' for precise color
                             removal on electronic documents (requires --color)
  --no-auto-classify          Disable automatic document type detection
  --show-strength             Display per-page strength parameters
  --debug-mask                Save debug preview of watermark detection
  --skip-errors               Skip pages with errors instead of failing
  
  --detection-method          Detection method: 'traditional' or 'yolo'
                             (default: traditional)
  --yolo-model PATH          Path to YOLO segmentation model (.pt or .onnx)
                             (default: yolov8n-seg.pt)
  --yolo-conf FLOAT          YOLO confidence threshold 0-1 (default: 0.25)
  --yolo-device              YOLO device: 'cpu', 'cuda', or 'auto'
                             (default: auto)
  --yolo-version             YOLO version: 'v8' or 'v12' (default: v8)
                             v8=fast baseline, v12=higher accuracy
  
  --lang TEXT                 Force language (zh_CN, en_US)
                             Auto-detect if not specified
  -v, --verbose               Enable verbose output
  --help                      Show help message
```

## Common Use Cases

### One-Click Intelligent Processing
```bash
# Auto-classify document type and optimize parameters
pdf-watermark-removal input.pdf output.pdf
```

### Fine-Tuned Control
```bash
# Adjust removal strength for stubborn watermarks
pdf-watermark-removal input.pdf output.pdf --inpaint-strength 1.3

# Stricter color matching for electronic documents
pdf-watermark-removal input.pdf output.pdf --color-tolerance 15

# Multi-pass for heavy watermarks
pdf-watermark-removal input.pdf output.pdf --multi-pass 2

# Debug mode to inspect detection
pdf-watermark-removal input.pdf output.pdf --debug-mask --verbose
```

### Batch Processing with Error Handling
```bash
# Continue processing if some pages fail
pdf-watermark-removal large_document.pdf output.pdf --skip-errors

# Process subset of pages only
pdf-watermark-removal input.pdf output.pdf --pages 1-50 --skip-errors
```

### YOLO-Based Detection (Accurate for Complex Watermarks)
```bash
# Use YOLOv8-seg for detection (fast baseline)
pdf-watermark-removal input.pdf output.pdf --detection-method yolo

# Use YOLO12-seg for detection (higher accuracy, slightly slower)
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-version v12

# Use specialized watermark model (maximum accuracy)
pdf-watermark-removal input.pdf output.pdf \
  --detection-method yolo \
  --yolo-version v11

# YOLO with GPU acceleration
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-device cuda

# YOLO with lower confidence threshold (detect more regions)
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-conf 0.15

# YOLO v11 with aggressive detection (recommended for complex watermarks)
pdf-watermark-removal input.pdf output.pdf \
  --detection-method yolo \
  --yolo-version v11 \
  --yolo-conf 0.10 \
  --yolo-device cpu

# YOLO with custom model and confidence
pdf-watermark-removal input.pdf output.pdf --detection-method yolo \
  --yolo-model "/path/to/custom-model.pt" \
  --yolo-conf 0.30

# List available models
pdf-watermark-removal --list-models
```

**Confidence Threshold Guide**:
- `--yolo-conf 0.25` (default): Balanced detection, fewer false positives
- `--yolo-conf 0.15`: More aggressive, detects subtle watermarks
- `--yolo-conf 0.10`: Very aggressive, catches thin/faint watermarks
- `--yolo-conf 0.05`: Maximum sensitivity, may detect noise

## How It Works

### 1. Color Detection & Selection
- Analyzes first page to detect dominant colors
- Shows most common non-photo colors (likely watermark/text)
- User selects watermark color or confirms automatic selection
- Supports coarse (3) and fine (10) color options

## Algorithm Details

### 0. Document Type Classification
- Analyzes first page using 4 visual dimensions:
  - **Color Discreteness**: Electronic docs have <50 colors, scanned >200 colors
  - **Text Concentration**: Pure black text (0-50 gray) indicates electronic doc
  - **Edge Sharpness**: Laplacian variance analysis
  - **Noise Level**: Denoising comparison
- Returns confidence score and auto-optimized parameters
- User can disable with `--no-auto-classify`

### 1. Intelligent Color Classification
The tool uses multi-dimensional analysis to classify colors:
- **BACKGROUND**: Gray level 240-255 + coverage >60% → confidence 0%
- **WATERMARK**: Gray level 100-250 + coverage 1-20% → dynamic confidence (20-100%)
- **TEXT**: Gray level 0-80 + coverage <5% → confidence 0%
- **NOISE**: All other patterns → confidence 0%

Confidence scoring formula:
```
confidence = (gray_factor × 0.5 + coverage_factor × 0.5) × 100
           + bonus_for_typical_range
```

### 2. Adaptive Watermark Detection
Combines multiple detection methods:
- **Adaptive Gaussian Thresholding**: Handles varying lighting conditions
- **Color-based Detection**: Uses detected watermark color to refine mask
- **Saturation Analysis**: Identifies low-saturation regions (watermarks, text)
- **Background Protection**: Explicitly excludes white/bright areas (>250 gray)
- **Morphological Refinement**: Opens (removes small noise) then closes (fills holes)

### 2. Adaptive Watermark Detection
Combines multiple detection strategies:
- **Adaptive Gaussian Thresholding**: Better than traditional Otsu for varying lighting
- **Color-based Detection**: Matches detected watermark color (±tolerance)
- **Saturation Analysis**: Identifies low-saturation watermark regions
- **Background Protection**: Excludes very bright areas (>250 gray)
- **Text Protection**: Preserves dark text regions (0-80 gray)

Dynamic Detection Parameters:
```
- Color tolerance: Adjusts based on document type (18-32)
- Kernel size: 3 for electronic, 5 for scanned documents
- Aspect ratio filtering: Removes thin text-like components
```

### 3. Strength-Controlled Inpainting
Uses OpenCV's TELEA algorithm with blending:
- **Inpaint Strength** (0.5-1.5): Controls blend ratio
  - 0.5 = 50% blend (preserve original)
  - 1.0 = 100% replacement (standard)
  - 1.5 = 150% radius boost (aggressive)
- **Dynamic Radius**: base_radius + (coverage × 10 × strength)
- **Multi-pass Support**: Progressive expansion for stubborn watermarks
- **Color Space Accuracy**: RGB→BGR conversion for proper processing

### 4. PDF Reconstruction
Preserves document fidelity:
- Maintains original page layout
- Preserves resolution based on input DPI
- Reconstructs from processed image sequence

## Detection Methods

### Traditional CV (Default)
Fast, lightweight detection using adaptive thresholding and color analysis:
- **Speed**: ~100-200ms per page (CPU)
- **Accuracy**: 85-95% for standard watermarks
- **Requirements**: opencv-python, numpy
- **Best for**: Simple, uniform watermarks; fast processing

### YOLO-based Detection (Experimental)
Deep learning-based instance segmentation for complex watermarks:

#### YOLOv8 (Fast Baseline)
```bash
pdf-watermark-removal input.pdf output.pdf --detection-method yolo
```
- **Speed**: ~500-1000ms per page (CPU), ~100-200ms (GPU)
- **Accuracy**: 90-98% for complex/semi-transparent watermarks
- **Model**: yolov8n-seg.pt (6.7 MB)
- **Parameters**: 3.2M
- **Best for**: Mixed content, semi-transparent, overlapping watermarks

#### YOLO12 (Higher Accuracy)
```bash
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-version v12
```
- **Speed**: ~600-1200ms per page (CPU), ~120-250ms (GPU)
- **Accuracy**: 92-99% with region attention mechanism
- **Model**: yolo12n-seg.pt (6.5 MB)
- **Parameters**: 2.6M (lighter)
- **Architecture**: Region Attention + R-ELAN (better for small/complex patterns)
- **Best for**: Complex documents, small watermarks, multi-scale patterns

#### YOLO11 XLarge (Specialized Watermark Detection) ⭐
```bash
pdf-watermark-removal input.pdf output.pdf \
  --detection-method yolo \
  --yolo-model yolo11x-watermark.pt
```
- **Speed**: ~2-3 seconds per page (CPU), ~500ms-1s (GPU)
- **Model**: yolo11x-watermark.pt (101 MB) - Large specialized segmentation model
- **Training**: Trained specifically on watermark detection dataset
- **Best for**: Complex watermark patterns, high-precision detection
- **⚠️ Trade-offs**: Larger model (101 MB), slower inference, requires more resources
- **Advantage**: Detects and segments watermark regions (not just classifies)

**When to use**:
- Maximum precision watermark detection and segmentation
- Production systems with complex/diverse watermarks
- When processing time is not critical
- With GPU acceleration for reasonable speed

#### Quick Selection Guide

| Use Case | Model | Speed | Disk |
|----------|-------|-------|------|
| Fast processing | yolov8n-seg.pt | ⚡⚡⚡ | ✓✓✓ |
| Balanced | yolov12n-seg.pt | ⚡⚡ | ✓✓✓ |
| Best accuracy | yolo11x-watermark.pt | ⚡ | ✓ |

**Quick Recommendations**:
- 📱 **Default/Fast**: Use `yolov8n-seg.pt` (balanced speed and accuracy)
- ⚡ **Balanced**: Use `yolov12n-seg.pt` (better accuracy, slightly slower)
- 🎯 **Maximum Accuracy**: Use `yolo11x-watermark.pt` (specialized, requires GPU)

---

## Fine-Tuning YOLO Detection

### Confidence Threshold (--yolo-conf)

The `--yolo-conf` parameter controls detection sensitivity (0.0-1.0):

```bash
# Default: 0.25 - Good balance
pdf-watermark-removal input.pdf output.pdf --detection-method yolo

# 0.15 - More aggressive detection
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-conf 0.15

# 0.10 - Very aggressive (catches faint watermarks)
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-conf 0.10

# 0.05 - Maximum sensitivity (may include noise)
pdf-watermark-removal input.pdf output.pdf --detection-method yolo --yolo-conf 0.05
```

**Recommendations**:
- **Faint/Thin watermarks**: Use `0.10-0.15`
- **Standard watermarks**: Use default `0.25`
- **Heavy watermarks**: Use `0.30-0.50`
- **Testing**: Start with `0.25`, adjust based on results

## Requirements

- Python 3.8+
- uv package manager (for tool installation)

### Automatic Dependencies
- OpenCV (opencv-python) - Image processing and inpainting
- NumPy - Array operations
- Pillow - Image I/O and PDF generation
- PyPDF - PDF utilities
- Click - CLI framework
- PyMuPDF - Fast PDF rendering
- Rich - Beautiful CLI with colors and progress bars

## Examples

## Example 1: Interactive Color Selection with Rich UI
```bash
$ pdf-watermark-removal contract.pdf contract_clean.pdf

┌─────────────────────────────── Configuration ──────────────────────────────┐
│ PDF Watermark Removal Tool                                                │
│ Input:  contract.pdf                                                      │
│ Output: contract_clean.pdf                                                │
└────────────────────────────────────────────────────────────────────────────┘

Would you like to interactively select the watermark color? [y/N]: y
Use coarse color selection (3 main colors)? [Y/n]: y

============================================================
WATERMARK COLOR DETECTION
============================================================

Analyzing 3 most common colors in the document...

Detected colors (likely watermark or text):

┌─────┬──────────────────────────┬──────────────────┬────────────┬──────────┐
│ Index │ Color Preview          │ RGB Value        │ Gray Level │ Percentage │
├─────┼──────────────────────────┼──────────────────┼────────────┼──────────┤
│ 0   │ ████████████████████░░░░ │ RGB(200, 200, 200) │ 200        │ 45.3%   │
│ 1   │ ███████████░░░░░░░░░░░░░ │ RGB(150, 150, 150) │ 150        │ 28.1%   │
│ 2   │ ████████░░░░░░░░░░░░░░░░ │ RGB(100, 100, 100) │ 100        │ 18.2%   │
└─────┴──────────────────────────┴──────────────────┴────────────┴──────────┘

Select color number (0-indexed) or 'a' for automatic [a]: 0

┌───────────────────── Selected Watermark Color ─────────────────┐
│                                                                 │
│                                                                 │
│ RGB Value: (200, 200, 200)                                     │
│ Gray Level: 200                                                │
│ Percentage in document: 45.30%                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Step 1: Converting PDF to images...
⠋ Loading PDF ────────────────────────────────── 0%
Loaded 34 pages

Step 2: Removing watermarks...
Processing pages ████████████████████░░░░░░░░░░ 66% - 0:00:45
Watermark removal completed

Step 3: Converting images back to PDF...
⠙ Saving PDF

┌──────────────────────────── Success ──────────────────────────┐
│ Watermark removal completed successfully!                    │
│ Output saved to: contract_clean.pdf                          │
└──────────────────────────────────────────────────────────────┘
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

## Contributing

Contributions welcome! Areas for enhancement:
- GPU acceleration for large documents
- Additional inpainting algorithms (e.g., Criminisi)
- Batch API interface
- Additional language support
- Performance benchmarking

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [INSTALL.md](INSTALL.md) - Installation and development guide
- [UV_TOOL_GUIDE.md](UV_TOOL_GUIDE.md) - UV tool configuration
- [ALGORITHM_FIX.md](ALGORITHM_FIX.md) - Detailed algorithm improvements
