# PDF Watermark Removal Tool

[![PyPI version](https://badge.fury.io/py/pdf-watermark-removal-otsu-inpaint.svg)](https://pypi.org/project/pdf-watermark-removal-otsu-inpaint/)
[![Version](https://img.shields.io/badge/version-0.5.5-green.svg)](https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-Tinnci-black.svg)](https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint)

A sophisticated command-line tool that removes watermarks from PDF files using advanced computer vision techniques including adaptive thresholding, intelligent color detection, and OpenCV inpainting. Features interactive watermark color selection and beautiful CLI with progress visualization.

## 🎯 Key Features

- **🧠 Intelligent Document Classification**: Auto-detects document type (Electronic/Scanned/Mixed)
  - Multi-dimensional visual analysis (color, text, edges, noise)
  - Automatic parameter optimization for each type
  - Confidence-scored classification
  - Can be disabled with `--no-auto-classify`

- **🎨 Intelligent Color Detection**: Automatically classifies colors as BACKGROUND, WATERMARK, TEXT, or NOISE
  - Multi-pass analysis with confidence scoring
  - Interactive visual color picker with preview
  - Smart background protection to avoid false positives
  
- **🔍 Advanced Watermark Detection**:
  - Adaptive Gaussian thresholding for better precision than traditional Otsu
  - Combined color and saturation analysis
  - Automatic background (white area) exclusion
  - Dark text protection (RGB 0-80 preserved)
  - Morphological operations for noise removal
  
- **🛠️ Precision Inpainting**:
  - OpenCV TELEA algorithm with dynamic radius adjustment
  - Coverage-based parameter optimization
  - **Strength control**: `--inpaint-strength` (0.5-1.5) for fine-tuned removal
  - Progressive multi-pass removal for stubborn watermarks
  - Accurate color space handling (RGB ↔ BGR conversion)

- **✨ Production Quality CLI**:
  - Beautiful Rich-formatted panels and progress bars
  - Multi-level progress tracking (Overall + Per-Page)
  - Internationalization support (English & Chinese)
  - Detailed logging and statistics
  - Error handling with `--skip-errors` to continue on failures

- **⚙️ Flexible Processing**:
  - Batch process multiple pages
  - Select specific pages or ranges
  - Adjustable DPI for different quality needs
  - Per-page statistics and coverage reporting
  - Debug mode: `--debug-mask` generates detection preview

## 🚀 Quick Start

### Installation

Choose your preferred installation method:

#### Option 1: UV Tool (Recommended)
```bash
# Install from PyPI
uv tool install pdf-watermark-removal-otsu-inpaint

# Or install with YOLO support
uv tool install pdf-watermark-removal-otsu-inpaint[yolo]

# Run directly
pdf-watermark-removal input.pdf output.pdf
```

#### Option 2: Development Installation
```bash
# Clone and install in development mode
git clone https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint.git
cd pdf-watermark-removal-otsu-inpaint
uv tool install --editable .

# Or with development dependencies
uv pip install -e ".[dev]"
```

#### Option 3: From PyPI with pip
```bash
pip install pdf-watermark-removal-otsu-inpaint

# With YOLO support
pip install pdf-watermark-removal-otsu-inpaint[yolo]
```

### Basic Usage

```bash
# Simple watermark removal
pdf-watermark-removal input.pdf output.pdf

# Specify watermark color
pdf-watermark-removal input.pdf output.pdf --color "200,200,200"

# Process specific pages
pdf-watermark-removal input.pdf output.pdf --pages 1,3,5-10

# Use YOLO detection for complex watermarks
pdf-watermark-removal input.pdf output.pdf --detection-method yolo

# Adjust removal strength
pdf-watermark-removal input.pdf output.pdf --inpaint-strength 1.2 --multi-pass 2
```

## 📋 Command Reference

### Core Options
```bash
pdf-watermark-removal input.pdf output.pdf [OPTIONS]

Options:
  --kernel-size INTEGER         Morphological kernel size [default: 3]
  --inpaint-radius INTEGER      Radius for inpainting [default: 2]
  --inpaint-strength FLOAT      Removal strength 0.5-1.5 [default: 1.0]
  --pages TEXT                  Pages to process ('1,3,5' or '1-5')
  --multi-pass INTEGER          Number of removal passes [default: 1]
  --dpi INTEGER                DPI for conversion [default: 150]
  --color TEXT                 Watermark color 'R,G,B'
  --auto-color                 Skip interactive color selection
  --protect-text               Protect dark text [default: True]
  --color-tolerance INTEGER    Color matching tolerance [default: 30]
  --debug-mask                 Save detection preview
  --skip-errors                Continue on page errors
  --show-strength             Display strength parameters
  --no-auto-classify          Disable document classification
  --detection-method TEXT     'traditional' or 'yolo' [default: traditional]
  --yolo-model PATH           YOLO model path
  --yolo-conf FLOAT          YOLO confidence threshold [default: 0.25]
  --yolo-device TEXT          Device: 'auto', 'cpu', 'cuda'
  --yolo-version TEXT         YOLO version: 'v8', 'v12', 'v11'
  --preset TEXT              Preset: 'electronic-color'
  --lang TEXT                Language: 'en_US', 'zh_CN'
  --verbose, -v              Enable verbose output
  --list-models              List available YOLO models
  --help                     Show this message and exit.
```

### Advanced Examples

```bash
# Electronic documents with precise color removal
pdf-watermark-removal report.pdf clean.pdf --preset electronic-color --color "210,210,210"

# Scanned documents with higher tolerance
pdf-watermark-removal scan.pdf clean.pdf --color-tolerance 40 --inpaint-strength 1.3

# Multi-pass removal for stubborn watermarks
pdf-watermark-removal stubborn.pdf clean.pdf --multi-pass 3 --inpaint-strength 1.5

# Debug mode with detailed output
pdf-watermark-removal input.pdf output.pdf --verbose --debug-mask --show-strength

# Batch processing with YOLO detection
pdf-watermark-removal batch/*.pdf output/ --detection-method yolo --yolo-version v12
```

## 🏗️ System Architecture

### Core Components

```
src/pdf_watermark_removal/
├── cli.py                    # Main CLI entry point and user interface
├── pdf_processor.py          # PDF I/O and image conversion
├── watermark_detector.py     # Traditional CV and YOLO detection
├── watermark_remover.py      # OpenCV inpainting implementation
├── color_selector.py         # Interactive color selection UI
├── color_analyzer.py         # Intelligent color classification
├── document_classifier.py    # Document type detection and optimization
├── yolo_detector.py          # YOLO-based watermark detection
├── model_manager.py          # YOLO model management and downloads
├── i18n.py                   # Internationalization support
└── stats.py                  # Processing statistics and reporting
```

### Processing Pipeline

```
PDF Input → Document Classification → Color Analysis → Watermark Detection → 
Inpainting Removal → PDF Reconstruction → Output
```

### Detection Methods

1. **Traditional CV (Default)**: Fast, lightweight using adaptive thresholding and color analysis
2. **YOLOv8**: Balanced speed and accuracy for complex watermarks  
3. **YOLOv12**: Higher accuracy with region attention mechanism
4. **YOLO11x**: Specialized watermark detection model (maximum accuracy)

### Recent Improvements

**"Protect First, Refine Second" Approach**: Fixed text "etching" by applying protection before morphological operations, achieving 59.6% reduction in text artifacts.

## 🔧 Development

### Repository Structure
```
pdf-watermark-removal-otsu-inpaint/
├── .github/                    # CI/CD workflows
├── docs/                       # Documentation and development images
├── example/                    # Example files and usage samples
├── src/                        # Source code
├── tests/                      # Test scripts and validation tools
├── [documentation files]       # This README and other docs
└── [configuration files]       # pyproject.toml, uv.lock, etc.
```

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/Tinnci/pdf-watermark-removal-otsu-inpaint.git
cd pdf-watermark-removal-otsu-inpaint

# Create virtual environment
uv venv
source .venv/bin/activate  # Unix
.
.venv\Scripts\activate     # Windows

# Install in development mode
uv pip install -e ".[dev]"

# Run tests
python tests/test_watermark.py
python tests/fix_validation/test_protection_order.py
```

### Code Quality

```bash
# Format and lint code
ruff format src/
ruff check src/
ruff check --fix src/
```

### Repository Structure Analysis

When analyzing the project structure, exclude the virtual environment directory to avoid cluttering the output:
```bash
# Exclude .venv when using tree command
tree /F /A | findstr /V "\.venv"

# Or use PowerShell to exclude multiple directories
tree /F /A | Where-Object { $_ -notmatch "\.venv|__pycache__|\.ruff_cache" }
```

## 📊 Performance & Results

### Quantitative Improvements (v0.5.3+)
- **59.6% reduction** in text artifacts after "Protect First, Refine Second" fix
- **29.7% reduction** in total mask pixels while maintaining detection accuracy
- **Multi-level progress tracking** for better user experience
- **Dynamic parameter optimization** based on document type

### Processing Performance
- **Traditional CV**: ~1-3 seconds per page at 150 DPI
- **YOLO-based**: ~3-8 seconds per page depending on model
- **Memory efficient**: Processes large documents in chunks
- **Multi-threading safe**: Can process multiple documents concurrently

## 🔍 Debugging

### Debug Mode
```bash
# Enable verbose logging
pdf-watermark-removal input.pdf output.pdf --verbose

# Generate detection preview
pdf-watermark-removal input.pdf output.pdf --debug-mask

# Show processing parameters
pdf-watermark-removal input.pdf output.pdf --show-strength
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Poor detection | Adjust `--kernel-size` or try different `--color-tolerance` |
| Text artifacts | Use `--protect-text` and lower `--inpaint-strength` |
| Memory errors | Lower `--dpi` or process specific `--pages` |
| YOLO not available | Install with `pip install pdf-watermark-removal-otsu-inpaint[yolo]` |

## 🌍 Internationalization

The tool supports multiple languages:
- **English** (default): `pdf-watermark-removal input.pdf output.pdf`
- **Chinese**: `pdf-watermark-removal input.pdf output.pdf --lang zh_CN`

## 📚 Additional Documentation

For detailed technical information, see:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture and technical details
- **[CLI_WORKFLOW_ANALYSIS.md](CLI_WORKFLOW_ANALYSIS.md)** - Detailed processing pipeline analysis
- **[PROTECTION_FIX_SUMMARY.md](PROTECTION_FIX_SUMMARY.md)** - Technical details of the text protection improvements
- **[tests/README.md](tests/README.md)** - Testing infrastructure and validation tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure code quality: `ruff check src/ && ruff format src/`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision algorithms
- [Ultralytics](https://ultralytics.com/) for YOLO models
- [Rich](https://rich.readthedocs.io/) for beautiful terminal UI
- [UV](https://docs.astral.sh/uv/) for modern Python package management