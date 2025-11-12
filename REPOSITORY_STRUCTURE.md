# Repository Structure

This document describes the organized structure of the PDF Watermark Removal project.

## Root Directory

```
pdf-watermark-removal-otsu-inpaint/
├── .github/                    # GitHub workflows and CI/CD
├── docs/                       # Documentation and development images
├── example/                    # Example files and usage samples
├── src/                        # Source code
├── tests/                      # Test scripts and validation tools
├── .venv/                      # Virtual environment (auto-generated)
├── dist/                       # Build distribution files
├── .gitignore                  # Git ignore rules
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
└── [*.md files]               # Documentation files
```

## Documentation Files

### Main Documentation
- **`README.md`** - Project overview, installation, and basic usage
- **`ARCHITECTURE.md`** - System architecture and design patterns
- **`CLI_WORKFLOW_ANALYSIS.md`** - Complete processing pipeline analysis
- **`AGENTS.md`** - Development guidelines and coding standards
- **`INSTALL.md`** - Detailed installation instructions
- **`UV_TOOL_GUIDE.md`** - UV package manager usage guide
- **`PROTECTION_FIX_SUMMARY.md`** - Technical summary of text protection improvements

### Development Documentation (`docs/`)
```
docs/
├── development/               # Development visualizations and test images
│   ├── debug_protection_steps.png
│   ├── detector_vs_manual.png
│   ├── improved_text_protection.png
│   ├── manual_protection_steps.png
│   ├── test_protection_results.png
│   └── [other test images]
│
└── README.md                  # Documentation directory overview
```

## Source Code (`src/`)

```
src/pdf_watermark_removal/
├── __init__.py                # Package initialization
├── cli.py                     # Command-line interface
├── pdf_processor.py           # PDF I/O operations
├── watermark_detector.py      # Detection algorithms (traditional CV + YOLO)
├── watermark_remover.py       # Inpainting-based removal
├── color_selector.py          # Interactive color selection
├── color_analyzer.py          # Intelligent color classification
├── document_classifier.py     # Document type detection
├── stats.py                   # Processing statistics
├── i18n.py                    # Internationalization
├── model_manager.py           # YOLO model management
└── yolo_detector.py           # YOLO-based detection
```

## Tests (`tests/`)

```
tests/
├── debug/                     # Development and debugging tools
│   ├── debug_actual_detector.py
│   ├── debug_protection.py
│   ├── debug_step_by_step.py
│   └── fix_text_protection.py
│
├── fix_validation/           # Validation of the text protection fix
│   ├── test_protection_order.py
│   └── test_protection_improvement.py
│
├── test_watermark.py         # Original unit tests
└── README.md                 # Tests directory overview
```

## Key Features

### Algorithm Composition
- **Traditional CV Detection**: Fast, lightweight using adaptive thresholding
- **YOLO-based Detection**: Accurate using YOLOv8/v12/v11 segmentation models
- **Intelligent Document Classification**: Auto-detects document type and optimizes parameters
- **Advanced Color Analysis**: Multi-pass analysis with interactive selection
- **Strength-Controlled Inpainting**: Configurable removal with dynamic radius adjustment

### Recent Improvements
- **"Protect First, Refine Second"**: Fixed text etching issue by reordering protection logic
- **Enhanced Text Protection**: Better handling of anti-aliased text edges
- **Comprehensive Testing**: Multiple validation scripts and visualizations

## Usage

### Installation
```bash
uv tool install pdf-watermark-removal-otsu-inpaint
```

### Basic Usage
```bash
pdf-watermark-removal input.pdf output.pdf
```

### Running Tests
```bash
# Unit tests
python tests/test_watermark.py

# Fix validation
python tests/fix_validation/test_protection_order.py

# Debug analysis
python tests/debug/debug_protection.py
```

## Development

### Code Quality
- **Ruff**: Linting and formatting (all checks pass)
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Google-style docstrings
- **Testing**: Multiple validation approaches

### Recent Work
The repository has been recently organized with:
- Clean separation of concerns
- Comprehensive documentation
- Thorough testing infrastructure
- Professional project structure

This structure supports both end-users and developers working on the project.