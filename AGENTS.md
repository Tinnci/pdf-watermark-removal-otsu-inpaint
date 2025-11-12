# AI Agent Development Guide - PDF Watermark Removal Tool

## Project Overview

**pdf-watermark-removal-otsu-inpaint** is a sophisticated Python-based command-line tool that removes watermarks from PDF documents using advanced computer vision techniques. The tool combines Otsu's automatic thresholding for intelligent watermark detection with OpenCV's inpainting algorithms for high-quality watermark removal.

### Key Features
- **Intelligent Document Classification**: Auto-detects document type (Electronic/Scanned/Mixed) with confidence scoring
- **Advanced Color Detection**: Multi-pass analysis with automatic watermark color classification and interactive selection
- **Dual Detection Methods**: Traditional CV (fast) and YOLO-based (accurate for complex watermarks)
- **Strength-Controlled Inpainting**: Configurable removal strength (0.5-1.5) with dynamic radius adjustment
- **Production-Quality CLI**: Rich-formatted interface with progress tracking and internationalization support
- **Flexible Processing**: Batch processing, selective pages, adjustable DPI, multi-pass removal

## Technology Stack

### Core Dependencies
- **OpenCV** (≥4.8.0): Image processing and inpainting algorithms
- **NumPy** (≥1.21.0): Array operations and masking
- **PyMuPDF** (≥1.23.0): Fast PDF rendering and manipulation
- **Pillow** (≥9.0.0): Image I/O and PDF generation
- **Click** (≥8.0.0): CLI framework
- **Rich** (≥13.0.0): Beautiful terminal UI with progress bars
- **Ultralytics** (≥8.3.227): YOLO detection models (optional)

### Development Tools
- **UV**: Modern Python package manager and tool installer
- **Hatchling**: Build backend for package distribution
- **pytest**: Testing framework
- **ruff**: Fast Python linter and code formatter

## Architecture

### Core Components

```
src/pdf_watermark_removal/
├── cli.py                    # Main CLI entry point and user interface
├── pdf_processor.py          # PDF I/O and image conversion
├── watermark_detector.py     # Traditional CV-based watermark detection
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

1. **Traditional CV (Default)**: Fast, lightweight using adaptive thresholding
2. **YOLOv8**: Balanced speed and accuracy for complex watermarks
3. **YOLOv12**: Higher accuracy with region attention mechanism
4. **YOLO11x**: Specialized watermark detection model (maximum accuracy)

## Build and Test Commands

### Installation
```bash
# Install as UV tool (recommended)
uv tool install pdf-watermark-removal-otsu-inpaint

# Install from local development
uv tool install --editable .

# Install with YOLO support
uv tool install pdf-watermark-removal-otsu-inpaint[yolo]
uv tool install pdf-watermark-removal-otsu-inpaint[yolo-gpu]
```

### Development Setup
```bash
# Create development environment
uv venv
source .venv/bin/activate  # Unix
.\.venv\Scripts\activate   # Windows

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run unit tests
pytest test_watermark.py

# Run with verbose output
pytest -v

# Test CLI functionality
pdf-watermark-removal --help
```

### Code Quality
```bash
# Format code
ruff format src/

# Lint code
ruff check src/

# Run both format and check
ruff check --fix src/

# Type checking (if mypy configured)
mypy src/
```

## Code Style Guidelines

### Python Standards
- **PEP 8**: Follow standard Python formatting
- **Type Hints**: Use type annotations for function parameters and returns
- **Docstrings**: Use Google-style docstrings for all public functions
- **Error Handling**: Use specific exceptions with meaningful messages
- **Logging**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

### Project-Specific Conventions
- **Module Structure**: One class per module, descriptive filenames
- **CLI Design**: Use Click decorators, rich formatting for user interface
- **Image Processing**: Use NumPy arrays, OpenCV conventions (BGR→RGB conversion)
- **Configuration**: Command-line parameters override defaults, no config files
- **Internationalization**: Use `t()` function for all user-facing strings

### Naming Conventions
- **Classes**: PascalCase (e.g., `WatermarkDetector`)
- **Functions/Methods**: snake_case (e.g., `detect_watermark_mask`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_KERNEL_SIZE`)
- **Private Methods**: Prefix with underscore (e.g., `_refine_mask`)

## Testing Instructions

### Unit Tests
```python
# Test watermark detection
def test_watermark_detection():
    image = create_test_image_with_watermark()
    detector = WatermarkDetector()
    mask = detector.detect_watermark_mask(image)
    assert mask is not None
    assert np.any(mask > 0)
```

### Integration Testing
1. **Synthetic Watermarks**: Create test images with known watermarks
2. **Real PDF Testing**: Use sample PDFs with various watermark types
3. **Parameter Validation**: Test different kernel sizes, inpaint radii
4. **Error Handling**: Test invalid inputs, missing files, corrupt PDFs

### Performance Testing
- Measure processing time per page at different DPI settings
- Test memory usage with large documents
- Validate multi-threading safety

## Security Considerations

### Input Validation
- **File Path Validation**: Ensure input files exist and are readable
- **PDF Security**: Handle encrypted/password-protected PDFs gracefully
- **Image Bounds**: Validate image dimensions to prevent memory exhaustion
- **Color Values**: Validate RGB color ranges (0-255)

### Resource Management
- **Memory Limits**: Process large documents in chunks
- **Temporary Files**: Clean up intermediate images
- **File Permissions**: Respect system file permissions
- **Network Requests**: Secure model downloads with checksums

### Data Privacy
- **Local Processing**: All processing happens locally, no cloud uploads
- **No Telemetry**: No usage data collection
- **Temporary Data**: Clean processing artifacts
- **Output Integrity**: Preserve original document structure

## Common Development Tasks

### Adding New Detection Methods
1. Create new detector class inheriting from base interface
2. Implement `detect_watermark_mask(image_rgb)` method
3. Add CLI option in `cli.py`
4. Update documentation and tests

### Modifying Inpainting Algorithm
1. Edit `watermark_remover.py`
2. Change `cv2.inpaint()` method or parameters
3. Test with various watermark types
4. Update performance benchmarks

### Adding CLI Options
1. Add Click option decorator in `cli.py`
2. Pass parameter to appropriate processor
3. Update help text and documentation
4. Add validation if needed

### Internationalization
1. Add new strings to `i18n.py`
2. Use `t('string_key')` in code
3. Update translation dictionaries
4. Test with different locales

## Debugging and Troubleshooting

### Common Issues
- **Poor Detection**: Adjust kernel size, try different color tolerance
- **Artifacts**: Reduce inpaint radius, use single-pass mode
- **Memory Errors**: Lower DPI, process specific pages
- **YOLO Issues**: Check model downloads, validate CUDA setup

### Debug Mode
```bash
# Enable verbose logging
pdf-watermark-removal input.pdf output.pdf --verbose

# Generate debug mask preview
pdf-watermark-removal input.pdf output.pdf --debug-mask

# Show processing parameters
pdf-watermark-removal input.pdf output.pdf --show-strength
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile specific function
cProfile.run('detector.detect_watermark_mask(image)', 'profile.stats')
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative').print_stats(10)
```

## Deployment and Distribution

### Package Building
```bash
# Build distribution packages
uv build

# Upload to PyPI
uv publish
```

### Version Management
- Update version in `pyproject.toml`
- Tag release in git
- Update changelog
- Test installation from PyPI

### Platform Support
- **Primary**: Windows, macOS, Linux
- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Architectures**: x86_64, ARM64

## API Reference

### Core Classes
- `PDFProcessor`: Handle PDF I/O operations
- `WatermarkDetector`: Traditional CV-based detection
- `WatermarkRemover`: Inpainting-based removal
- `ColorSelector`: Interactive color selection
- `DocumentClassifier`: Document type detection
- `YOLODetector`: Deep learning-based detection

### Key Methods
- `pdf_to_images(pdf_path, pages=None)`: Convert PDF to images
- `detect_watermark_mask(image_rgb)`: Generate watermark mask
- `remove_watermark(image_rgb)`: Remove watermarks via inpainting
- `images_to_pdf(images, output_path)`: Convert images back to PDF

This guide provides comprehensive information for AI agents working on this project. Always refer to the actual codebase and documentation for the most up-to-date implementation details.