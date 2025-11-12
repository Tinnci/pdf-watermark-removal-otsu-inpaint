# Architecture & Technical Documentation

## Project Overview

**pdf-watermark-removal-otsu-inpaint** is a sophisticated Python-based UV tool that automatically removes watermarks from PDF documents using advanced computer vision techniques: Otsu's automatic thresholding for watermark detection and OpenCV's inpainting algorithm for watermark removal.

## System Architecture

### High-Level Processing Flow

```
PDF Input
    ↓
[Document Classifier] → Determine document type → Optimize parameters
    ↓
[PDF Processor] → Convert to Images (configurable DPI)
    ↓
[Color Analyzer] → Detect/classify colors (BACKGROUND/WATERMARK/TEXT/NOISE)
    ↓
[Watermark Detector] → Multi-method detection (Traditional CV or YOLO)
    ↓
[Watermark Remover] → Inpainting with strength control
    ↓
[PDF Processor] → Convert back to PDF
    ↓
PDF Output
```

### Core Components

#### 1. PDF Processor (`pdf_processor.py`)
**Responsibility**: Handle PDF I/O operations and image conversions

**Key Methods**:
- `pdf_to_images(pdf_path, pages=None)`: Converts PDF pages to RGB numpy arrays
  - Uses PyMuPDF (fitz) for fast, high-quality PDF rendering
  - Supports selective page processing
  - Configurable DPI for quality/performance tradeoff
  
- `images_to_pdf(images, output_path)`: Converts processed images back to PDF
  - Uses Pillow for image-to-PDF conversion
  - Preserves image quality and document structure

**Configuration**:
- `dpi`: DPI for rendering (default: 150, can be 300+ for high quality)
- `verbose`: Logging output

#### 2. Document Classifier (`document_classifier.py`)
**Responsibility**: Automatically classify document type for parameter optimization

**Document Types**:
- **Electronic**: Discrete colors, pure black text, sharp edges
- **Scanned**: Continuous colors, noise, blurred edges  
- **Mixed**: Conservative parameters needed

**Classification Features**:
- Color discreteness analysis
- Text gray concentration
- Edge sharpness detection
- Noise level measurement

#### 3. Watermark Detector (`watermark_detector.py`)
**Responsibility**: Identify watermark regions using multiple detection methods

**Detection Methods**:

**Traditional CV (Default)**:
- **Color-based detection**: Precise color matching with tolerance
- **Automatic detection**: Adaptive thresholding + saturation analysis
- **Protection layers**: Text and background protection before morphological operations

**YOLO-based**:
- **YOLOv8n-seg**: Fast baseline model
- **YOLOv12n-seg**: Higher accuracy with region attention
- **YOLO11x-watermark**: Specialized watermark detection

**Key Innovation - "Protect First, Refine Second"**:
```python
# New approach (v0.5.3+)
raw_mask = create_color_mask()
text_protected = protect_text_from(raw_mask)     # PROTECT FIRST
background_protected = protect_background(text_protected)
final_mask = apply_morphology(background_protected)  # REFINE SECOND

# Old approach (caused text etching)
raw_mask = create_color_mask()
refined_mask = apply_morphology(raw_mask)        # EXPANDED FIRST
final_mask = try_to_protect_from(refined_mask)  # TOO LATE!
```

#### 4. Color Analyzer (`color_analyzer.py`)
**Responsibility**: Intelligent color classification and analysis

**Color Classification**:
- **BACKGROUND**: Very light (240-255) and high coverage (>60%)
- **WATERMARK**: Mid-high grayscale (180-240) and moderate coverage (2-15%)
- **TEXT**: Dark regions (0-150) with specific characteristics
- **NOISE**: Small components that don't match other categories

#### 5. Watermark Remover (`watermark_remover.py`)
**Responsibility**: Remove detected watermarks using inpainting

**Algorithm**: OpenCV TELEA (Fast Marching Method)
- **Dynamic radius**: Based on watermark coverage percentage
- **Strength control**: Configurable blending (0.5 = light, 1.5 = strong)
- **Multi-pass processing**: Progressive mask expansion for stubborn watermarks
- **Color accuracy**: Proper RGB↔BGR conversion for OpenCV operations

#### 6. CLI Interface (`cli.py`)
**Responsibility**: User interface and command processing

**Features**:
- **Rich formatting**: Beautiful panels and progress bars
- **Multi-level progress**: Overall + per-page tracking
- **Internationalization**: English and Chinese support
- **Error handling**: Graceful failure with `--skip-errors`
- **Debug mode**: Detailed logging and mask preview generation

## Algorithm Details

### Watermark Detection Algorithm (Traditional CV)

**Color-Based Mode** (when color specified):
```python
# 1. Create raw color mask
target_gray = mean(watermark_color)
color_diff = abs(gray - target_gray)
color_mask = (color_diff < tolerance).astype(uint8) * 255

# 2. PROTECT FIRST - Apply protection before morphological operations
text_protect_mask = get_text_protect_mask(gray)
protected_mask = bitwise_and(color_mask, not(text_protect_mask))

# 3. REFINE SECOND - Apply morphological operations
kernel = get_structuring_element(MORPH_ELLIPSE, (kernel_size, kernel_size))
opened = morphology_ex(protected_mask, MORPH_OPEN, kernel, iterations=1)
closed = morphology_ex(opened, MORPH_CLOSE, kernel, iterations=2)
```

**Automatic Mode** (no color specified):
```python
# 1. Create multiple detectors
binary = adaptive_threshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)
saturation_mask = (s_channel < saturation_threshold).astype(uint8) * 255

# 2. Combine detectors
combined_mask = bitwise_or(binary, saturation_mask)

# 3. Apply protection
protected_mask = protect_text_and_background(combined_mask)

# 4. Refine with morphology
final_mask = apply_morphological_operations(protected_mask)
```

### Recent Technical Improvements

**"Protect First, Refine Second" Fix (v0.5.3)**:
- **Problem**: Morphological operations caused mask to "bleed" into text areas
- **Solution**: Apply text/background protection BEFORE morphological operations
- **Result**: 59.6% reduction in text artifacts while maintaining detection accuracy

**Enhanced Text Protection**:
- Lower threshold (140 vs 150) to catch more anti-aliased edges
- Larger expansion (3 vs 2 pixels) for better text boundary coverage
- Removed counterproductive watermark mask refinement

## Development Guidelines

### Code Style Standards

**Python Conventions**:
- **PEP 8**: Follow standard Python formatting
- **Type Hints**: Use type annotations for function parameters and returns
- **Docstrings**: Use Google-style docstrings for all public functions
- **Error Handling**: Use specific exceptions with meaningful messages
- **Logging**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

**Project-Specific Conventions**:
- **Module Structure**: One class per module, descriptive filenames
- **CLI Design**: Use Click decorators, rich formatting for user interface
- **Image Processing**: Use NumPy arrays, OpenCV conventions (BGR→RGB conversion)
- **Configuration**: Command-line parameters override defaults, no config files
- **Internationalization**: Use `t()` function for all user-facing strings

**Naming Conventions**:
- **Classes**: PascalCase (e.g., `WatermarkDetector`)
- **Functions/Methods**: snake_case (e.g., `detect_watermark_mask`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_KERNEL_SIZE`)
- **Private Methods**: Prefix with underscore (e.g., `_refine_mask`)

### Testing Strategy

**Unit Tests** (`tests/test_watermark.py`):
- Test image creation with synthetic watermarks
- Test watermark detection accuracy
- Test mask refinement
- Test removal output consistency

**Integration Testing** (`tests/fix_validation/`):
- Use real PDFs with various watermark types
- Test with different parameter combinations
- Validate output PDF quality and size

**Performance Testing**:
- Measure processing time per page at different DPI settings
- Test memory usage with large documents
- Validate multi-threading safety

### Repository Structure Analysis

When analyzing the project structure, exclude the virtual environment directory to avoid cluttering the output:
```bash
# Exclude .venv when using tree command
tree /F /A | findstr /V "\.venv"

# Or use PowerShell to exclude multiple directories
tree /F /A | Where-Object { $_ -notmatch "\.venv|__pycache__|\.ruff_cache" }
```

## Dependencies & Build System

### Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Image processing, inpainting |
| numpy | ≥1.21.0 | Array operations, masking |
| pillow | ≥9.0.0 | Image I/O, PDF generation |
| pypdf | ≥3.0.0 | PDF utilities (metadata, etc.) |
| click | ≥8.0.0 | CLI framework |
| PyMuPDF | ≥1.23.0 | Fast PDF rendering |
| rich | ≥13.0.0 | Terminal UI and progress bars |
| ultralytics | ≥8.3.227 | YOLO detection models (optional) |

### Development Tools
- **UV**: Modern Python package manager and tool installer
- **Hatchling**: Build backend for package distribution
- **pytest**: Testing framework
- **ruff**: Fast Python linter and code formatter

## Extension Points

### Adding New Detection Methods
1. Create new detector class in separate module
2. Inherit interface from `WatermarkDetector`
3. Implement `detect_watermark_mask()` method
4. Add CLI option in `cli.py`
5. Update documentation and tests

### Adding New Inpainting Methods
1. Create new remover class or extend `WatermarkRemover`
2. Use different `cv2.inpaint()` algorithms:
   - `cv2.INPAINT_TELEA`: Current (good general purpose)
   - `cv2.INPAINT_NS`: Navier-Stokes (for texture)

### Adding New Output Formats
1. Extend `PDFProcessor`
2. Implement format-specific `images_to_*()` methods
3. Examples: images_to_tiff(), images_to_jpeg()

## Performance Considerations

### Factors Affecting Performance
- Image resolution (DPI)
- Watermark complexity (number of regions)
- Inpaint radius (larger = slower)
- Multi-pass count (linear scaling)
- System hardware (CPU/memory)

### Optimization Strategies
- Process large documents in chunks
- Use appropriate DPI for the task
- Optimize kernel sizes for specific watermark types
- Leverage document classification for parameter tuning

## Future Improvements

### Technical Enhancements
- **Machine Learning Integration**: Train custom models for specific watermark types
- **GPU Acceleration**: CUDA support for large batch processing
- **Real-time Processing**: Stream processing for live document workflows
- **Advanced Inpainting**: GAN-based or diffusion models for complex textures

### User Experience
- **GUI Interface**: Desktop application for non-technical users
- **Web Interface**: Browser-based processing with file upload
- **API Service**: RESTful API for integration with other systems
- **Mobile Support**: Lightweight version for mobile devices

This architecture provides a robust, extensible foundation for watermark removal while maintaining clean separation of concerns and supporting future enhancements.