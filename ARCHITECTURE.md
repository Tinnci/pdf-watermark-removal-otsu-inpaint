# Architecture & Technical Documentation

## Project Overview

**pdf-watermark-removal-otsu-inpaint** is a Python-based UV tool that automatically removes watermarks from PDF documents using advanced computer vision techniques: Otsu's automatic thresholding for watermark detection and OpenCV's inpainting algorithm for watermark removal.

## System Architecture

### High-Level Flow

```
PDF Input
    ↓
[PDF Processor] → Convert to Images
    ↓
[Watermark Detector] → Otsu Thresholding + Morphological Ops
    ↓
[Watermark Remover] → OpenCV Inpaint
    ↓
[PDF Processor] → Convert back to PDF
    ↓
PDF Output
```

## Core Components

### 1. PDF Processor (`pdf_processor.py`)

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

### 2. Watermark Detector (`watermark_detector.py`)

**Responsibility**: Identify watermark regions using image processing

**Algorithm - Otsu's Thresholding Method**:

1. **Grayscale Conversion**: Converts RGB image to grayscale
   - Input: RGB image (0-255 values)
   - Output: Grayscale image (0-255)

2. **Otsu Threshold (Otsu阈值分割)**:
   - Automatically calculates optimal threshold value
   - No manual parameter tuning needed
   - Separates foreground (watermark) from background (document)
   - Output: Binary image (0 or 255)

3. **Morphological Operations**:
   - Opening: Removes small noise and thin structures
     - Erode + Dilate with elliptical kernel
   - Closing: Fills small holes in watermark regions
     - Dilate + Erode with elliptical kernel
   - Kernel size: Configurable (default: 3x3)

4. **Connected Component Analysis**:
   - `refine_mask()`: Removes noise components
   - Filters by minimum area threshold
   - Only keeps significant watermark regions

**Key Methods**:
- `detect_watermark_mask(image_rgb)`: Returns binary mask of watermark regions
- `refine_mask(mask, min_area=100)`: Removes noise and small artifacts

### 3. Watermark Remover (`watermark_remover.py`)

**Responsibility**: Remove detected watermarks using inpainting

**Algorithm - OpenCV Inpainting**:

1. **Mask Preparation**:
   - Uses binary mask from WatermarkDetector
   - Refines mask to reduce noise

2. **Inpainting Process**:
   - Algorithm: Telea's method (`cv2.INPAINT_TELEA`)
   - Radius: Determines neighborhood size for reconstruction (default: 2)
   - Reconstructs removed regions using surrounding texture
   - Preserves document content and structure

3. **Multi-Pass Mode** (Optional):
   - Single pass: Standard one-shot inpainting
   - Multi-pass (2+): Iteratively applies removal
   - Benefits: Better for stubborn watermarks, but slower

**Key Methods**:
- `remove_watermark(image_rgb)`: Single-pass watermark removal
- `remove_watermark_multi_pass(image_rgb, passes=2)`: Iterative removal

### 4. CLI Interface (`cli.py`)

**Responsibility**: Command-line interface and user interaction

**Command Structure**:
```bash
pdf-watermark-removal INPUT_PDF OUTPUT_PDF [OPTIONS]
```

**Options**:
- `--kernel-size`: Morphological kernel size (1, 3, 5, 7, etc.)
- `--inpaint-radius`: Inpainting radius (1-5 typical)
- `--pages`: Selective page processing ("1,3,5" or "1-5")
- `--multi-pass`: Number of removal iterations
- `--dpi`: PDF rendering quality
- `--verbose`: Detailed logging
- `-v`: Shorthand for verbose

**Error Handling**:
- FileNotFoundError: Input PDF not found
- ImportError: Missing dependencies (PyMuPDF)
- Graceful error messages with exit codes

## Data Flow & Transformations

### Image Representations

The system uses NumPy arrays internally with specific formats:

1. **PDF → Image** (RGB, uint8):
   - Source: PDF page
   - Format: Numpy array [height, width, 3]
   - Values: 0-255

2. **RGB → Grayscale** (uint8):
   - Conversion: `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
   - Format: Numpy array [height, width]
   - Values: 0-255

3. **Grayscale → Binary Mask** (uint8):
   - Method: Otsu threshold
   - Format: Numpy array [height, width]
   - Values: 0 (not watermark) or 255 (watermark)

4. **Inpainted Image** (RGB, uint8):
   - Input: Original image + binary mask
   - Output: Watermark-free image
   - Format: Numpy array [height, width, 3]
   - Values: 0-255

5. **Images → PDF**:
   - Format: Pillow Image objects
   - Compression: Automatic
   - Pages: Concatenated into single PDF

## Parameter Tuning Guide

### Watermark Detection (Otsu Threshold)

**Kernel Size** (`--kernel-size`):
- Smaller (1-3): Better for thin watermarks, more noise
- Larger (5-7): Better for thick watermarks, fills small gaps
- Typical: 3 (default)

**Otsu Threshold**: Automatic (no tuning needed)

### Watermark Removal (Inpainting)

**Inpaint Radius** (`--inpaint-radius`):
- Smaller (1-2): Faster, preserves details (default: 2)
- Larger (3-5): Slower, smoother results
- Trade-off: Speed vs. quality

**Multi-Pass**:
- 1 pass: Fast (default)
- 2+ passes: Better results for stubborn watermarks
- Performance: Linear time increase with passes

### PDF Rendering

**DPI** (`--dpi`):
- 150 (default): Balanced speed/quality
- 300+: High quality, slower processing
- Lower: Faster, lower quality

## Performance Characteristics

### Memory Usage
- Per-page: ~(width × height × 3) × 2 (original + processed)
- Example: 2000×2500 RGB image ≈ 60 MB

### Processing Time (Typical)
- PDF to Images: ~100-500ms per page
- Watermark Detection: ~50-100ms per page
- Inpainting: ~200-500ms per page
- Images to PDF: ~100-300ms per image
- **Total**: ~500ms-1.5s per page

### Factors Affecting Performance
- Image resolution (DPI)
- Watermark complexity (number of regions)
- Inpaint radius (larger = slower)
- Multi-pass count (linear scaling)
- System hardware (CPU/memory)

## Dependencies & Justification

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Image processing, inpainting |
| numpy | ≥1.21.0 | Array operations, masking |
| pillow | ≥9.0.0 | Image I/O, PDF generation |
| pypdf | ≥3.0.0 | PDF utilities (metadata, etc.) |
| click | ≥8.0.0 | CLI framework |
| PyMuPDF | ≥1.23.0 | Fast PDF rendering |

## Extension Points

### Adding New Detection Methods
1. Create new detector class in separate module
2. Inherit interface from `WatermarkDetector`
3. Implement `detect_watermark_mask()` method

### Adding New Inpainting Methods
1. Create new remover class or extend `WatermarkRemover`
2. Use different `cv2.inpaint()` algorithms:
   - `cv2.INPAINT_TELEA`: Current (good general purpose)
   - `cv2.INPAINT_NS`: Navier-Stokes (for texture)

### Adding New Output Formats
1. Extend `PDFProcessor`
2. Implement format-specific `images_to_*()` methods
3. Examples: images_to_tiff(), images_to_jpeg()

## Testing Strategy

### Unit Tests (`test_watermark.py`)
- Test image creation with synthetic watermarks
- Test watermark detection accuracy
- Test mask refinement
- Test removal output consistency

### Integration Testing (Manual)
1. Use real PDFs with various watermark types
2. Test with different parameter combinations
3. Validate output PDF quality and size

## Future Improvements

1. **Advanced Detection**:
   - Deep learning-based watermark localization
   - Adaptive threshold based on image content
   - Support for transparent watermarks

2. **Performance**:
   - GPU acceleration with CUDA
   - Batch processing optimization
   - Parallel multi-page processing

3. **Features**:
   - Watermark content preservation (extraction)
   - Batch mode with progress tracking
   - Web API interface
   - GUI application

4. **Quality**:
   - Better artifact removal post-inpainting
   - Content-aware fill improvements
   - Machine learning-based restoration
