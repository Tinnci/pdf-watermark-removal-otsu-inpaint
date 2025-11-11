# UV Tool Configuration

This document explains how this project is configured as a UV tool.

## Project Structure for UV Tools

```
pdf-watermark-removal-otsu-inpaint/
├── pyproject.toml                 # Project metadata and tool configuration
├── src/
│   └── pdf_watermark_removal/     # Package source
│       ├── __init__.py
│       ├── cli.py                 # CLI entry point
│       ├── pdf_processor.py
│       ├── watermark_detector.py
│       └── watermark_remover.py
├── README.md
├── INSTALL.md
├── ARCHITECTURE.md
└── examples.py
```

## pyproject.toml Configuration

### Project Metadata

```toml
[project]
name = "pdf-watermark-removal-otsu-inpaint"
version = "0.1.0"
description = "Remove watermarks from PDF using Otsu threshold segmentation and OpenCV inpaint"
requires-python = ">=3.8"
```

### Console Entry Point (CLI Tool)

```toml
[project.scripts]
pdf-watermark-removal = "pdf_watermark_removal.cli:main"
```

This makes the tool available as `pdf-watermark-removal` command after installation.

### Dependencies

All dependencies are automatically resolved and installed by UV:

```toml
dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.21.0",
    "pillow>=9.0.0",
    "pypdf>=3.0.0",
    "click>=8.0.0",
    "PyMuPDF>=1.23.0",
]
```

### Build System

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Uses Hatchling for package building (minimal and fast).

## UV Installation Methods

### Method 1: Tool Installation (Recommended for End Users)

```bash
# Install from PyPI (when published)
uv tool install pdf-watermark-removal-otsu-inpaint

# Install from local directory (during development)
uv tool install --editable .

# Run the command
pdf-watermark-removal input.pdf output.pdf
```

**Characteristics**:
- Creates isolated virtual environment in `~/.local/share/uv/tools/`
- Executables available on PATH globally
- Survives across sessions
- Best for production use

### Method 2: Tool Execution (No Installation)

```bash
# Run from PyPI without installing
uvx pdf-watermark-removal-otsu-inpaint input.pdf output.pdf

# Run from local directory
uvx -e . pdf-watermark-removal input.pdf output.pdf
```

**Characteristics**:
- Temporary environment (cached, reused)
- No global PATH modification
- Latest version by default
- Good for one-off usage

### Method 3: Development Virtual Environment

```bash
# Create local virtual environment
uv venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Unix)
source .venv/bin/activate

# Install in development mode
uv pip install -e .

# Use normally
pdf-watermark-removal input.pdf output.pdf
```

**Characteristics**:
- Local development environment
- Full control over dependencies
- Easy debugging and testing
- Best for contributors

## UV Tool Versioning

### Version Management

UV respects the version specified in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

### Updating Version for Release

1. Update `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. Reinstall if using tool install:
   ```bash
   uv tool install --force --editable .
   ```

### Version-Specific Installation

```bash
# Install specific version from PyPI
uv tool install pdf-watermark-removal-otsu-inpaint==0.1.0

# Use latest version explicitly
uv tool install pdf-watermark-removal-otsu-inpaint@latest
```

## Dependency Management with UV

### Adding Dependencies

1. Update `pyproject.toml`:
   ```toml
   dependencies = [
       "existing-package>=1.0",
       "new-package>=2.0",  # Add here
   ]
   ```

2. Reinstall:
   ```bash
   # For tool installation
   uv tool upgrade pdf-watermark-removal-otsu-inpaint

   # For development venv
   uv pip install -e .
   ```

### Development Dependencies

Defined separately in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

Install with:
```bash
uv pip install -e ".[dev]"
```

### Transitive Dependencies

UV automatically resolves and manages transitive dependencies:
- OpenCV depends on NumPy → NumPy auto-installed
- PyMuPDF → All its dependencies auto-installed
- No manual dependency management needed

## Python Version Selection

UV uses the default Python version by default, but you can specify:

```toml
[project]
requires-python = ">=3.8"
```

### Tool-Specific Python Version

Install with specific Python version:

```bash
# Use Python 3.11
uv tool install --python 3.11 --editable .

# Check Python version used
python --version
```

## Updating Tools

### Upgrade Tool and Dependencies

```bash
# Upgrade all dependencies in tool environment
uv tool upgrade pdf-watermark-removal-otsu-inpaint

# Upgrade specific package in tool environment
uv tool upgrade pdf-watermark-removal-otsu-inpaint --upgrade-package opencv-python
```

## Troubleshooting UV Tool Issues

### Tool Not Found

```bash
# Check installed tools
uv tool list

# Check if bin directory is in PATH
echo $PATH  # Unix
echo %PATH%  # Windows
```

### Dependency Conflicts

```bash
# Reinstall with clean environment
uv tool uninstall pdf-watermark-removal-otsu-inpaint
uv tool install --editable .
```

### Environment Issues

```bash
# Clean UV cache
uv cache clean

# Reinstall fresh
uv tool install --force --editable .
```

## Publishing to PyPI

To make this tool available on PyPI:

1. Build the package:
   ```bash
   uv build
   ```

2. Upload to PyPI:
   ```bash
   # Using twine
   pip install twine
   twine upload dist/*
   ```

3. Users can then install with:
   ```bash
   uv tool install pdf-watermark-removal-otsu-inpaint
   ```

## UV Lock File (Future)

When UV's lock file feature is available, generate with:

```bash
uv lock
```

This creates `uv.lock` for reproducible installations across different machines.

## Best Practices for UV Tools

1. **Keep dependencies minimal**: Reduces installation time and size
2. **Pin major versions**: Prevents breaking changes
3. **Test with uvx**: Verify tool works in isolated environment
4. **Document Python version**: Specify minimum required Python
5. **Use console_scripts**: For clean command-line interface
6. **Add version info**: Help users verify installed version
