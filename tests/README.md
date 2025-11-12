# Tests Directory

This directory contains test scripts and validation tools for the PDF Watermark Removal project.

## Directory Structure

```
tests/
├── debug/                    # Debug scripts for development
│   ├── debug_actual_detector.py
│   ├── debug_protection.py
│   ├── debug_step_by_step.py
│   └── fix_text_protection.py
│
├── fix_validation/          # Scripts to validate the text protection fix
│   ├── test_protection_order.py
│   └── test_protection_improvement.py
│
└── test_watermark.py        # Original unit tests
```

## Debug Scripts

The `debug/` directory contains scripts used during development to understand and fix the text protection issue:

- **`debug_actual_detector.py`**: Step-by-step debugging of the actual detector implementation
- **`debug_protection.py`**: Analysis of the protection mechanism and mask creation
- **`debug_step_by_step.py`**: Manual recreation of the detection process
- **`fix_text_protection.py`**: Testing and validation of the improved text protection

## Fix Validation Scripts

The `fix_validation/` directory contains comprehensive tests to validate the "Protect First, Refine Second" approach:

- **`test_protection_order.py`**: Tests the core protection mechanism
- **`test_protection_improvement.py`**: Comprehensive evaluation across multiple scenarios

## Usage

### Running Debug Scripts
```bash
python tests/debug/debug_protection.py
```

### Running Validation Tests
```bash
python tests/fix_validation/test_protection_order.py
python tests/fix_validation/test_protection_improvement.py
```

### Running Original Unit Tests
```bash
python tests/test_watermark.py
```

## Test Results

Debug scripts generate visualization images that are saved to `docs/development/`:

- `debug_protection_steps.png` - Step-by-step protection process
- `manual_protection_steps.png` - Manual recreation of protection
- `test_protection_results.png` - Test results visualization
- Various color analysis and comparison images

## Purpose

These tests were created to validate the fix for the text "etching" issue where morphological operations were causing the watermark mask to bleed into text areas. The "Protect First, Refine Second" approach ensures that text and background protection is applied before any mask expansion operations.