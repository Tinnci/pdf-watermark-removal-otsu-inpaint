#!/usr/bin/env python3
"""Demo script showing the new QR code auto-detection and prompting feature."""

import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2
from PIL import Image
import qrcode

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_demo_pdf_with_qr_codes():
    """Create a demo PDF with QR codes for testing auto-detection."""
    # Create a document background
    image = np.ones((1200, 900, 3), dtype=np.uint8) * 255
    
    # Add some content
    cv2.putText(image, "Document with QR Codes", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image, "This document contains QR codes that should be auto-detected.", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Create QR codes using the qrcode library
    qr1 = qrcode.make("https://example.com/advertisement")
    qr2 = qrcode.make("https://company.com/legitimate-info")
    
    # Convert PIL to OpenCV format and resize
    qr1_cv = cv2.cvtColor(np.array(qr1.convert('RGB')), cv2.COLOR_RGB2BGR)
    qr2_cv = cv2.cvtColor(np.array(qr2.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    qr1_resized = cv2.resize(qr1_cv, (150, 150))
    qr2_resized = cv2.resize(qr2_cv, (120, 120))
    
    # Place QR codes on the document
    image[200:350, 50:200] = qr1_resized  # Advertisement QR
    image[400:520, 600:720] = qr2_resized  # Information QR
    
    # Add more content
    cv2.putText(image, "Content Section:", (50, 500), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "- Important document information", (70, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(image, "- QR codes detected automatically", (70, 590), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(image, "- User prompted for removal options", (70, 630), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Convert to PIL and save as PDF
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    pil_image.save(temp_pdf.name, 'PDF', resolution=150)
    
    return temp_pdf.name

def main():
    print("=== PDF Watermark Removal Tool - QR Code Auto-Detection Demo ===\n")
    
    # Create demo PDF
    print("Creating demo document with QR codes...")
    demo_pdf = create_demo_pdf_with_qr_codes()
    print(f"Demo PDF created: {demo_pdf}")
    
    print("\n" + "="*65)
    print("NEW AUTO-DETECTION FEATURE:")
    print("="*65)
    
    print("\n1. Automatic QR Code Detection:")
    print("   - CLI analyzes document without user specifying --detect-qr-codes")
    print("   - Detects QR codes automatically using OpenCV")
    print("   - Shows count of detected QR codes")
    
    print("\n2. Interactive Prompting:")
    print("   - When QR codes are found, prompts user:")
    print("\n   [Auto-detected] Found 2 potential QR code(s) in document.")
    print("   Enable QR code scanning and removal? [Y/n]:")
    
    print("\n3. Smart Defaults:")
    print("   - Default answer is 'Y' (Yes) to enable detection")
    print("   - If user accepts, enables full QR code processing")
    print("   - If user declines, shows how to enable later")
    
    print("\n4. Seamless Integration:")
    print("   - Works with existing workflow")
    print("   - No duplicate loading of pages")
    print("   - Compatible with other features (color selection, etc.)")
    
    print("\n" + "="*65)
    print("WHAT HAPPENS DURING PROCESSING:")
    print("="*65)
    
    print("\nStep 1: Document Analysis")
    print("   - Load first page for analysis")
    print("   - Run QR code detection algorithm")
    print("   - Count detected QR codes")
    
    print("\nStep 2: User Interaction (if QR codes found)")
    print("   - Show detection results")
    print("   - Prompt for user confirmation")
    print("   - Apply user choice")
    
    print("\nStep 3: Processing")
    print("   - If enabled: Full QR code analysis and removal")
    print("   - If disabled: Continue normal processing")
    print("   - Provide helpful feedback to user")
    
    print("\n" + "="*65)
    print("TO TEST THIS FEATURE:")
    print("="*65)
    
    print(f"\nRun: pdf-watermark-removal {demo_pdf} output.pdf")
    print("\nThe tool will automatically detect QR codes and prompt for removal!")
    print("\nCompare with manual mode: pdf-watermark-removal {demo_pdf} output2.pdf --detect-qr-codes")
    
    print("\n" + "="*65)
    print("KEY BENEFITS:")
    print("="*65)
    
    print("\n[OK] DISCOVERY - Users find QR code feature automatically")
    print("[OK] ZERO CONFIG - No need to remember command-line flags")
    print("[OK] USER FRIENDLY - Clear prompts with helpful information")
    print("[OK] BACKWARD COMPATIBLE - Existing workflows unchanged")
    print("[OK] EFFICIENT - No duplicate page loading or processing")
    print("[OK] INTUITIVE - Natural flow that guides users")

if __name__ == "__main__":
    # Check if qrcode library is available
    try:
        import qrcode
    except ImportError:
        print("This demo requires the 'qrcode' library.")
        print("Install with: pip install qrcode")
        sys.exit(1)
    
    main()