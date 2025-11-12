#!/usr/bin/env python3
"""Comprehensive test of the text protection improvement."""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.pdf_watermark_removal.watermark_detector import WatermarkDetector

def create_test_cases():
    """Create various test cases to evaluate text protection."""
    test_cases = []
    
    # Test Case 1: Original problem - light gray watermark with black text
    print("Creating Test Case 1: Light gray watermark with black text...")
    image1 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(image1, (50, 50), (350, 150), (200, 200, 200), -1)  # Light gray watermark
    cv2.putText(image1, "PROTECT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(image1, "TEXT", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    test_cases.append(("Light gray watermark + Black text", image1, (200, 200, 200)))
    
    # Test Case 2: Darker watermark with gray text (more challenging)
    print("Creating Test Case 2: Darker watermark with gray text...")
    image2 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(image2, (50, 50), (350, 150), (180, 180, 180), -1)  # Darker watermark
    cv2.putText(image2, "GRAY", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3)  # Gray text
    cv2.putText(image2, "TEXT", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3)
    test_cases.append(("Darker watermark + Gray text", image2, (180, 180, 180)))
    
    # Test Case 3: Electronic document scenario (very light watermark)
    print("Creating Test Case 3: Electronic document scenario...")
    image3 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(image3, (50, 50), (350, 150), (210, 210, 210), -1)  # Very light watermark
    cv2.putText(image3, "ELECTRONIC", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image3, "DOC", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    test_cases.append(("Electronic document", image3, (210, 210, 210)))
    
    # Test Case 4: Scanned document scenario (noisy background)
    print("Creating Test Case 4: Scanned document scenario...")
    image4 = np.ones((200, 400, 3), dtype=np.uint8) * 245  # Off-white background
    # Add some noise to simulate scanned document
    noise = np.random.normal(0, 5, image4.shape).astype(np.int16)
    image4 = np.clip(image4.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.rectangle(image4, (50, 50), (350, 150), (190, 190, 190), -1)  # Watermark
    cv2.putText(image4, "SCANNED", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    cv2.putText(image4, "TEXT", (130, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    test_cases.append(("Scanned document", image4, (190, 190, 190)))
    
    return test_cases

def evaluate_text_protection(image, mask, text_region=None):
    """Evaluate how well text is protected in a specific region."""
    if text_region is None:
        # Default text region (approximate area where text is placed)
        text_region = mask[80:120, 100:300]  # Central area where text typically is
    else:
        text_region = mask[text_region[1]:text_region[3], text_region[0]:text_region[2]]
    
    text_pixels = np.count_nonzero(text_region)
    total_text_area = text_region.size
    text_percentage = (text_pixels / total_text_area) * 100
    
    # Quality assessment
    if text_pixels < 100:  # Very few text pixels in mask
        quality = "EXCELLENT"
    elif text_pixels < 500:  # Some text pixels, but acceptable
        quality = "GOOD"
    elif text_pixels < 1500:  # Moderate text protection
        quality = "FAIR"
    else:  # Poor text protection
        quality = "POOR"
    
    return {
        "text_pixels": text_pixels,
        "text_percentage": text_percentage,
        "quality": quality,
        "total_mask_pixels": np.count_nonzero(mask),
        "mask_coverage": (np.count_nonzero(mask) / mask.size) * 100
    }

def test_protection_improvement():
    """Test the improvement in text protection."""
    print("TESTING TEXT PROTECTION IMPROVEMENT")
    print("=" * 60)
    
    test_cases = create_test_cases()
    results = []
    
    for case_name, image, watermark_color in test_cases:
        print(f"\nTesting: {case_name}")
        print("-" * 40)
        
        # Test with color-based detection (where our fix applies)
        detector = WatermarkDetector(
            detection_method="traditional",
            watermark_color=watermark_color,
            color_tolerance=30,
            kernel_size=3,
            protect_text=True,
            verbose=False  # Reduce noise
        )
        
        mask = detector.detect_watermark_mask(image)
        evaluation = evaluate_text_protection(image, mask)
        
        print(f"  Total mask pixels: {evaluation['total_mask_pixels']:,}")
        print(f"  Mask coverage: {evaluation['mask_coverage']:.1f}%")
        print(f"  Text pixels in mask: {evaluation['text_pixels']:,}")
        print(f"  Text area coverage: {evaluation['text_percentage']:.1f}%")
        print(f"  Protection quality: {evaluation['quality']}")
        
        results.append({
            'case_name': case_name,
            'evaluation': evaluation
        })
    
    return results

def compare_with_baseline():
    """Compare with what would happen without text protection."""
    print("\n" + "=" * 60)
    print("COMPARING WITH NO PROTECTION BASELINE")
    print("=" * 60)
    
    # Create a simple test case
    image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (50, 50), (350, 150), (200, 200, 200), -1)
    cv2.putText(image, "TEST", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    print("\nScenario: Light gray watermark with black text")
    
    # Test WITH text protection
    detector_with = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=30,
        protect_text=True,
        verbose=False
    )
    
    mask_with = detector_with.detect_watermark_mask(image)
    eval_with = evaluate_text_protection(image, mask_with)
    
    # Test WITHOUT text protection
    detector_without = WatermarkDetector(
        detection_method="traditional",
        watermark_color=(200, 200, 200),
        color_tolerance=30,
        protect_text=False,
        verbose=False
    )
    
    mask_without = detector_without.detect_watermark_mask(image)
    eval_without = evaluate_text_protection(image, mask_without)
    
    print(f"\nWITH text protection:")
    print(f"  Mask pixels: {eval_with['total_mask_pixels']:,}")
    print(f"  Text pixels in mask: {eval_with['text_pixels']:,}")
    print(f"  Protection quality: {eval_with['quality']}")
    
    print(f"\nWITHOUT text protection:")
    print(f"  Mask pixels: {eval_without['total_mask_pixels']:,}")
    print(f"  Text pixels in mask: {eval_without['text_pixels']:,}")
    print(f"  Protection quality: {eval_without['quality']}")
    
    improvement = eval_without['text_pixels'] - eval_with['text_pixels']
    improvement_pct = (improvement / eval_without['text_pixels']) * 100 if eval_without['text_pixels'] > 0 else 0
    
    print(f"\nIMPROVEMENT:")
    print(f"  Text pixels removed: {improvement:,}")
    print(f"  Improvement percentage: {improvement_pct:.1f}%")
    print(f"  Protection effectiveness: {'EXCELLENT' if improvement_pct > 50 else 'GOOD' if improvement_pct > 25 else 'FAIR'}")
    
    return improvement_pct

def create_summary_report(results):
    """Create a summary report of all tests."""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\nTested {len(results)} scenarios:")
    
    for i, result in enumerate(results, 1):
        eval_data = result['evaluation']
        print(f"\n{i}. {result['case_name']}:")
        print(f"   Mask coverage: {eval_data['mask_coverage']:.1f}%")
        print(f"   Text protection: {eval_data['quality']} ({eval_data['text_pixels']} pixels)")
    
    # Overall assessment
    excellent_count = sum(1 for r in results if r['evaluation']['quality'] == 'EXCELLENT')
    good_count = sum(1 for r in results if r['evaluation']['quality'] == 'GOOD')
    fair_count = sum(1 for r in results if r['evaluation']['quality'] == 'FAIR')
    poor_count = sum(1 for r in results if r['evaluation']['quality'] == 'POOR')
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Excellent protection: {excellent_count}")
    print(f"  Good protection: {good_count}")
    print(f"  Fair protection: {fair_count}")
    print(f"  Poor protection: {poor_count}")
    
    if excellent_count + good_count >= len(results) * 0.7:
        overall_quality = "EXCELLENT"
    elif excellent_count + good_count >= len(results) * 0.5:
        overall_quality = "GOOD"
    else:
        overall_quality = "NEEDS IMPROVEMENT"
    
    print(f"  Overall text protection: {overall_quality}")

if __name__ == "__main__":
    # Run comprehensive tests
    results = test_protection_improvement()
    improvement_pct = compare_with_baseline()
    create_summary_report(results)
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION:")
    print(f"The 'Protect First, Refine Second' approach with improved")
    print(f"text protection achieves {improvement_pct:.1f}% reduction in text")
    print(f"pixels being included in the watermark mask.")
    print("=" * 60)