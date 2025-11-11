"""PDF processing utilities."""

import io
from pathlib import Path

import numpy as np
from PIL import Image


class PDFProcessor:
    """Handles PDF to image conversion and reconstruction."""

    def __init__(self, dpi=150, verbose=False):
        """Initialize PDF processor.

        Args:
            dpi: DPI for PDF to image conversion
            verbose: Enable verbose logging
        """
        self.dpi = dpi
        self.verbose = verbose

    def pdf_to_images(self, pdf_path, pages=None):
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to input PDF
            pages: List of page numbers to convert (1-indexed), or None for all

        Returns:
            List of images as RGB numpy arrays
        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. Install it with: pip install PyMuPDF"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        images = []

        total_pages = len(doc)
        if pages is None:
            pages = list(range(1, total_pages + 1))

        if self.verbose:
            print(f"Converting {len(pages)} pages from PDF...")

        for page_num in pages:
            if page_num < 1 or page_num > total_pages:
                if self.verbose:
                    print(f"Skipping invalid page {page_num}")
                continue

            if self.verbose:
                print(f"  Processing page {page_num}/{total_pages}...")

            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72))
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            # PIL Image.open returns RGB format, no color conversion needed
            img_rgb = np.array(img)
            images.append(img_rgb)

        doc.close()
        return images

    def page_to_image(self, pdf_path, page_num):
        """Convert a single PDF page to image (memory-efficient).

        Args:
            pdf_path: Path to input PDF
            page_num: Page number (1-indexed)

        Returns:
            Image as RGB numpy array
        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. Install it with: pip install PyMuPDF"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if page_num < 1 or page_num > total_pages:
            doc.close()
            raise ValueError(
                f"Invalid page number {page_num}. PDF has {total_pages} pages."
            )

        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72))
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        img_rgb = np.array(img)

        doc.close()
        return img_rgb

    def images_to_pdf(self, images, output_path):
        """Convert images back to PDF.

        Args:
            images: List of images as RGB numpy arrays
            output_path: Path for output PDF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Converting {len(images)} images to PDF...")

        pil_images = []
        for i, img_rgb in enumerate(images):
            if self.verbose:
                print(f"  Processing image {i + 1}/{len(images)}...")

            pil_img = Image.fromarray(img_rgb)
            pil_images.append(pil_img)

        if pil_images:
            pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:])

        if self.verbose:
            print(f"PDF saved to: {output_path}")
