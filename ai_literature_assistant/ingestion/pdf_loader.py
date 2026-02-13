# ingestion/pdf_loader.py

import io
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union, List, Dict, TypedDict
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)

MAX_PDF_SIZE_MB = 100


class PageResult(TypedDict):
    source: str
    page_num: int
    total_pages: int
    text: str


def list_pdfs(pdf_dir: str) -> List[str]:
    """Lists all PDF files in a directory, sorted for deterministic ordering."""
    pdfs = sorted([
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ])
    logger.info(f"[PDF_LOADER] Found {len(pdfs)} PDF file(s).")
    return pdfs


def clean_extracted_text(text: str) -> str:
    """Clean raw PDF-extracted text for better downstream processing."""
    if not text:
        return ""

    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

    ligature_map = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2026': '...',
    }
    for lig, replacement in ligature_map.items():
        text = text.replace(lig, replacement)

    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s*[\d#*|]\s*\n', '\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def _validate_pdf_source(pdf_source: Union[str, bytes]) -> None:
    if isinstance(pdf_source, str):
        if not os.path.isfile(pdf_source):
            raise FileNotFoundError(f"[PDF_LOADER] PDF not found: {pdf_source}")
        size_mb = os.path.getsize(pdf_source) / 1024 / 1024
        if size_mb > MAX_PDF_SIZE_MB:
            raise ValueError(
                f"[PDF_LOADER] PDF too large: {size_mb:.1f}MB (limit: {MAX_PDF_SIZE_MB}MB)"
            )
    elif isinstance(pdf_source, bytes):
        size_mb = len(pdf_source) / 1024 / 1024
        if size_mb > MAX_PDF_SIZE_MB:
            raise ValueError(
                f"[PDF_LOADER] PDF bytes too large: {size_mb:.1f}MB (limit: {MAX_PDF_SIZE_MB}MB)"
            )


def _extract_page_text(page, page_num: int) -> str:
    text = page.extract_text(extraction_mode="layout") or ""
    if not text.strip():
        text = page.extract_text() or ""
    return clean_extracted_text(text)


def extract_text_from_pdf(pdf_source: Union[str, bytes]) -> str:
    _validate_pdf_source(pdf_source)

    try:
        if isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
            pdf_name = os.path.basename(pdf_source)
        else:
            reader = PdfReader(io.BytesIO(pdf_source))
            pdf_name = "bytes_input"

        num_pages = len(reader.pages)
        logger.info(f"[PDF_LOADER] Processing: {pdf_name} | Pages: {num_pages}")

        pages_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = _extract_page_text(page, page_num)

                if text.strip():
                    pages_text.append(f"[PAGE {page_num}]\n{text}")

            except Exception as e:
                logger.warning(f"[PDF_LOADER] Page {page_num} extraction failed: {e}")
                continue

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ValueError("No text extracted — PDF may be scanned or corrupted.")

        logger.info(
            f"[PDF_LOADER] Extracted {len(full_text)} characters from {pdf_name}"
        )

        return full_text

    except Exception as e:
        logger.error(f"[PDF_LOADER] Error processing PDF: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")


def extract_text_with_pages(pdf_source: Union[str, bytes]) -> List[PageResult]:
    _validate_pdf_source(pdf_source)

    try:
        if isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
            source_name = os.path.basename(pdf_source)
        else:
            reader = PdfReader(io.BytesIO(pdf_source))
            source_name = "bytes_input"

        total_pages = len(reader.pages)
        pages: List[PageResult] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = _extract_page_text(page, page_num)

                if text.strip():
                    pages.append(PageResult(
                        source=source_name,
                        page_num=page_num,
                        total_pages=total_pages,
                        text=text,
                    ))

            except Exception as e:
                logger.warning(f"[PDF_LOADER] Skipped page {page_num} in {source_name}: {e}")
                continue

        return pages

    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


def batch_extract(pdf_dir: str, max_workers: int = 4) -> List[Dict]:
    paths = list_pdfs(pdf_dir)
    if not paths:
        logger.warning(f"[PDF_LOADER] No PDFs found in: {pdf_dir}")
        return []

    logger.info(f"[PDF_LOADER] Batch extracting {len(paths)} PDFs with {max_workers} workers.")
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_text_from_pdf, p): p for p in paths}

        for future in as_completed(futures):
            path = futures[future]
            try:
                text = future.result()
                results.append({"path": path, "text": text})
                logger.info(f"[PDF_LOADER] ✓ Completed: {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"[PDF_LOADER] ✗ Failed: {os.path.basename(path)}: {e}")

    logger.info(f"[PDF_LOADER] Batch done: {len(results)}/{len(paths)} succeeded.")
    return results