# ingestion/image_extractor.py
"""
Image extraction and analysis pipeline for PDF documents.
Extracts embedded images from PDFs, analyzes them with GPT-4o Vision,
and creates searchable text chunks from the image descriptions.
"""

import os
import io
import time
import base64
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from PIL import Image
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
IMAGES_DIR = os.path.abspath(IMAGES_DIR)
os.makedirs(IMAGES_DIR, exist_ok=True)


MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_BYTES = 5000 


_VISION_CLIENT: Optional[OpenAI] = None


def _get_vision_client() -> Optional[OpenAI]:
    """Return (or lazily create) the shared OpenAI client."""
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _VISION_CLIENT = OpenAI(api_key=api_key)
        else:
            logger.warning("[IMAGE_EXTRACTOR] OPENAI_API_KEY not set — vision disabled.")
    return _VISION_CLIENT


@dataclass
class ExtractedImage:
    """Represents an image extracted from a PDF."""
    image_bytes: bytes
    page_num: int
    width: int
    height: int
    image_format: str
    image_hash: str
    saved_path: str = ""


def extract_images_from_pdf(
    pdf_source,
    paper_id: str = "",
) -> List[ExtractedImage]:
    
    try:
        if isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
        else:
            reader = PdfReader(io.BytesIO(pdf_source))
    except Exception as e:
        logger.error(f"[IMAGE_EXTRACTOR] Failed to read PDF: {e}")
        return []

    paper_img_dir = os.path.join(IMAGES_DIR, paper_id) if paper_id else IMAGES_DIR
    os.makedirs(paper_img_dir, exist_ok=True)

    extracted_images: List[ExtractedImage] = []
    seen_hashes: set = set()

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            if not hasattr(page, "images") or not page.images:
                continue

            for img_idx, image in enumerate(page.images):
                try:
                    img_bytes = image.data

                    if len(img_bytes) < MIN_IMAGE_BYTES:
                        continue

                    # Deduplicate identical images (e.g. repeated header logos)
                    img_hash = hashlib.md5(img_bytes).hexdigest()[:10]
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        width, height = pil_img.size
                    except Exception:
                        continue

                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue

                    # Normalise to PNG (handles CMYK, palette modes, etc.)
                    png_buffer = io.BytesIO()
                    if pil_img.mode not in ("RGB", "RGBA"):
                        pil_img = pil_img.convert("RGB")
                    pil_img.save(png_buffer, format="PNG", optimize=True)
                    png_bytes = png_buffer.getvalue()

                    img_filename = f"page{page_num}_img{img_idx}_{img_hash}.png"
                    img_path = os.path.join(paper_img_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(png_bytes)

                    extracted_images.append(ExtractedImage(
                        image_bytes=png_bytes,
                        page_num=page_num,
                        width=width,
                        height=height,
                        image_format="PNG",
                        image_hash=img_hash,
                        saved_path=img_path,
                    ))

                    logger.info(
                        f"[IMAGE_EXTRACTOR] Extracted image: "
                        f"page {page_num}, {width}×{height}"
                    )

                except Exception as e:
                    logger.warning(
                        f"[IMAGE_EXTRACTOR] Failed to extract image {img_idx} "
                        f"from page {page_num}: {e}"
                    )

        except Exception as e:
            logger.warning(
                f"[IMAGE_EXTRACTOR] Failed to process page {page_num}: {e}"
            )

    logger.info(
        f"[IMAGE_EXTRACTOR] Extracted {len(extracted_images)} images "
        f"from PDF '{paper_id}'"
    )
    return extracted_images


def analyze_image_with_vision(
    image_bytes: bytes,
    page_num: int = 0,
    context: str = "",
    model: str = "gpt-4o-mini",
) -> str:
    
    client = _get_vision_client()
    if client is None:
        return "Image analysis unavailable (no API key)"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an expert academic research assistant analysing figures, charts, "
        "tables, and diagrams from research papers. Provide a detailed, structured "
        "description that captures ALL information in the image, including:\n"
        "1. Type of visual (chart, table, diagram, photograph, etc.)\n"
        "2. Title/caption if visible\n"
        "3. All data points, labels, axes, legends\n"
        "4. Key findings or trends shown\n"
        "5. Any numerical values visible\n"
        "Be thorough — your description will be used to answer questions about this image."
    )

    user_text = (
        f"Analyse this image from page {page_num} of a research paper. "
        + (f"Context: {context} " if context else "")
        + "Provide a comprehensive description."
    )

   
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        },
    ]

  
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,          
            )
            description = response.choices[0].message.content
            logger.info(
                f"[IMAGE_EXTRACTOR] Image analysed successfully "
                f"({len(description)} chars)"
            )
            return description

        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt   # 1 s, then 2 s
                logger.warning(
                    f"[IMAGE_EXTRACTOR] Vision API error (attempt {attempt + 1}/3), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[IMAGE_EXTRACTOR] Vision API failed after 3 attempts: {e}"
                )
                return f"Image on page {page_num} (analysis unavailable)"

    return f"Image on page {page_num} (analysis unavailable)"


def process_pdf_images(
    pdf_source,
    paper_id: str = "",
    max_images: int = 15,
) -> List[Dict[str, Any]]:

    images = extract_images_from_pdf(pdf_source, paper_id)

    if not images:
        logger.info(f"[IMAGE_EXTRACTOR] No meaningful images in PDF '{paper_id}'")
        return []

    # FIX 4: keep first N by page order — do NOT sort by size
    if len(images) > max_images:
        logger.info(
            f"[IMAGE_EXTRACTOR] Capping at {max_images} of {len(images)} images "
            f"(chronological order)."
        )
        images = images[:max_images]

    image_chunks: List[Dict[str, Any]] = []

    for i, img in enumerate(images):
        logger.info(
            f"[IMAGE_EXTRACTOR] Analysing image {i + 1}/{len(images)} "
            f"(page {img.page_num}, {img.width}×{img.height})..."
        )

        description = analyze_image_with_vision(
            image_bytes=img.image_bytes,
            page_num=img.page_num,
        )

        if not description or len(description) < 20:
            continue

        chunk_text = f"[Figure from page {img.page_num}]\n{description}"

        image_chunks.append({
            "text": chunk_text,
            "metadata": {
                "paper_id": paper_id,
                "chunk_type": "image_description",
                "is_metadata": False,
                "section": "figure",
                "page_start": img.page_num,
                "page_end": img.page_num,
                "image_path": img.saved_path,
                "image_width": img.width,
                "image_height": img.height,
                "image_hash": img.image_hash,
            },
        })

    logger.info(
        f"[IMAGE_EXTRACTOR] Created {len(image_chunks)} image chunks "
        f"for paper '{paper_id}'"
    )
    return image_chunks