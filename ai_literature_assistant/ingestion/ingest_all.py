# ingestion/ingest_all.py

import os
import sys
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .pdf_loader import extract_text_from_pdf, list_pdfs, batch_extract
from .preprocess import ResearchPaperChunker
from .embed_store import store_documents_enhanced

logger = logging.getLogger(__name__)


PDF_DIR = Path(
    os.getenv(
        "PDF_DIR",
        str(Path(__file__).parent.parent / "data" / "pdfs"),
    )
)


def validate_environment() -> List[Path]:
    """Validate PDF directory and return list of PDF paths."""
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"[INGEST] PDF directory not found: {PDF_DIR}")

    pdf_files = list_pdfs(str(PDF_DIR))

    if not pdf_files:
        raise ValueError(f"[INGEST] No PDF files found in {PDF_DIR}")

    return [Path(p) for p in pdf_files]


def process_single_pdf(
    pdf_path: Path,
    chunker: ResearchPaperChunker,
) -> Tuple[List[Any], List[str]]:
   
    try:
        text = extract_text_from_pdf(str(pdf_path))

        if not text or len(text.strip()) < 100:
            logger.warning(f"[INGEST] Skipping {pdf_path.name}: insufficient text")
            return [], []

        paper_id = pdf_path.stem
        chunks = chunker.chunk_text(text, paper_id)

        # Use the chunk_id already set by ResearchPaperChunker (no duplication)
        chunk_ids = [c.chunk_id for c in chunks]

        return chunks, chunk_ids

    except Exception as e:
        logger.error(f"[INGEST] Failed to process {pdf_path.name}: {e}")
        return [], []


def ingest_pdfs_enhanced(
    collection_name: str = "research_papers",
    chunk_size: int = 1000,
    overlap: int = 150,
    max_workers: int = 4,
) -> Dict[str, Any]:
   
    try:
        pdf_files = validate_environment()
    except Exception as e:
        logger.error(f"[INGEST] {e}")
        return {"error": str(e)}

    logger.info("=" * 60)
    logger.info(f"[INGEST] Starting ingestion of {len(pdf_files)} PDF(s)")
    logger.info("=" * 60)

    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)

    all_chunks: List[Any] = []
    all_ids: List[str] = []
    processed = 0
    failed = 0

   
    logger.info(f"[INGEST] Extracting text with {max_workers} parallel workers...")
    extraction_results = batch_extract(str(PDF_DIR), max_workers=max_workers)


    text_by_path = {r["path"]: r["text"] for r in extraction_results}

    for pdf_path in pdf_files:
        raw_text = text_by_path.get(str(pdf_path))

        if not raw_text or len(raw_text.strip()) < 100:
            logger.warning(f"[INGEST] Skipping {pdf_path.name}: insufficient text")
            failed += 1
            continue

        try:
            paper_id = pdf_path.stem
            chunks = chunker.chunk_text(raw_text, paper_id)

            if not chunks:
                logger.warning(f"[INGEST] No chunks from {pdf_path.name}")
                failed += 1
                continue

            chunk_ids = [c.chunk_id for c in chunks]

            all_chunks.extend(chunks)
            all_ids.extend(chunk_ids)
            processed += 1
            logger.info(f"[INGEST] ✔ {pdf_path.name} → {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"[INGEST] Chunking failed for {pdf_path.name}: {e}")
            failed += 1

    if not all_chunks:
        logger.error("[INGEST] No chunks generated from any PDF.")
        return {"error": "No chunks generated"}

    logger.info(f"[INGEST] Storing {len(all_chunks)} chunks in ChromaDB...")
    store_documents_enhanced(
        chunks=all_chunks,
        ids=all_ids,
        collection_name=collection_name,
    )

    stats = {
        "total_pdfs": len(pdf_files),
        "processed_pdfs": processed,
        "failed_pdfs": failed,
        "total_chunks": len(all_chunks),
        "avg_chunks_per_pdf": round(len(all_chunks) / processed, 1) if processed else 0,
    }

    logger.info("[INGEST] Ingestion Summary")
    logger.info("-" * 60)
    for k, v in stats.items():
        logger.info(f"[INGEST]   {k.replace('_', ' ').title()}: {v}")

    return stats         


# ── Backward compatibility ─────────────────────────────────────────────────

def ingest_pdfs() -> None:
    """Legacy simple ingestion (no metadata, no parallel processing)."""
    from .preprocess import chunk_text
    from .embed_store import store_documents

    all_chunks: List[str] = []
    ids: List[str] = []
    idx = 0

    for pdf in list_pdfs(str(PDF_DIR)):
        logger.info(f"[INGEST] Processing: {Path(pdf).name}")
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text, chunk_size=800, overlap=120)
        for chunk in chunks:
            all_chunks.append(chunk)
            ids.append(f"legacy_{idx:04d}")
            idx += 1

    store_documents(all_chunks, ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 60)
    logger.info("AI Research Literature Assistant — Ingestion Pipeline")
    logger.info("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        logger.info("Running legacy ingestion...")
        ingest_pdfs()
    else:
        result = ingest_pdfs_enhanced()
        if "error" in result:
            sys.exit(1)