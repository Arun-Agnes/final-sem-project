# ingestion/preprocess.py

import os
import re
import uuid
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class PaperMetadata:
    """Structured metadata for a research paper."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    journal_conference: str = ""
    doi: str = ""


@dataclass
class TextChunk:
    """A single text chunk with associated metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str = ""


# ─────────────────────────────────────────────────────────────
# LLM Title Extraction (module-level, imports resolved once)
# ─────────────────────────────────────────────────────────────

try:
    from openai import AsyncOpenAI, OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from config import OPENAI_MODEL
except ImportError:
    OPENAI_MODEL = "gpt-4o-mini"


def _extract_title_with_llm(text_preview: str) -> str:
    
    if not _OPENAI_AVAILABLE:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the EXACT title of the research paper from the first-page text. "
                        "Return ONLY the title. Ignore journal names, volume numbers, or headers. "
                        "Return nothing if no clear paper title is found."
                    ),
                },
                {"role": "user", "content": text_preview},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        title = response.choices[0].message.content.strip()
        return title if len(title) <= 300 else ""

    except Exception as e:
        logger.warning(f"[PREPROCESS] LLM title extraction failed: {e}")
        return ""


async def _extract_title_with_llm_async(text_preview: str) -> str:
    
    if not _OPENAI_AVAILABLE:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the EXACT title of the research paper from the first-page text. "
                        "Return ONLY the title. Ignore journal names, volume numbers, or headers. "
                        "Return nothing if no clear paper title is found."
                    ),
                },
                {"role": "user", "content": text_preview},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        title = response.choices[0].message.content.strip()
        return title if len(title) <= 300 else ""

    except Exception as e:
        logger.warning(f"[PREPROCESS] Async LLM title extraction failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Top-level worker function for ProcessPoolExecutor
# (must be picklable — defined at module level, not inside a class)
# ─────────────────────────────────────────────────────────────

def _process_single_document(doc_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    
    try:
        chunker = ResearchPaperChunker()
        chunks = chunker.chunk_document(doc_data)
        paper_id = doc_data.get("paper_id") or doc_data.get("title", "unknown")
        logger.info(f"[PREPROCESS] ✓ Processed: {paper_id} → {len(chunks)} chunks")
        return [{"text": c.text, "metadata": c.metadata, "chunk_id": c.chunk_id} for c in chunks]
    except Exception as e:
        paper_id = doc_data.get("paper_id") or doc_data.get("title", "unknown")
        logger.error(f"[PREPROCESS] ✗ Failed: {paper_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Main Chunker
# ─────────────────────────────────────────────────────────────

class ResearchPaperChunker:

    # ── Compiled patterns (class-level, compiled once) ──────
    _TITLE_STOP_RE = re.compile(
        r'^(abstract|authors?|introduction|keywords?)\b', re.IGNORECASE
    )
    _AUTHOR_MARKER_RE = re.compile(r'[0-9].*[,∗]|[∗].*[0-9]')
    _JOURNAL_NOISE_RE = re.compile(
        r'(journal|proceedings|transactions|volume|issue|publisher|press|'
        r'conference|accepted|received|published)', re.IGNORECASE
    )
    _DOI_RE = re.compile(r'^(doi:|arxiv:|pages?:)', re.IGNORECASE)
    _PUBLISHER_RE = re.compile(
        r'(ieee|springer|acm|elsevier|wiley|nature|science)\s+(transactions|proceedings|journal)',
        re.IGNORECASE,
    )
    _COPYRIGHT_RE = re.compile(r'(copyright|©|\(c\))', re.IGNORECASE)
    _URL_RE = re.compile(r'(http://|https://|www\.|@.+\..+)', re.IGNORECASE)
    _YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
    _PAGE_MARKER_RE = re.compile(r'^\[PAGE\s+(\d+)\]$')
    _WHITESPACE_NORMALIZE_RE = re.compile(r'\s+')
    _CAMEL_SPLIT_RE = re.compile(r'([a-z])([A-Z])')
    _CAMEL_UPPER_RE = re.compile(r'([a-z])([A-Z]{2,})')
    _CHUNK_ID_CLEAN_RE = re.compile(r'[^a-zA-Z0-9_\-\.]')

    AUTHOR_PATTERNS = [
        re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)'),
        re.compile(r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)'),
        re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+et\s+al\.)'),
    ]

    SECTION_HEADERS: Dict[str, List[str]] = {
        "abstract":     [r"^Abstract", r"^ABSTRACT", r"^Summary"],
        "introduction": [r"^1\.", r"^Introduction", r"^INTRODUCTION"],
        "methodology":  [r"^2\.", r"^Methodology", r"^Methods", r"^METHOD"],
        "results":      [r"^3\.", r"^Results", r"^RESULTS", r"^Findings"],
        "discussion":   [r"^4\.", r"^Discussion", r"^DISCUSSION"],
        "conclusion":   [r"^5\.", r"^Conclusion", r"^CONCLUSIONS"],
        "references":   [r"^References", r"^Bibliography", r"^REFERENCES"],
    }
    # Pre-compile section header patterns
    _SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
        sec: [re.compile(p, re.IGNORECASE) for p in pats]
        for sec, pats in SECTION_HEADERS.items()
    }

    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        )

    # ──────────────────────────────────────────────────────
    # Batch Processing (parallel)
    # ──────────────────────────────────────────────────────

    def batch_process_documents(
        self,
        documents: List[Dict[str, Any]],
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        
        if not documents:
            return []

        logger.info(
            f"[PREPROCESS] Batch processing {len(documents)} docs "
            f"with {max_workers} workers..."
        )

        all_chunks: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_document, doc): doc
                for doc in documents
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_chunks.extend(result)

        logger.info(
            f"[PREPROCESS] Batch complete → {len(all_chunks)} total chunks "
            f"from {len(documents)} documents."
        )
        return all_chunks

    async def batch_process_with_llm_async(
        self,
        documents: List[Dict[str, Any]],
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
       
        if not documents:
            return []

        logger.info(
            f"[PREPROCESS] Async batch: pre-fetching LLM titles for "
            f"{len(documents)} docs..."
        )

        # Step 1: Concurrently fetch LLM titles for docs that need it
        async def _maybe_fetch_title(doc: Dict[str, Any]) -> Dict[str, Any]:
            text = doc.get("content", "")
            quick_title = self._extract_title(text.split("\n")[:40])
            if not quick_title or len(quick_title) < 15:
                first_40_lines = "\n".join(text.split("\n")[:40])
                llm_title = await _extract_title_with_llm_async(first_40_lines)
                if llm_title:
                    doc = {**doc, "_llm_title": llm_title}
            return doc

        enriched_docs = await asyncio.gather(
            *[_maybe_fetch_title(d) for d in documents]
        )

        # Step 2: CPU-bound chunking in parallel processes
        all_chunks: List[Dict[str, Any]] = []
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, _process_single_document, doc)
                for doc in enriched_docs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_chunks.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"[PREPROCESS] Worker error: {result}")

        logger.info(f"[PREPROCESS] Async batch complete → {len(all_chunks)} chunks.")
        return all_chunks

    # ──────────────────────────────────────────────────────
    # Single-document entry points
    # ──────────────────────────────────────────────────────

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[TextChunk]:
        
        text = doc_data.get("content", "")
        paper_id = (
            doc_data.get("paper_id")
            or doc_data.get("title")
            or str(uuid.uuid4())
        )
        known_title = doc_data.get("title", "")
        prefetched_llm_title = doc_data.get("_llm_title", "")
        return self.chunk_text(text, paper_id, known_title, prefetched_llm_title)

    def chunk_text(
        self,
        text: str,
        paper_id: str = "",
        known_title: str = "",
        prefetched_llm_title: str = "",
    ) -> List[TextChunk]:
        
        metadata = self.extract_metadata(text)

        # ── Title resolution ──────────────────────────────
        if not metadata.title or len(metadata.title) < 15:
            if prefetched_llm_title:
                metadata.title = prefetched_llm_title
                logger.debug(f"[PREPROCESS] Used pre-fetched LLM title: {prefetched_llm_title}")
            else:
                first_40 = "\n".join(text.split("\n")[:40])
                llm_title = _extract_title_with_llm(first_40)
                if llm_title:
                    metadata.title = llm_title
                    logger.debug(f"[PREPROCESS] LLM title: {llm_title}")

        if not metadata.title and known_title:
            metadata.title = known_title
            logger.debug(f"[PREPROCESS] Falling back to known title: {known_title}")
        # ─────────────────────────────────────────────────

        chunks: List[TextChunk] = []
        chunks.extend(self._create_metadata_chunks(metadata, paper_id))
        chunks.extend(self._create_content_chunks(text, paper_id))

        # Assign stable chunk IDs
        clean_prefix = self._CHUNK_ID_CLEAN_RE.sub("_", paper_id or "chunk")
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                chunk.chunk_id = f"{clean_prefix}_{i}"

        return chunks

    # ──────────────────────────────────────────────────────
    # Metadata extraction
    # ──────────────────────────────────────────────────────

    def extract_metadata(self, text: str) -> PaperMetadata:
        """Extract all structured metadata from document text."""
        lines = text.split("\n")
        meta = PaperMetadata()
        meta.title = self._extract_title(lines)
        meta.authors, meta.affiliations = self._extract_authors_and_affiliations(lines)
        meta.abstract = self._extract_abstract(lines)
        meta.keywords = self._extract_keywords(lines)
        meta.publication_year = self._extract_publication_year(lines)
        return meta

    def _extract_title(self, lines: List[str]) -> str:
        
        candidates: List[str] = []
        in_title = False

        for i, line in enumerate(lines[:40]):
            clean = line.strip()

            if not clean or len(clean) < 3:
                if in_title and len(" ".join(candidates)) > 20:
                    break
                continue

            # Stop at known section markers
            if self._TITLE_STOP_RE.search(clean):
                break

            # Stop at author/affiliation footnote lines early in doc
            if i < 10 and self._AUTHOR_MARKER_RE.search(clean):
                break

            # ── Noise filtering (first 15 lines) ──
            if i < 15:
                if re.match(r"^\d+$", clean) or re.match(r"^(19|20)\d{2}$", clean):
                    continue
                if self._JOURNAL_NOISE_RE.search(clean):
                    continue
                if clean.lower().endswith(("journal", "humanity", "science", "research", "review")):
                    continue
                if re.search(r"volume\s*\d+|issue\s*\d+", clean, re.IGNORECASE):
                    continue
                if self._DOI_RE.search(clean):
                    continue
                if self._PUBLISHER_RE.search(clean):
                    continue
                if self._COPYRIGHT_RE.search(clean):
                    continue

            # Skip URL / email lines
            if self._URL_RE.search(clean):
                continue

            # Skip lines that look like symbol/number artefacts
            if re.match(r"^[\d\W]{5,}$", clean):
                continue

            # If we have candidates and hit an author-like line, stop
            if candidates and re.match(r"^([A-Z][a-z]+\s+[A-Z][a-z]+(,\s*)?)+$", clean):
                break

            word_count = len(clean.split())
            is_candidate = (
                10 <= len(clean) <= 200
                and word_count >= 2
                and clean[0].isupper()
            ) or (
                clean.isupper()
                and 10 <= len(clean) <= 200
                and i < 5
            )

            if is_candidate:
                in_title = True
                candidates.append(clean)

                if len(" ".join(candidates)) > 15:
                    if clean.endswith((".", "!", "?", ":")):
                        break
                    if i + 1 < len(lines):
                        nxt = lines[i + 1].strip()
                        if not nxt or (nxt and nxt[0].islower()) or "@" in nxt:
                            break
            elif in_title:
                break

            if len(" ".join(candidates)) > 250:
                break

        title = " ".join(candidates).strip()
        title = self._WHITESPACE_NORMALIZE_RE.sub(" ", title)
        title = re.sub(r"^[\W\d]+", "", title)
        title = re.sub(r"[\W\d]+$", "", title)

        # Fix merged-word PDF artefacts (e.g. "testGap" → "test Gap")
        title = self._CAMEL_SPLIT_RE.sub(r"\1 \2", title)
        title = self._CAMEL_UPPER_RE.sub(r"\1 \2", title)

        # Convert all-caps to title case, preserving short acronyms
        if title and title.isupper() and len(title) > 20:
            title = " ".join(
                w if len(w) <= 3 else w.capitalize()
                for w in title.split()
            )

        result = title if title and len(title) >= 10 else ""
        logger.debug(f"[PREPROCESS] Extracted title: {result!r}")
        return result

    def _extract_authors_and_affiliations(
        self, lines: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Extract authors and affiliations, preserving insertion order."""
        authors_seen: Dict[str, None] = {}
        affiliations_seen: Dict[str, None] = {}

        for line in lines[:20]:
            stripped = line.strip()
            for pattern in self.AUTHOR_PATTERNS:
                for match in pattern.findall(stripped):
                    authors_seen.setdefault(match, None)

            if any(kw in stripped.lower() for kw in ("university", "institute", "college", "lab")):
                affiliations_seen.setdefault(stripped, None)

            if "abstract" in stripped.lower() or re.match(r"^1\.", stripped):
                break

        return list(authors_seen), list(affiliations_seen)

    def _extract_abstract(self, lines: List[str]) -> str:
        
        abstract_lines: List[str] = []
        in_abstract = False
        char_count = 0
        MAX_ABSTRACT_CHARS = 3000

        for line in lines:
            stripped = line.strip()

            if re.match(r"^(Abstract|ABSTRACT|Summary)", stripped, re.IGNORECASE):
                in_abstract = True
                continue

            if in_abstract:
                if re.match(r"^(Keywords|KEYWORDS|1\.|Introduction)", stripped, re.IGNORECASE):
                    break
                if char_count >= MAX_ABSTRACT_CHARS:
                    break
                if stripped:
                    abstract_lines.append(stripped)
                    char_count += len(stripped)

        return " ".join(abstract_lines)

    def _extract_keywords(self, lines: List[str]) -> List[str]:
        """Extract keywords from a 'Keywords:' line."""
        for line in lines:
            if re.search(r"Keywords?[:\s]", line, re.IGNORECASE):
                keyword_text = re.split(r"Keywords?[:\s]+", line, flags=re.IGNORECASE)[-1]
                return [k.strip() for k in re.split(r"[;,]", keyword_text) if k.strip()]
        return []

    def _extract_publication_year(self, lines: List[str]) -> Optional[int]:
        """Return the first 4-digit year found in the first 50 lines."""
        for line in lines[:50]:
            m = self._YEAR_RE.search(line)
            if m:
                try:
                    return int(m.group())
                except ValueError:
                    continue
        return None

    # ──────────────────────────────────────────────────────
    # Chunk construction
    # ──────────────────────────────────────────────────────

    def _create_metadata_chunks(
        self, metadata: PaperMetadata, paper_id: str
    ) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        title_text = metadata.title or "Title not detected from PDF"

        # Primary title chunk
        chunks.append(TextChunk(
            text=f"Title: {title_text}",
            metadata={
                "paper_id": paper_id,
                "chunk_type": "title",
                "is_metadata": True,
                "section": "metadata",
                "importance": "high",
                "title": title_text,
            },
        ))

        # Bare title for direct semantic matching
        if metadata.title:
            chunks.append(TextChunk(
                text=title_text,
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "title_reference",
                    "is_metadata": True,
                    "section": "metadata",
                    "importance": "high",
                    "title": title_text,
                },
            ))

        if metadata.authors:
            chunks.append(TextChunk(
                text=f"Authors: {', '.join(metadata.authors)}",
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "authors",
                    "is_metadata": True,
                    "section": "metadata",
                    "author_count": len(metadata.authors),
                },
            ))

        if metadata.abstract:
            abstract_parts = self.text_splitter.split_text(metadata.abstract)
            total = len(abstract_parts)
            for i, part in enumerate(abstract_parts):
                chunks.append(TextChunk(
                    text=f"Abstract (Part {i + 1}/{total}): {part}",
                    metadata={
                        "paper_id": paper_id,
                        "chunk_type": "abstract",
                        "is_metadata": True,
                        "section": "metadata",
                        "part": i + 1,
                        "total_parts": total,
                    },
                ))

        return chunks

    def _create_content_chunks(self, text: str, paper_id: str) -> List[TextChunk]:
        """Split document body into section-aware, page-tracked chunks."""
        chunks: List[TextChunk] = []
        lines = text.split("\n")

        current_section = "introduction"
        current_content: List[str] = []
        current_page = 1
        section_start = 0
        section_pages: set = {1}

        def _flush(section: str, content: List[str], pages: set) -> List[TextChunk]:
            if not content:
                return []
            return self._split_section_text(
                "\n".join(content), paper_id, section, pages
            )

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Update page counter
            pm = self._PAGE_MARKER_RE.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                section_pages.add(current_page)
                continue

            detected = self._detect_section(stripped)
            if detected and i > section_start + 5:
                chunks.extend(_flush(current_section, current_content, section_pages))
                current_section = detected
                current_content = [stripped]
                section_pages = {current_page}
                section_start = i
            elif stripped:
                current_content.append(stripped)
                section_pages.add(current_page)

        chunks.extend(_flush(current_section, current_content, section_pages))
        return chunks

    def _detect_section(self, line: str) -> Optional[str]:
        """Return section name if line matches a known section header."""
        for sec_name, patterns in self._SECTION_PATTERNS.items():
            for pat in patterns:
                if pat.match(line):
                    return sec_name
        return None

    def _split_section_text(
        self,
        text: str,
        paper_id: str,
        section_name: str,
        pages: set,
    ) -> List[TextChunk]:
        """Split section text into sized chunks with page-range metadata."""
        text = re.sub(r"\[PAGE\s+\d+\]", "", text).strip()
        if not text:
            return []

        raw_chunks = self.text_splitter.split_text(text)
        page_start = min(pages) if pages else 0
        page_end = max(pages) if pages else 0
        total = len(raw_chunks)

        return [
            TextChunk(
                text=chunk.strip(),
                metadata={
                    "paper_id": paper_id,
                    "chunk_type": "content",
                    "is_metadata": False,
                    "section": section_name,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "page_start": page_start,
                    "page_end": page_end,
                },
            )
            for idx, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]


# ─────────────────────────────────────────────────────────────
# Convenience batch entry-points
# ─────────────────────────────────────────────────────────────

def batch_chunk_documents(
    documents: List[Dict[str, Any]],
    max_workers: int = 4,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    
    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.batch_process_documents(documents, max_workers=max_workers)


def batch_chunk_documents_async(
    documents: List[Dict[str, Any]],
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
  
    chunker = ResearchPaperChunker()
    return asyncio.run(
        chunker.batch_process_with_llm_async(documents, max_workers=max_workers)
    )


# ─────────────────────────────────────────────────────────────
# Backward-compatible helpers
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning (backward compat)."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Legacy: simple chunking, returns list of strings."""
    chunker = ResearchPaperChunker(chunk_size=chunk_size, overlap=overlap)
    return [c.text for c in chunker.chunk_text(text)]


def chunk_text_with_metadata(text: str, paper_id: str = "") -> List[Dict[str, Any]]:
    """Legacy: returns list of {'text': ..., 'metadata': ...} dicts."""
    chunker = ResearchPaperChunker()
    return [{"text": c.text, "metadata": c.metadata} for c in chunker.chunk_text(text, paper_id)]