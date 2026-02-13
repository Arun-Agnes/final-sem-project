# app.py - AI Research Literature Assistant (Production Auth)
"""
AI Research Literature Assistant
Production-ready persistent authentication using query parameters
"""

import streamlit as st
import os
from datetime import datetime
import uuid
import hashlib
import logging

# Third-party imports
import chromadb
from sqlalchemy.orm import Session as DBSession

# Local imports
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.preprocess import ResearchPaperChunker
from ingestion.embed_store import store_documents_enhanced, CHROMA_DIR
from ingestion.image_extractor import process_pdf_images
from rag.pipeline import RAGPipeline
from citation_generator import IEEECitationGenerator
from auth import auth_dialog, logout
from database import Session as ChatSession, Message, SessionLocal, init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="AI Research Literature Assistant",
    page_icon="📚",
    layout="wide"
)

# ==================== CSS STYLING ====================
def load_css():
    """Load custom CSS styles"""
    try:
        with open("static/style.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Initialize database
init_db()

# Load custom CSS immediately after set_page_config
load_css()

# Utility imports
from utils.document_manager import DocumentManager
from utils.document_selector import DocumentSelector
from utils.metrics_tracker import MetricsTracker, ConfidenceCalculator
from utils.ui_components import SearchModeSelector
from utils.export_tools import ChatExporter
from utils.comparison_tools import DocumentComparator
from utils.auth_helper import init_session_auth, cleanup_expired_sessions

# ==================== CRITICAL: INITIALIZE AUTH FIRST ====================
# This validates session from URL and restores authentication state
init_session_auth()

# Cleanup old sessions periodically (runs once per page load)
cleanup_expired_sessions()

# ==================== SESSION INITIALIZATION ====================
# Initialize authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()

if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []

if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "semantic"

if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.7

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "chat"

if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = False


# ==================== HELPER FUNCTIONS ====================

@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    """Retrieve or create the RAG pipeline."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = client.get_collection("research_papers")
            if collection.count() == 0:
                return None
        except ValueError:
            return None  # Collection doesn't exist
        
        pipeline = RAGPipeline()
        # pipeline.initialize() # Removed as it doesn't exist on the class
        return pipeline
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return None

def load_chat_session(session_id):
    """Load conversation history from database"""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).get(session_id)
        if session:
            st.session_state.current_session_id = session.id
            st.session_state.conversation = []
            
            messages = db.query(Message).filter(
                Message.session_id == session.id
            ).order_by(Message.timestamp).all()
            
            for msg in messages:
                entry = {
                    "role": msg.role,
                    "content": msg.content
                }
                # Try to restore sources from JSON if stored
                if msg.sources_json:
                    try:
                        import json
                        entry["sources"] = json.loads(msg.sources_json)
                    except Exception:
                        pass
                st.session_state.conversation.append(entry)
    except Exception as e:
        logger.error(f"Error loading chat session: {e}")
        st.error("Failed to load chat session")
    finally:
        db.close()

def enforce_max_sessions(user_id, max_sessions=20):
    """Enforce max N chat sessions per user, deleting oldest beyond the limit."""
    db = SessionLocal()
    try:
        all_sessions = db.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(ChatSession.last_updated.desc()).all()
        
        if len(all_sessions) > max_sessions:
            sessions_to_delete = all_sessions[max_sessions:]
            for s in sessions_to_delete:
                db.delete(s)
            db.commit()
            logger.info(f"Pruned {len(sessions_to_delete)} old sessions for user {user_id}")
    except Exception as e:
        logger.error(f"Error enforcing max sessions: {e}")
    finally:
        db.close()

def save_message(role, content, sources=None):
    """Save message to database, creating session if needed"""
    if not st.session_state.authenticated:
        return
    
    db = SessionLocal()
    
    try:
        current_sid = st.session_state.get('current_session_id')
        if not current_sid:
            title = content[:30] + "..." if len(content) > 30 else content
            new_sess = ChatSession(
                user_id=st.session_state.user_id,
                title=title
            )
            db.add(new_sess)
            db.commit()
            db.refresh(new_sess)
            current_sid = new_sess.id
            st.session_state.current_session_id = current_sid
            # Enforce max 20 sessions per user
            enforce_max_sessions(st.session_state.user_id, max_sessions=20)
        
        # Serialize sources to JSON if provided
        sources_json_str = None
        if sources:
            try:
                import json
                sources_json_str = json.dumps(sources, default=str)
            except Exception:
                pass
        
        msg = Message(
            session_id=current_sid,
            role=role,
            content=str(content),
            sources_json=sources_json_str
        )
        db.add(msg)
        
        sess = db.query(ChatSession).get(current_sid)
        if sess:
            sess.last_updated = datetime.utcnow()
        
        db.commit()
    except Exception as e:
        logger.error(f"Error saving message: {e}")
    finally:
        db.close()

def _process_single_file(file_name, file_content, doc_manager):
    """Process a single PDF file (thread-safe, no Streamlit calls)."""
    import io
    from pypdf import PdfReader
    
    file_size = len(file_content)
    
    if doc_manager.document_exists(file_name, file_size):
        return None  # Skip already processed
    
    file_hash = hashlib.md5(file_content).hexdigest()[:12]
    
    doc_info = doc_manager.add_document(file_name, file_size, file_hash)
    
    # Count pages first
    try:
        reader = PdfReader(io.BytesIO(file_content))
        num_pages = len(reader.pages)
    except Exception:
        num_pages = 0
    
    # Extract text from PDF (with cleaning)
    text = extract_text_from_pdf(file_content)
    
    # Chunk the document
    chunker = ResearchPaperChunker()
    chunks = chunker.chunk_document({
        'content': text,
        'paper_id': file_hash,
        'title': file_name
    })
    
    # Update document manager with page count and chunk count
    doc_manager.update_document(
        file_hash, 
        status='completed',
        num_chunks=len(chunks),
        num_pages=num_pages
    )
    
    logger.info(f"Processed '{file_name}': {num_pages} pages, {len(chunks)} chunks, {len(text)} chars")
    
    return {
        'file_name': file_name,
        'file_hash': file_hash,
        'file_content': file_content,  # Keep bytes for image extraction
        'chunks': chunks,
        'num_pages': num_pages,
    }


@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

def process_uploaded_files(uploaded_files) -> bool:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        # Load model early to ensure it's ready
        embedding_model = load_embedding_model()
        if not embedding_model:
            st.error("Failed to load embedding model.")
            return False

        # Pre-read all file contents (must happen in main thread)
        file_data = []
        for f in uploaded_files:
            content = f.read()
            file_data.append((f.name, content))
        
        total = len(file_data)
        progress_bar = st.progress(0, text="Starting parallel processing...")
        status_text = st.empty()
        
        all_chunks = []
        all_chunk_ids = []
        processed_results = []  # Store results for image extraction phase
        completed = 0
        
        # ── Phase 1: Text extraction & chunking (parallel) ──
        with ThreadPoolExecutor(max_workers=min(2, total)) as executor:
            futures = {
                executor.submit(
                    _process_single_file, name, content, st.session_state.doc_manager
                ): name 
                for name, content in file_data
            }
            
            for future in as_completed(futures):
                file_name = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        for chunk in result['chunks']:
                            all_chunks.append(chunk)
                            all_chunk_ids.append(chunk.chunk_id)
                        st.session_state.uploaded_files.add(result['file_name'])
                        processed_results.append(result)
                        status_text.text(f"✅ {result['file_name']} (text extracted)")
                    else:
                        status_text.text(f"⏭️ {file_name} (already processed)")
                except Exception as e:
                    status_text.text(f"❌ {file_name}: {str(e)}")
                    logger.error(f"Error processing {file_name}: {e}")
                
                progress_bar.progress(
                    (completed / total) * 0.5,
                    text=f"📄 Extracting text: {completed}/{total} files..."
                )
        
        # ── Phase 2: Image extraction & Vision analysis (sequential per file) ──
        if processed_results:
            status_text.text("🖼️ Extracting and analyzing images...")
            image_chunk_count = 0
            
            for idx, result in enumerate(processed_results):
                file_content = result.get('file_content')
                file_hash = result['file_hash']
                file_name = result['file_name']
                
                if file_content:
                    try:
                        progress_bar.progress(
                            0.5 + (idx / len(processed_results)) * 0.3,
                            text=f"🖼️ Analyzing images in {file_name}..."
                        )
                        
                        image_chunks = process_pdf_images(
                            pdf_source=file_content,
                            paper_id=file_hash,
                            max_images=10,
                        )
                        
                        for ic in image_chunks:
                            chunk_id = f"{file_hash}_img_{image_chunk_count}"
                            all_chunks.append(ic)
                            all_chunk_ids.append(chunk_id)
                            image_chunk_count += 1
                        
                        if image_chunks:
                            status_text.text(
                                f"🖼️ {file_name}: {len(image_chunks)} images analyzed"
                            )
                    except Exception as e:
                        logger.warning(f"Image extraction failed for {file_name}: {e}")
                        status_text.text(f"⚠️ {file_name}: image extraction skipped")
            
            if image_chunk_count > 0:
                logger.info(f"Total image chunks created: {image_chunk_count}")
        
        # ── Phase 3: Generate embeddings & store in ChromaDB ──
        if all_chunks:
            progress_bar.progress(0.85, text="💾 Generating embeddings...")
            status_text.text(f"Generating embeddings for {len(all_chunks)} chunks...")
            
            store_documents_enhanced(all_chunks, all_chunk_ids, embedding_model=embedding_model)
            
            progress_bar.progress(1.0, text="✅ All done!")
            status_text.empty()
            return True
        
        progress_bar.progress(1.0, text="✅ Complete (no new files to process)")
        status_text.empty()
        return False
            
    except Exception as e:
        import traceback
        logger.error(f"Error processing files: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error processing files: {str(e)}")
        return False

def render_chat_message(qa: dict):
    with st.chat_message(qa["role"]):
        st.markdown(qa["content"])
        
        # Display any images referenced in the response
        if qa.get("images"):
            for img_info in qa["images"]:
                img_path = img_info.get("path", "")
                caption = img_info.get("caption", "Figure")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=caption, use_container_width=True)
        
        if st.session_state.show_sources and qa.get("sources"):
            with st.expander("📎 Sources & Figures"):
                for i, source in enumerate(qa["sources"][:5]):
                    meta = source.get("metadata", {})
                    chunk_type = meta.get("chunk_type", "content")
                    
                    # Try to find a good title/ID
                    title = meta.get("title") or meta.get("paper_id") or source.get("id", "Unknown")
                    
                    # Display image if this source is an image chunk
                    if chunk_type == "image_description":
                        image_path = meta.get("image_path", "")
                        page = meta.get("page_start", "?")
                        st.markdown(f"**🖼️ Figure (Page {page}):**")
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        # Show the description below the image
                        preview = source.get("content_preview", "")
                        if preview:
                            st.caption(preview[:300])
                        relevance = source.get("similarity", "N/A")
                        st.markdown(f"- Relevance: {relevance}")
                    else:
                        st.markdown(f"**Source {i+1}:** {title}")
                        page_start = meta.get("page_start", "")
                        page_end = meta.get("page_end", "")
                        if page_start and page_end and page_start != page_end:
                            st.markdown(f"- Pages: {page_start}-{page_end}")
                        elif page_start:
                            st.markdown(f"- Page: {page_start}")
                        section = meta.get("section", "")
                        if section and section != "unknown":
                            st.markdown(f"- Section: {section.title()}")
                        relevance = source.get("similarity", "N/A")
                        st.markdown(f"- Relevance: {relevance}")
                    
                    if i < len(qa["sources"][:5]) - 1:
                        st.divider()

# ==================== TOP PROFILE BUTTON ====================
def render_top_profile():
    """Render the profile button in the top right corner"""
    if st.session_state.authenticated:
        # Wrap the popover in a container with a key for reliable CSS targeting
        with st.container(key="nav-profile-popover"):
            with st.popover("Profile"):
                st.write(f"Logged in: **{st.session_state.username}**")
                if st.button("Log Out", key="logout-btn-top", use_container_width=True):
                    logout()
    else:
        # Wrap the login button in a container with a key for reliable CSS targeting
        with st.container(key="nav-profile-top"):
            if st.button("Login / Sign Up", key="login-btn-trigger"):
                auth_dialog()

render_top_profile()

# ==================== SIDEBAR NAVIGATION ====================

sidebar_width = "250px" if st.session_state.sidebar_expanded else "80px"
sidebar_align = "left" if st.session_state.sidebar_expanded else "center"
btn_padding = "0.5rem 1rem" if st.session_state.sidebar_expanded else "0rem"
btn_align = "left" if st.session_state.sidebar_expanded else "center"
btn_justify = "flex-start" if st.session_state.sidebar_expanded else "center"

st.markdown(f"""
    <style>
    :root {{
        --sidebar-width: {sidebar_width};
        --sidebar-text-align: {sidebar_align};
        --sidebar-btn-padding: {btn_padding};
        --sidebar-btn-content-align: {btn_align};
        --sidebar-btn-justify: {btn_justify};
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Custom sidebar toggle with panel icon (CSS provides the SVG)
    if st.button(".", key="toggle-sidebar", help="Toggle sidebar"):
        st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
        st.session_state._sidebar_toggling = True  # Flag to prevent session restore
        st.rerun()
    
    if st.session_state.sidebar_expanded:
        if st.button("Chat", key="nav-chat", help="Chat", use_container_width=True):
            st.session_state.current_tab = "chat"
            st.rerun()
        
        if st.button("Documents", key="nav-docs", help="Documents", use_container_width=True):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button("Compare", key="nav-compare", help="Compare", use_container_width=True):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button("Export", key="nav-export", help="Export", use_container_width=True):
            st.session_state.current_tab = "export"
            st.rerun()
        
        if st.session_state.authenticated:
            if st.button("New Session", key="nav-new-session", help="Start New Session", use_container_width=True):
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.current_tab = "chat"
                st.rerun()
            
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">HISTORY</p>', unsafe_allow_html=True)
            
            # Clear toggle flag — history clicks in THIS render cycle are safe
            is_toggling = st.session_state.pop('_sidebar_toggling', False)
            
            try:
                db = SessionLocal()
                user_sessions = db.query(ChatSession).filter(
                    ChatSession.user_id == st.session_state.user_id
                ).order_by(ChatSession.last_updated.desc()).limit(20).all()
                
                for s in user_sessions:
                    label = s.title if s.title else "New Chat"
                    if len(label) > 18:
                        label = label[:16] + "..."
                    
                    if st.button(label, key=f"hist_{s.id}", use_container_width=True):
                        # Only load session if we're NOT in the middle of a toggle rerun
                        if not is_toggling:
                            load_chat_session(s.id)
                            st.session_state.current_tab = "chat"
                            st.rerun()
                db.close()
            except Exception as e:
                logger.error(f"Failed to load history: {e}")

    else:
        if st.button(".", key="nav-chat-icon", help="Chat"):
            st.session_state.current_tab = "chat"
            st.rerun()
        
        if st.button(".", key="nav-docs-icon", help="Documents"):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button(".", key="nav-compare-icon", help="Compare"):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button(".", key="nav-export-icon", help="Export"):
            st.session_state.current_tab = "export"
            st.rerun()
        
        if st.session_state.authenticated:
            if st.button(".", key="nav-new-session-icon", help="Start New Session"):
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.current_tab = "chat"
                st.rerun()

# ==================== MAIN HEADER ====================
st.markdown("""
    <div class="main-header-container">
        <h1>AI Research Literature Assistant</h1>
        <p>Intelligent document analysis powered by RAG</p>
    </div>
""", unsafe_allow_html=True)

# ==================== CHAT TAB ====================
if st.session_state.current_tab == "chat":
    if st.session_state.doc_manager.get_document_count() == 0:
        if not st.session_state.authenticated:
            with st.container(key="auth-trigger-upload"):
                if st.button("Login\nto Upload Research PDFs", key="auth-btn-trigger", type="primary"):
                    auth_dialog()
        else:
            with st.container(key="authenticated-upload"):
                uploaded_files = st.file_uploader(
                    "Upload Research PDFs",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="uploader-component"
                )
            
            # Cache uploaded file data in session state so it survives sidebar toggle reruns
            if uploaded_files:
                cached_files = []
                seen_names = set()
                duplicate_names = set()
                for f in uploaded_files:
                    if f.name not in seen_names:
                        seen_names.add(f.name)
                        content = f.read()
                        cached_files.append({"name": f.name, "content": content, "size": len(content)})
                        f.seek(0)  # Reset for later use
                    else:
                        duplicate_names.add(f.name)
                st.session_state._uploaded_file_data = cached_files
                st.session_state._uploaded_file_objects = uploaded_files
                if duplicate_names:
                    already_reported = st.session_state.get("reported_duplicates", set())
                    new_dupes = duplicate_names - already_reported
                    if new_dupes:
                        st.session_state.reported_duplicates = already_reported | new_dupes
                        st.toast(f"{len(new_dupes)} duplicate file(s) removed from selection.", icon="⚠️")
                else:
                    st.session_state.pop("reported_duplicates", None)
            
            # Display selected files from cache (survives sidebar toggle)
            cached = st.session_state.get('_uploaded_file_data', [])
            if cached:
                with st.container(key="selected-files-list-section"):
                    st.markdown("---")
                    st.markdown("### Selected Documents")
                    
                    for fd in cached:
                        size_mb = fd["size"] / 1024 / 1024
                        st.markdown(f"**{fd['name']}** ({size_mb:.2f} MB)")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Process Documents", key="process-docs-btn"):
                        # Show full-screen processing overlay with animated SVG
                        st.markdown("""
                        <style>
                        .processing-overlay {
                            position: fixed;
                            top: 0; left: 0; right: 0; bottom: 0;
                            background: rgba(0, 0, 0, 0.85);
                            z-index: 9999999;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            gap: 1.5rem;
                            animation: fadeIn 0.3s ease;
                        }
                        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                        @keyframes pulse { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
                        .processing-text {
                            color: #e0e0e0;
                            font-family: 'Inter', sans-serif;
                            font-size: 1.2rem;
                            font-weight: 500;
                            animation: pulse 2s ease-in-out infinite;
                            margin-top: 1rem;
                        }
                        .processing-subtext {
                            color: #888;
                            font-family: 'Inter', sans-serif;
                            font-size: 0.9rem;
                        }
                        .custom-spinner {
                            width: 100px;
                            height: 50px;
                        }
                        </style>
                        <div class="processing-overlay">
                            <div class="custom-spinner">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">
                                    <path fill="none" stroke="#6B7280" stroke-width="15" stroke-linecap="round" stroke-dasharray="300 385" stroke-dashoffset="0" d="M275 75c0 31-27 50-50 50-58 0-92-100-150-100-28 0-50 22-50 50s23 50 50 50c58 0 92-100 150-100 24 0 50 19 50 50Z">
                                        <animate attributeName="stroke-dashoffset" calcMode="spline" dur="2" values="685;-685" keySplines="0 0 1 1" repeatCount="indefinite"></animate>
                                    </path>
                                </svg>
                            </div>
                            <div class="processing-text">Processing Documents...</div>
                            <div class="processing-subtext">Extracting text, analyzing images & building embeddings</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Use cached file objects if available, otherwise fall back
                        files_to_process = st.session_state.get('_uploaded_file_objects', uploaded_files)
                        if files_to_process:
                             # Use fewer threads to avoid memory crashes
                            if process_uploaded_files(files_to_process):
                                # Clear cached file data after successful processing
                                st.session_state.pop('_uploaded_file_data', None)
                                st.session_state.pop('_uploaded_file_objects', None)
                                st.success("Documents processed successfully!")
                                st.cache_resource.clear()  # Force reload of RAG pipeline
                                st.rerun()
                            else:
                                st.error("Processing failed or no new files to process.")
    else:
        with st.expander("Search Settings"):
            search_mode, alpha = SearchModeSelector.show()
            st.session_state.search_mode = search_mode
            st.session_state.alpha = alpha
            show_sources = st.checkbox("Show source citations", value=st.session_state.show_sources)
            st.session_state.show_sources = show_sources

        for qa in st.session_state.conversation:
            render_chat_message(qa)

        user_query = st.chat_input("Ask a question about your documents...")
        
        if user_query:
            pipeline = get_rag_pipeline()
            
            if pipeline:
                # Custom SVG spinner for "Thinking" state
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown("""
                    <div style="display: flex; justify-content: center; align-items: center; margin: 2rem 0; width: 100%;">
                         <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" style="width: 60px; height: 60px;">
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="40" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.4"></animate>
                            </circle>
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="100" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.2"></animate>
                            </circle>
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="160" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="0"></animate>
                            </circle>
                        </svg>
                    </div>
                """, unsafe_allow_html=True)
                
                try:
                    result = pipeline.query(
                        user_query,
                        search_mode=st.session_state.search_mode,
                        alpha=st.session_state.alpha
                    )
                finally:
                    spinner_placeholder.empty()
                
                st.session_state.conversation.append({"role": "user", "content": user_query})
                if st.session_state.authenticated:
                    save_message("user", user_query)
                
                response_content = result.get('answer', result.get('response', ''))
                sources = result.get('sources', [])
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response_content,
                    "sources": sources
                })
                if st.session_state.authenticated:
                    save_message("assistant", response_content, sources=sources)
                
                st.rerun()
            else:
                st.error("Pipeline not initialized. Please upload documents first.")

# ==================== DOCUMENTS TAB ====================
elif st.session_state.current_tab == "documents":
    st.subheader("Document Library")

    if st.session_state.doc_manager.get_document_count() > 0:
        # Convert DataFrame to list of dicts for iteration
        df_docs = st.session_state.doc_manager.get_all_documents()
        docs = df_docs.to_dict('records')
        
        for doc in docs:
            doc_name = doc.get('name', 'Unknown Document')
            with st.expander(doc_name):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Uploaded:**", doc.get('upload_time', 'Unknown'))
                    st.write("**Status:**", doc.get('status', 'Unknown'))
                with col2:
                    st.write("**Chunks:**", doc.get('num_chunks', 0))
                    st.write("**Size:**", f"{doc.get('size_mb', 0)} MB")
    else:
        st.info("No documents uploaded yet. Go to Chat tab to upload PDFs.")

# ==================== COMPARE TAB ====================
elif st.session_state.current_tab == "compare":
    st.subheader("Compare Research Papers")

    if st.session_state.doc_manager.get_document_count() > 1:
        doc_names = st.session_state.doc_manager.get_document_names()
        
        selected_docs = st.multiselect("Select papers to compare:", doc_names)
        question = st.text_input("What would you like to compare?")
        
        if st.button("Compare", type="primary"):
            if len(selected_docs) < 2:
                st.warning("Please select at least 2 documents")
            elif not question:
                st.warning("Please enter a comparison question")
            else:
                st.info("Comparison engine will analyze the selected papers...")
    else:
        st.info("Upload at least 2 documents to use the comparison feature")

# ==================== EXPORT TAB ====================
elif st.session_state.current_tab == "export":
    st.subheader("Export Chat History")

    if st.session_state.conversation:
        exporter = ChatExporter()
        
        format_type = st.selectbox("Select export format:", ["JSON", "CSV", "TXT"])
        
        if st.button("Export Chat", type="primary"):
            try:
                path = exporter.export_chat(
                    st.session_state.conversation,
                    format_type.lower(),
                    st.session_state.session_id
                )
                
                st.success("Export ready!")
                
                with open(path, 'r') as f:
                    st.download_button(
                        "Download File",
                        f.read(),
                        file_name=f"chat_export_{st.session_state.session_id}.{format_type.lower()}",
                        mime=f"text/{format_type.lower()}"
                    )
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No chat history to export. Start a conversation first!")

# ==================== PIPELINE ====================