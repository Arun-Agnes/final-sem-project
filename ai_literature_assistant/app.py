# app.py - AI Research Literature Assistant (Updated with Proper Sidebar Navigation)
"""
AI Research Literature Assistant
Updated UI with Streamlit-native sidebar navigation
"""

import streamlit as st
import os
from datetime import datetime
import uuid

# Third-party imports
import chromadb

# Local imports
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.preprocess import ResearchPaperChunker
from ingestion.embed_store import store_documents_enhanced, CHROMA_DIR
from rag.pipeline import RAGPipeline
from citation_generator import IEEECitationGenerator

# Utility imports
from utils.document_manager import DocumentManager
from utils.document_selector import DocumentSelector
from utils.metrics_tracker import MetricsTracker, ConfidenceCalculator
from utils.ui_components import SearchModeSelector
from utils.export_tools import ChatExporter
from utils.comparison_tools import DocumentComparator

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
        # If CSS file is not found, continue without custom styling
        pass

# Load custom CSS
load_css()

# ==================== SESSION INITIALIZATION ====================
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

# ==================== HELPER FUNCTIONS ====================

def process_uploaded_files(uploaded_files) -> bool:
    try:
        all_chunks = []
        all_metadata = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.uploaded_files:
                continue
                
            paper_data = {
                'title': uploaded_file.name,
                'authors': [],
                'abstract': '',
                'content': extract_text_from_pdf(uploaded_file.read())
            }
            
            chunker = ResearchPaperChunker()
            chunks = chunker.chunk_document(paper_data)
            
            metadata = {
                'title': paper_data.get('title', uploaded_file.name),
                'authors': paper_data.get('authors', []),
                'abstract': paper_data.get('abstract', ''),
                'file_name': uploaded_file.name,
                'upload_time': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }
            
            all_chunks.extend(chunks)
            all_metadata.append(metadata)
            st.session_state.uploaded_files.add(uploaded_file.name)
        
        if all_chunks:
            store_documents_enhanced(all_chunks, all_metadata)
            return True
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return False


def render_chat_message(qa: dict):
    with st.chat_message(qa["role"], avatar="🤖" if qa["role"] == "assistant" else "👤"):
        st.markdown(qa["content"])
        
        if st.session_state.show_sources and qa.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(qa["sources"][:3]):
                    st.markdown(f"**Source {i+1}:** {source.get('title', 'Unknown')}")
                    st.markdown(f"- Page {source.get('page', 'N/A')}")
                    st.markdown(f"- Relevance: {source.get('relevance_score', 'N/A'):.2f}")


# ==================== CUSTOM COLLAPSIBLE SIDEBAR ====================

# ==================== STREAMLIT-NATIVE COLLAPSIBLE SIDEBAR ====================
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

sidebar_width = 60 if st.session_state.sidebar_collapsed else 220
toggle_icon = '➡️' if st.session_state.sidebar_collapsed else '⬅️'

st.markdown(f"""
    <style>
    .custom-sidebar {{
        position: fixed;
        left: 0;
        top: 0;
        height: 100vh;
        width: {sidebar_width}px;
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: width 0.2s;
    }}
    .main-content {{
        margin-left: {sidebar_width}px !important;
        transition: margin-left 0.2s;
    }}
    .sidebar-label {{
        display: {'none' if st.session_state.sidebar_collapsed else 'inline'};
        margin-left: 0.5rem;
        font-size: 1rem;
    }}
    .sidebar-btn {{
        width: 90%;
        margin-bottom: 1rem;
        padding: 0.7rem 0;
        border: none;
        background: transparent;
        color: #374151;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        cursor: pointer;
        border-radius: 6px;
        justify-content: {'center' if st.session_state.sidebar_collapsed else 'flex-start'};
    }}
    .sidebar-clear-btn {{
        width: 90%;
        margin-bottom: 1rem;
        padding: 0.7rem 0;
        border: none;
        background: #6B7280;
        color: white;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        cursor: pointer;
        border-radius: 6px;
        justify-content: {'center' if st.session_state.sidebar_collapsed else 'flex-start'};
    }}
    </style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components
with st.container():
    # Toggle button
    if st.button(toggle_icon, key="sidebar-toggle-btn"):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
    st.markdown(f"""
        <div class="custom-sidebar">
            <div style='width:100%;text-align:center;margin-bottom:2rem;margin-top:2.5rem;'>
                <span style='font-size:2rem;'>📚</span>
                <span class="sidebar-label" style='font-weight:600;font-size:1.1rem;color:#6B7280;'>Research Assistant</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # Navigation buttons
    if st.button("💬 Chat", key="nav-chat", help="Chat", use_container_width=True):
        st.session_state.current_tab = "chat"
    if st.button("📄 Documents", key="nav-docs", help="Documents", use_container_width=True):
        st.session_state.current_tab = "documents"
    if st.button("🔍 Compare", key="nav-compare", help="Compare", use_container_width=True):
        st.session_state.current_tab = "compare"
    if st.button("⬇ Export", key="nav-export", help="Export", use_container_width=True):
        st.session_state.current_tab = "export"
    st.markdown("<hr style='margin:2rem 0;border:0;border-top:1px solid #eee;'>", unsafe_allow_html=True)
    if st.button("🧹 Clear Session", key="nav-clear", help="Clear Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ==================== MAIN HEADER ====================

st.title("AI Research Literature Assistant")
st.caption("Intelligent document analysis powered by RAG")

# ==================== CHAT TAB ====================

if st.session_state.current_tab == "chat":

    if st.session_state.doc_manager.get_document_count() == 0:
        uploaded_files = st.file_uploader(
            "Upload Research PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if process_uploaded_files(uploaded_files):
                st.success("Documents processed successfully!")
                st.rerun()
    
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
            pipeline = get_pipeline()
            
            if pipeline:
                result = pipeline.query(
                    user_query,
                    search_mode=st.session_state.search_mode,
                    alpha=st.session_state.alpha
                )
                
                st.session_state.conversation.append({"role": "user", "content": user_query})
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": result.get('response', ''),
                    "sources": result.get('sources', [])
                })
                
                st.rerun()

# ==================== DOCUMENTS TAB ====================

elif st.session_state.current_tab == "documents":

    st.subheader("Document Library")

    if st.session_state.doc_manager.get_document_count() > 0:
        docs = st.session_state.doc_manager.get_all_documents()
        
        for doc in docs:
            with st.expander(doc.get('title', 'Unknown')):
                st.write("Authors:", ', '.join(doc.get('authors', [])))
                st.write("Uploaded:", doc.get('upload_time'))
                st.write("Chunks:", doc.get('chunk_count'))
                st.write("File:", doc.get('file_name'))
    else:
        st.info("No documents uploaded yet")

# ==================== COMPARE TAB ====================

elif st.session_state.current_tab == "compare":

    st.subheader("Compare Research Papers")

    if st.session_state.doc_manager.get_document_count() > 1:
        doc_names = st.session_state.doc_manager.get_document_names()
        
        selected_docs = st.multiselect("Select papers:", doc_names)
        question = st.text_input("Comparison question:")
        
        if st.button("Compare"):
            st.info("Comparison engine will run here.")
    else:
        st.info("Upload at least 2 documents to compare")

# ==================== EXPORT TAB ====================

elif st.session_state.current_tab == "export":

    st.subheader("Export Chat History")

    if st.session_state.conversation:
        exporter = ChatExporter()
        
        format_type = st.selectbox("Format", ["JSON", "CSV", "TXT"])
        
        if st.button("Export"):
            path = exporter.export_chat(
                st.session_state.conversation,
                format_type.lower(),
                st.session_state.session_id
            )
            
            st.success("Export ready")
            
            with open(path, 'r') as f:
                st.download_button(
                    "Download File",
                    f.read(),
                    file_name=f"chat_export.{format_type.lower()}"
                )
    else:
        st.info("No chat history to export")

# ==================== PIPELINE ====================

@st.cache_resource
def get_pipeline():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("research_papers")
        
        if collection.count() == 0:
            return None
        
        pipeline = RAGPipeline()
        pipeline.initialize()
        return pipeline
    except Exception:
        return None
