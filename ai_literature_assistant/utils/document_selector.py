# utils/document_selector.py
"""
Document Selection and Filtering System
Allows users to select specific documents for querying
"""

from typing import List, Optional, Set


class DocumentSelector:
    """Manages document selection for scoped queries"""
    
    def __init__(self):
        self.available_docs: Set[str] = set()
        self.selected_docs: Set[str] = set()
        self.search_mode: str = "all"  # all, selected, current
        
    def add_available_document(self, doc_id: str):
        """Add a document to available pool"""
        self.available_docs.add(doc_id)
    
    def remove_available_document(self, doc_id: str):
        """Remove document from available pool"""
        self.available_docs.discard(doc_id)
        self.selected_docs.discard(doc_id)
    
    def set_selected_documents(self, doc_ids: List[str]):
        """Set the selected documents for querying"""
        self.selected_docs = set(doc_ids) & self.available_docs
    
    def add_to_selection(self, doc_id: str):
        """Add a document to selection"""
        if doc_id in self.available_docs:
            self.selected_docs.add(doc_id)
    
    def remove_from_selection(self, doc_id: str):
        """Remove a document from selection"""
        self.selected_docs.discard(doc_id)
    
    def clear_selection(self):
        """Clear all selections"""
        self.selected_docs.clear()
    
    def select_all(self):
        """Select all available documents"""
        self.selected_docs = self.available_docs.copy()
    
    def set_search_mode(self, mode: str):
        """
        Set search mode
        
        Args:
            mode: 'all', 'selected', or 'current'
        """
        if mode in ['all', 'selected', 'current']:
            self.search_mode = mode
    
    def get_query_filter(self) -> Optional[dict]:
        """
        Get ChromaDB where filter based on current selection
        
        Returns:
            Filter dict for ChromaDB query or None for all documents
        """
        if self.search_mode == "all":
            return None
        elif self.search_mode == "selected" and self.selected_docs:
            return {"paper_id": {"$in": list(self.selected_docs)}}
        elif self.search_mode == "current" and len(self.selected_docs) == 1:
            return {"paper_id": list(self.selected_docs)[0]}
        else:
            return None
    
    def get_selected_count(self) -> int:
        """Get count of selected documents"""
        return len(self.selected_docs)
    
    def get_available_count(self) -> int:
        """Get count of available documents"""
        return len(self.available_docs)
    
    def is_selected(self, doc_id: str) -> bool:
        """Check if a document is selected"""
        return doc_id in self.selected_docs
