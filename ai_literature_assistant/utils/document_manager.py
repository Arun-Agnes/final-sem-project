# utils/document_manager.py
"""
Enhanced Document Management System
Tracks document metadata, upload times, processing status, and document details
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import hashlib


class DocumentManager:
    """Manages document metadata and tracking"""
    
    def __init__(self):
        self.docs: List[Dict] = []
    
    def document_exists(self, file_name: str, file_size: int = None) -> bool:
        """
        Check if a document already exists
        
        Args:
            file_name: Name of the file to check
            file_size: Optional file size for more accurate matching
            
        Returns:
            True if document exists, False otherwise
        """
        for doc in self.docs:
            # Check by file name
            if doc['name'] == file_name:
                # If file size is provided, also check size for accuracy
                if file_size is None or doc['size_mb'] * (1024 * 1024) == file_size:
                    return True
        return False
    
    def get_document_by_name(self, file_name: str) -> Optional[Dict]:
        """
        Get document by file name
        
        Args:
            file_name: Name of the file
            
        Returns:
            Document info dictionary or None if not found
        """
        for doc in self.docs:
            if doc['name'] == file_name:
                return doc
        return None

    def add_document(self, file_name: str, file_size: int, file_hash: str = None) -> Dict:
        """
        Add a new document to tracking
        
        Args:
            file_name: Name of the uploaded file
            file_size: Size in bytes
            file_hash: Optional hash for uniqueness
            
        Returns:
            Document info dictionary
        """
        if file_hash is None:
            file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        
        doc_info = {
            'id': file_hash,
            'name': file_name,
            'upload_time': datetime.now(),
            'num_chunks': 0,
            'num_pages': 0,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'status': 'processing',
            'image_count': 0,
            'diagram_count': 0,
            'has_abstract': False,
            'has_keywords': False
        }
        
        self.docs.append(doc_info)
        return doc_info
    
    def update_document(self, doc_id: str, **kwargs):
        """
        Update document metadata
        
        Args:
            doc_id: Document ID to update
            **kwargs: Fields to update
        """
        for doc in self.docs:
            if doc['id'] == doc_id:
                doc.update(kwargs)
                break
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        for doc in self.docs:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> pd.DataFrame:
        """
        Get all documents as a DataFrame
        
        Returns:
            DataFrame with document information
        """
        if not self.docs:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.docs)
        
        # Format upload time
        if 'upload_time' in df.columns:
            df['upload_time'] = df['upload_time'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, datetime) else x
            )
        
        # Convert string columns to object dtype to prevent dtype incompatibility
        string_columns = df.select_dtypes(include=['object', 'string']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).astype('object')
        
        return df
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.docs)
    
    def get_total_pages(self) -> int:
        """Get total pages across all documents"""
        return sum(doc.get('num_pages', 0) for doc in self.docs)
    
    def get_total_chunks(self) -> int:
        """Get total chunks across all documents"""
        return sum(doc.get('num_chunks', 0) for doc in self.docs)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from tracking
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, doc in enumerate(self.docs):
            if doc['id'] == doc_id:
                self.docs.pop(i)
                return True
        return False
    
    def clear_all(self):
        """Clear all documents"""
        self.docs = []
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.docs:
            return {
                'total_documents': 0,
                'total_pages': 0,
                'total_chunks': 0,
                'total_size_mb': 0,
                'avg_pages_per_doc': 0,
                'processed_count': 0,
                'failed_count': 0
            }
        
        processed = [d for d in self.docs if d['status'] == 'completed']
        failed = [d for d in self.docs if d['status'] == 'failed']
        
        return {
            'total_documents': len(self.docs),
            'total_pages': self.get_total_pages(),
            'total_chunks': self.get_total_chunks(),
            'total_size_mb': round(sum(d.get('size_mb', 0) for d in self.docs), 2),
            'avg_pages_per_doc': round(self.get_total_pages() / len(self.docs), 1) if self.docs else 0,
            'processed_count': len(processed),
            'failed_count': len(failed)
        }
