# utils/comparison_tools.py
"""
Multi-Document Comparison Tools
Compare information across multiple research papers
"""

import pandas as pd
from typing import List, Dict, Optional
from st_aggrid import AgGrid, GridOptionsBuilder


class DocumentComparator:
    """Tools for comparing information across multiple documents"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_papers(
        self,
        query: str,
        doc_ids: List[str],
        retriever_func,
        generator_func,
        doc_names: Dict[str, str] = None
    ) -> Dict[str, Dict]:
        """
        Compare how different papers answer the same question
        
        Args:
            query: The question to ask
            doc_ids: List of document IDs to compare
            retriever_func: Function to retrieve chunks for a document
            generator_func: Function to generate answer from chunks
            doc_names: Optional mapping of doc_id to readable names
            
        Returns:
            Dictionary mapping doc_id to answer and supporting chunks
        """
        results = {}
        
        for doc_id in doc_ids:
            try:
                # Retrieve chunks from this specific document
                chunks = retriever_func(query, doc_filter=[doc_id])
                
                # Generate answer
                answer = generator_func(query, chunks) if chunks else "No relevant information found."
                
                doc_name = doc_names.get(doc_id, doc_id) if doc_names else doc_id
                
                results[doc_id] = {
                    'doc_name': doc_name,
                    'answer': answer,
                    'chunks': chunks,
                    'num_sources': len(chunks),
                    'avg_relevance': sum(c.get('score', 0) for c in chunks) / len(chunks) if chunks else 0
                }
            except Exception as e:
                results[doc_id] = {
                    'doc_name': doc_names.get(doc_id, doc_id) if doc_names else doc_id,
                    'answer': f"Error: {str(e)}",
                    'chunks': [],
                    'num_sources': 0,
                    'avg_relevance': 0
                }
        
        self.comparison_results = results
        return results
    
    def create_feature_table(
        self,
        doc_ids: List[str],
        features: List[str],
        extractor_func,
        doc_names: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Create a comparison table of features across papers
        
        Args:
            doc_ids: List of document IDs
            features: List of features to extract (e.g., "Dataset", "Methodology")
            extractor_func: Function to extract feature value from a document
            doc_names: Optional mapping of doc_id to readable names
            
        Returns:
            DataFrame with papers as rows and features as columns
        """
        comparison_data = []
        
        for doc_id in doc_ids:
            doc_name = doc_names.get(doc_id, doc_id) if doc_names else doc_id
            row = {'Paper': doc_name}
            
            for feature in features:
                try:
                    # Extract this feature from the paper
                    query = f"What {feature} does this paper use?"
                    value = extractor_func(doc_id, query)
                    row[feature] = value
                except Exception as e:
                    row[feature] = "N/A"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_consensus_answer(
        self,
        comparison_results: Dict[str, Dict],
        threshold: float = 0.7
    ) -> str:
        """
        Find consensus across multiple paper answers
        
        Args:
            comparison_results: Results from compare_papers
            threshold: Similarity threshold for consensus
            
        Returns:
            Consensus answer or indication of disagreement
        """
        answers = [r['answer'] for r in comparison_results.values() if r['answer']]
        
        if not answers:
            return "No answers available."
        
        # Simple consensus: if most answers are similar length and non-empty
        avg_length = sum(len(a) for a in answers) / len(answers)
        consistent = sum(1 for a in answers if abs(len(a) - avg_length) < avg_length * 0.3)
        
        if consistent / len(answers) >= threshold:
            # Return the longest answer as representative
            return max(answers, key=len)
        else:
            return "Papers provide different perspectives. See individual answers for details."
    
    def highlight_differences(
        self,
        comparison_results: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Identify key differences between paper answers
        
        Args:
            comparison_results: Results from compare_papers
            
        Returns:
            List of difference highlights
        """
        differences = []
        
        answers = [(doc_id, r['answer']) for doc_id, r in comparison_results.items()]
        
        # Simple difference detection based on answer length and content
        for i, (doc_id_1, ans_1) in enumerate(answers):
            for doc_id_2, ans_2 in answers[i+1:]:
                # If answers are very different in length
                if abs(len(ans_1) - len(ans_2)) > max(len(ans_1), len(ans_2)) * 0.5:
                    differences.append({
                        'type': 'length',
                        'papers': [doc_id_1, doc_id_2],
                        'description': f"Significant difference in answer depth"
                    })
        
        return differences


class FeatureExtractor:
    """Extract specific features from research papers"""
    
    COMMON_FEATURES = [
        "Dataset",
        "Methodology",
        "Model Architecture",
        "Evaluation Metrics",
        "Key Results",
        "Baseline Comparisons",
        "Limitations",
        "Future Work",
        "Publication Year",
        "Main Contribution"
    ]
    
    @staticmethod
    def extract_feature(
        doc_id: str,
        feature_name: str,
        retriever_func,
        generator_func,
        max_length: int = 100
    ) -> str:
        """
        Extract a specific feature from a paper
        
        Args:
            doc_id: Document ID
            feature_name: Feature to extract
            retriever_func: Retrieval function
            generator_func: Generation function
            max_length: Maximum length of extracted text
            
        Returns:
            Extracted feature value
        """
        # Create targeted query
        query = f"What {feature_name.lower()} is used or mentioned in this paper?"
        
        try:
            # Retrieve relevant chunks
            chunks = retriever_func(query, doc_filter=[doc_id], top_k=3)
            
            if not chunks:
                return "Not mentioned"
            
            # Generate concise answer
            answer = generator_func(query, chunks)
            
            # Truncate if too long
            if len(answer) > max_length:
                answer = answer[:max_length] + "..."
            
            return answer
        except Exception as e:
            return f"Error: {str(e)}"
