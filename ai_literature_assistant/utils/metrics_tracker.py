# utils/metrics_tracker.py
"""
Metrics Tracking and Dashboard System
Tracks query performance, retrieval quality, and system metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px


class MetricsTracker:
    """Tracks and analyzes system performance metrics"""
    
    def __init__(self):
        self.queries: List[Dict] = []
        
    def log_query(
        self,
        query: str,
        retrieval_time: float,
        generation_time: float,
        similarity_scores: List[float],
        num_chunks: int,
        response_length: int,
        search_mode: str = "semantic",
        confidence: float = None
    ):
        """
        Log a query with all metrics
        """
        
        # Clean and convert similarity scores to floats
        cleaned_scores = []
        for score in similarity_scores:
            if isinstance(score, str):
                # Handle percentage strings like '28.5%'
                if score.endswith('%'):
                    try:
                        score = float(score.rstrip('%')) / 100
                    except ValueError:
                        score = 0.0
                else:
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0.0
            cleaned_scores.append(float(score))
        
        similarity_scores = cleaned_scores
        
        query_data = {
            "timestamp": datetime.now(),
            "query": query,
            "query_length": len(query),
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "num_chunks": num_chunks,
            "response_length": response_length,
            'avg_similarity': float(np.mean(similarity_scores)) if similarity_scores and len(similarity_scores) > 0 else 0.0,
            'max_similarity': float(max(similarity_scores)) if similarity_scores and len(similarity_scores) > 0 else 0.0,
            'min_similarity': float(min(similarity_scores)) if similarity_scores and len(similarity_scores) > 0 else 0.0,
            'score_variance': float(np.var(similarity_scores)) if similarity_scores and len(similarity_scores) > 0 else 0.0,
            "search_mode": search_mode,
            "confidence": confidence
        }
        
        self.queries.append(query_data)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get all queries as DataFrame"""
        if not self.queries:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(self.queries)
            
            # Format timestamp
            if 'timestamp' in df.columns:
                df['time_str'] = df['timestamp'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else ''
                )
            
            # Explicitly convert object/string columns to prevent dtype incompatibility
            # Only convert columns that are object and not expected to be numeric
            numeric_cols = [
                'retrieval_time', 'generation_time', 'total_time', 'confidence', 'avg_similarity',
                'num_chunks', 'num_pages', 'image_count', 'diagram_count'
            ]
            for col in df.columns:
                if (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])) and col not in numeric_cols:
                    df[col] = df[col].astype(str)
            
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.queries:
            return {
                'total_queries': 0,
                'avg_retrieval_time': 0,
                'avg_generation_time': 0,
                'avg_total_time': 0,
                'avg_confidence': 0,
                'avg_similarity': 0
            }
        
        df = self.get_dataframe()
        
        if df.empty:
            return {
                'total_queries': 0,
                'avg_retrieval_time': 0.0,
                'avg_generation_time': 0.0,
                'avg_total_time': 0.0,
                'avg_confidence': 0.0,
                'avg_similarity': 0.0,
                'max_retrieval_time': 0.0,
                'min_retrieval_time': 0.0
            }
        
        return {
            'total_queries': len(self.queries),
            'avg_retrieval_time': float(df['retrieval_time'].mean()) if 'retrieval_time' in df else 0.0,
            'avg_generation_time': float(df['generation_time'].mean()) if 'generation_time' in df else 0.0,
            'avg_total_time': float(df['total_time'].mean()) if 'total_time' in df else 0.0,
            'avg_confidence': float(df['confidence'].mean()) if 'confidence' in df else 0.0,
            'avg_similarity': float(df['avg_similarity'].mean()) if 'avg_similarity' in df else 0.0,
            'max_retrieval_time': float(df['retrieval_time'].max()) if 'retrieval_time' in df else 0.0,
            'min_retrieval_time': float(df['retrieval_time'].min()) if 'retrieval_time' in df else 0.0
        }
    
    def create_response_time_chart(self) -> go.Figure:
        """Create response time line chart"""
        df = self.get_dataframe()
        
        if df.empty or 'retrieval_time' not in df or 'generation_time' not in df:
            return go.Figure().add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )
        
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(df.index),
                y=list(df['retrieval_time'].astype(float)),
                name='Retrieval Time',
                mode='lines+markers',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(df.index),
                y=list(df['generation_time'].astype(float)),
                name='Generation Time',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig.update_layout(
                title='Response Time Over Queries',
                xaxis_title='Query Number',
                yaxis_title='Time (seconds)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            return fig
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"Error creating chart: {str(e)}",
                showarrow=False,
                font=dict(size=20)
            )
    
    def create_similarity_distribution(self) -> go.Figure:
        """Create similarity score distribution histogram"""
        df = self.get_dataframe()
        
        if df.empty or 'avg_similarity' not in df:
            return go.Figure().add_annotation(
                text="No similarity data available",
                showarrow=False,
                font=dict(size=20)
            )
        
        try:
            fig = px.histogram(
                df,
                x='avg_similarity',
                nbins=20,
                title='Retrieval Quality Distribution',
                labels={'avg_similarity': 'Average Similarity Score'},
                color_discrete_sequence=['#2ca02c']
            )
            
            fig.update_layout(
                xaxis_title='Average Similarity Score',
                yaxis_title='Count',
                template='plotly_white'
            )
            
            return fig
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"Error creating distribution chart: {str(e)}",
                showarrow=False,
                font=dict(size=20)
            )
    
    def create_confidence_chart(self) -> go.Figure:
        """Create confidence score chart"""
        df = self.get_dataframe()
        
        if df.empty or 'confidence' not in df.columns:
            return go.Figure().add_annotation(
                text="No confidence data available",
                showarrow=False,
                font=dict(size=20)
            )
        
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(df.index),
                y=list(df['confidence'].astype(float)),
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#9467bd', width=2),
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            
            # Add confidence thresholds
            fig.add_hline(y=70, line_dash="dash", line_color="green",
                         annotation_text="High Confidence")
            fig.add_hline(y=50, line_dash="dash", line_color="orange",
                         annotation_text="Medium Confidence")
            
            fig.update_layout(
                title='Confidence Scores Over Time',
                xaxis_title='Query Number',
                yaxis_title='Confidence (%)',
                yaxis_range=[0, 100],
                hovermode='x unified',
                template='plotly_white'
            )
            
            return fig
        except Exception as e:
            return go.Figure().add_annotation(
                text=f"Error creating confidence chart: {str(e)}",
                showarrow=False,
                font=dict(size=20)
            )
    
    def create_performance_summary(self) -> go.Figure:
        """Create performance summary gauge charts"""
        stats = self.get_summary_stats()
        
        fig = go.Figure()
        
        # Create subplots for different metrics
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Avg Retrieval Time', 'Avg Confidence', 'Avg Similarity'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Retrieval time gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stats['avg_retrieval_time'],
            title={'text': "seconds"},
            gauge={'axis': {'range': [0, 5]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 1], 'color': "lightgreen"},
                       {'range': [1, 3], 'color': "yellow"},
                       {'range': [3, 5], 'color': "red"}
                   ]}
        ), row=1, col=1)
        
        # Confidence gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stats['avg_confidence'],
            title={'text': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "purple"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightcoral"},
                       {'range': [50, 70], 'color': "lightyellow"},
                       {'range': [70, 100], 'color': "lightgreen"}
                   ]}
        ), row=1, col=2)
        
        # Similarity gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stats['avg_similarity'] * 100,
            title={'text': "score"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"}}
        ), row=1, col=3)
        
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def clear_metrics(self):
        """Clear all tracked metrics"""
        self.queries = []


class ConfidenceCalculator:
    """Calculates confidence scores for responses"""
    
    @staticmethod
    def calculate_confidence(
        similarity_scores: List[float],
        answer_length: int,
        num_chunks: int,
        min_answer_length: int = 50
    ) -> float:
        """
        Calculate confidence score for a response
        
        Args:
            similarity_scores: List of similarity scores from retrieval
            answer_length: Length of generated answer
            num_chunks: Number of chunks retrieved
            min_answer_length: Minimum expected answer length
            
        Returns:
            Confidence score (0-100)
        """
        if not similarity_scores:
            return 0.0
        
        # Ensure all similarity scores are floats
        try:
            similarity_scores = [float(score) for score in similarity_scores]
        except (ValueError, TypeError):
            return 0.0
        
        # Handle any remaining percentage strings or invalid values
        cleaned_scores = []
        for score in similarity_scores:
            if isinstance(score, str):
                if score.endswith('%'):
                    try:
                        score = float(score.rstrip('%')) / 100
                    except ValueError:
                        score = 0.0
                else:
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0.0
            cleaned_scores.append(float(score))
        
        similarity_scores = cleaned_scores
        
        if not similarity_scores or len(similarity_scores) == 0:
            return 0.0
        
        # Factor 1: Average similarity (50% weight)
        if similarity_scores and len(similarity_scores) > 0:
            avg_similarity = float(np.mean(similarity_scores))
        else:
            avg_similarity = 0.0
        similarity_factor = avg_similarity * 0.5
        
        # Factor 2: Top similarity (30% weight)
        if similarity_scores and len(similarity_scores) > 0:
            max_similarity = float(max(similarity_scores))
        else:
            max_similarity = 0.0
        top_factor = max_similarity * 0.3
        
        # Factor 3: Score consistency - lower variance is better (10% weight)
        if similarity_scores and len(similarity_scores) > 0:
            score_variance = float(np.var(similarity_scores))
        else:
            score_variance = 0.0
        consistency_factor = (1 - min(score_variance, 1.0)) * 0.1
        
        # Factor 4: Answer completeness (10% weight)
        completeness = min(answer_length / min_answer_length, 1.0)
        completeness_factor = completeness * 0.1
        
        # Combine factors
        confidence = (
            similarity_factor +
            top_factor +
            consistency_factor +
            completeness_factor
        ) * 100
        
        return min(confidence, 100.0)
    
    @staticmethod
    def get_confidence_level(confidence: float) -> tuple:
        """
        Get confidence level and color
        
        Returns:
            (level_name, color, emoji)
        """
        if confidence >= 70:
            return ("High", "green", "🟢")
        elif confidence >= 50:
            return ("Medium", "orange", "🟡")
        else:
            return ("Low", "red", "🔴")
