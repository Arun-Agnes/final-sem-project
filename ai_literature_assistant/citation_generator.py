"""
IEEE Citation Generator for Research Papers
Generates IEEE-formatted citations from paper metadata
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class CitationMetadata:
    """Metadata for citation generation"""
    title: str = ""
    authors: List[str] = None
    year: Optional[int] = None
    journal_conference: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    publisher: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []

class IEEECitationGenerator:
    """Generates IEEE-formatted citations"""
    
    def __init__(self):
        self.month_abbreviations = {
            'January': 'Jan.', 'February': 'Feb.', 'March': 'Mar.',
            'April': 'Apr.', 'May': 'May', 'June': 'Jun.',
            'July': 'Jul.', 'August': 'Aug.', 'September': 'Sep.',
            'October': 'Oct.', 'November': 'Nov.', 'December': 'Dec.'
        }
    
    def extract_citation_metadata(self, text: str) -> CitationMetadata:
        """Extract citation metadata from paper text"""
        metadata = CitationMetadata()
        lines = text.split('\n')
        
        # Extract title (first substantial line)
        title_found = False
        for line in lines[:20]:
            clean_line = line.strip()
            if (len(clean_line) > 10 and len(clean_line) < 300 and 
                not clean_line.lower().startswith(('abstract', 'introduction', 'keywords', 'author', 'by')) and
                not re.search(r'\d{4}', clean_line) and  # Skip lines with years
                not re.search(r'\b(email|www\.|http|doi:)\b', clean_line, re.IGNORECASE) and
                not clean_line.count(',') > 3):  # Skip lines with many commas (likely author lists)
                metadata.title = clean_line
                title_found = True
                break
        
        # Extract authors - simplified approach
        # Look for lines that are likely to contain author names
        author_lines = []
        start_idx = 0
        if title_found:
            for i, line in enumerate(lines[:20]):
                if metadata.title in line:
                    start_idx = i + 1
                    break
        
        # Collect potential author lines
        for line in lines[start_idx:start_idx + 10]:
            clean_line = line.strip()
            
            # Stop at major sections
            if re.match(r'^(abstract|introduction|keywords|\d+\.)', clean_line, re.IGNORECASE):
                break
            
            # Skip empty lines, title, or obvious affiliations
            if (not clean_line or 
                clean_line == metadata.title or
                any(word in clean_line.lower() for word in ['university', 'institute', 'college', 'department', 'lab', 'google', 'microsoft', 'deepmind'])):
                continue
            
            # Add lines that might contain authors
            if (len(clean_line) < 200 and 
                re.search(r'[A-Z][a-z]+', clean_line) and
                not clean_line.lower().startswith(('abstract', 'introduction', 'conclusion', 'references'))):
                author_lines.append(clean_line)
        
        # Extract author names from collected lines
        all_authors = []
        for line in author_lines:
            # Split by common separators
            parts = re.split(r',\s*|\sand\s+|\&\s*', line)
            
            for part in parts:
                part = part.strip()
                # Look for name patterns
                if (len(part) > 3 and len(part) < 50 and
                    re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', part) and
                    not part.lower() in ['abstract', 'introduction', 'conclusion', 'references']):
                    all_authors.append(part)
        
        # Remove duplicates and limit
        seen = set()
        unique_authors = []
        for author in all_authors:
            author_lower = author.lower()
            if author_lower not in seen and len(unique_authors) < 8:
                seen.add(author_lower)
                unique_authors.append(author)
        
        metadata.authors = unique_authors
        
        # Extract year (look for 4-digit years in reasonable range)
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        for year_str in year_matches:
            try:
                year = int(year_str)
                if 2020 <= year <= 2030:  # More recent range for papers
                    metadata.year = year
                    break
            except ValueError:
                continue
        
        # Extract DOI
        doi_match = re.search(r'doi:\s*([^\s\n]+)', text, re.IGNORECASE)
        if doi_match:
            metadata.doi = doi_match.group(1)
        
        # Extract journal/conference (look for common patterns)
        journal_patterns = [
            r'proceedings?\s+of\s+([^,\n]+)',
            r'in\s+([^,\n]+)\s+(conference|symposium|workshop)',
            r'([^,\n]+)\s+journal',
            r'([^,\n]+)\s+conference',
            r'([^,\n]+)\s+symposium',
            r'([^,\n]+)\s+workshop',
        ]
        
        for pattern in journal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                journal = match.group(1).strip()
                if len(journal) > 5 and len(journal) < 100:  # Reasonable length
                    metadata.journal_conference = journal
                    break
        
        return metadata
    
    def format_author_name(self, author: str) -> str:
        """Format author name in IEEE style: First Initial. Last Name"""
        # Handle different name formats
        if ', ' in author:
            # "Last, First" format
            parts = author.split(', ')
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_part = parts[1].strip()
                # Extract first initial
                first_initial = first_part[0].upper() if first_part else ''
                return f"{first_initial}. {last_name}"
        
        # Handle "First I. Last" or "First Middle Last" format
        words = author.split()
        if len(words) >= 2:
            last_name = words[-1]
            first_initial = words[0][0].upper()
            
            # Check if there's already a middle initial
            if len(words) >= 3 and '.' in words[1]:
                return f"{first_initial}. {words[1]} {last_name}"
            else:
                return f"{first_initial}. {last_name}"
        
        return author
    
    def generate_ieee_citation(self, metadata: CitationMetadata) -> str:
        """Generate IEEE-formatted citation"""
        if not metadata.title:
            return "Insufficient information to generate citation"
        
        citation_parts = []
        
        # Authors (IEEE format: First Initial. Last Name)
        if metadata.authors:
            formatted_authors = []
            for i, author in enumerate(metadata.authors):
                formatted = self.format_author_name(author)
                if i == len(metadata.authors) - 1 and len(metadata.authors) > 1:
                    # Last author - add "and"
                    formatted_authors.append(f"and {formatted}")
                else:
                    formatted_authors.append(formatted)
            
            # Join authors with commas
            if len(formatted_authors) == 1:
                citation_parts.append(formatted_authors[0])
            else:
                citation_parts.append(', '.join(formatted_authors[:-1]) + ', ' + formatted_authors[-1])
        
        # Title in quotes
        citation_parts.append(f'"{metadata.title}"')
        
        # Journal/Conference (italicized in real IEEE, we'll use quotes)
        if metadata.journal_conference:
            citation_parts.append(f"*{metadata.journal_conference}*")
        
        # Volume and Issue
        if metadata.volume:
            vol_issue = f"vol. {metadata.volume}"
            if metadata.issue:
                vol_issue += f", no. {metadata.issue}"
            citation_parts.append(vol_issue)
        
        # Pages
        if metadata.pages:
            citation_parts.append(f"pp. {metadata.pages}")
        
        # Month and Year
        if metadata.year:
            citation_parts.append(str(metadata.year))
        
        # DOI
        if metadata.doi:
            citation_parts.append(f"DOI: {metadata.doi}")
        
        # URL (if no DOI)
        if not metadata.doi and metadata.url:
            citation_parts.append(f"Available: {metadata.url}")
        
        return '. '.join(citation_parts) + '.'
    
    def generate_bibtex(self, metadata: CitationMetadata) -> str:
        """Generate BibTeX entry"""
        if not metadata.title:
            return "% Insufficient information to generate BibTeX"
        
        # Create citation key
        key = ""
        if metadata.authors:
            first_author = metadata.authors[0].split()[-1].lower()
            key = first_author
        if metadata.year:
            key += str(metadata.year)
        if not key:
            key = "unknown"
        
        bibtex = f"@article{{{key},\n"
        bibtex += f'  title="{{{metadata.title}}}",\n'
        
        if metadata.authors:
            authors_str = " and ".join(metadata.authors)
            bibtex += f'  author="{{{authors_str}}}",\n'
        
        if metadata.journal_conference:
            bibtex += f'  journal="{{{metadata.journal_conference}}}",\n'
        
        if metadata.year:
            bibtex += f'  year={{{metadata.year}}},\n'
        
        if metadata.volume:
            bibtex += f'  volume="{{{metadata.volume}}}",\n'
        
        if metadata.issue:
            bibtex += f'  number="{{{metadata.issue}}}",\n'
        
        if metadata.pages:
            bibtex += f'  pages="{{{metadata.pages}}}",\n'
        
        if metadata.doi:
            bibtex += f'  doi="{{{metadata.doi}}}",\n'
        
        # Remove trailing comma and close
        bibtex = bibtex.rstrip(',\n') + '\n}'
        
        return bibtex
    
    def generate_all_formats(self, text: str) -> Dict[str, str]:
        """Generate all citation formats"""
        metadata = self.extract_citation_metadata(text)
        
        return {
            'ieee': self.generate_ieee_citation(metadata),
            'bibtex': self.generate_bibtex(metadata),
            'metadata': {
                'title': metadata.title,
                'authors': metadata.authors,
                'year': metadata.year,
                'journal': metadata.journal_conference,
                'doi': metadata.doi
            }
        }
