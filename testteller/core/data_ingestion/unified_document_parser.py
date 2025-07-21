"""
Unified Document Parser for TestTeller

This module provides a unified interface for parsing various document formats
that can be used by both the RAG agent for ingestion and TestWriter for 
test case automation. It extends the existing DocumentLoader with enhanced
parsing capabilities and structured output options.

Supported formats:
- Markdown (.md)
- Text files (.txt)  
- PDF (.pdf)
- Word Documents (.docx)
- Excel files (.xlsx)

Usage:
    # For RAG ingestion
    parser = UnifiedDocumentParser()
    content = await parser.parse_for_rag(file_path)
    
    # For TestWriter automation
    test_cases = await parser.parse_for_automation(file_path)
    
    # Get structured metadata
    metadata = await parser.extract_metadata(file_path)
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Import existing document loader
from .document_loader import DocumentLoader

# Import TestWriter components with fallback
try:
    from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser, TestCase
    HAS_TESTWRITER = True
except ImportError:
    # Define minimal stubs if testwriter is not available
    class TestCase:
        def __init__(self, title="", description="", steps=None, expected_result=""):
            self.title = title
            self.description = description
            self.steps = steps or []
            self.expected_result = expected_result
    
    class MarkdownTestCaseParser:
        def parse_test_cases_from_text(self, text):
            return []
    
    HAS_TESTWRITER = False

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    TEST_CASES = "test_cases"
    REQUIREMENTS = "requirements"
    SPECIFICATIONS = "specifications"  
    DOCUMENTATION = "documentation"
    API_DOCS = "api_docs"
    UNKNOWN = "unknown"


class ParseMode(Enum):
    """Parsing mode for different use cases"""
    RAG_INGESTION = "rag_ingestion"      # For vector store ingestion
    AUTOMATION = "automation"            # For test automation generation
    ANALYSIS = "analysis"                # For document analysis
    METADATA_ONLY = "metadata_only"     # Extract metadata only


@dataclass
class DocumentMetadata:
    """Enhanced document metadata"""
    file_path: str
    file_type: str
    document_type: DocumentType
    title: Optional[str] = None
    sections: List[str] = None
    test_case_count: int = 0
    word_count: int = 0
    character_count: int = 0
    language: str = "en"
    structure_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = []
        if self.structure_info is None:
            self.structure_info = {}


@dataclass  
class ParsedDocument:
    """Unified parsed document result"""
    metadata: DocumentMetadata
    content: str
    structured_content: Dict[str, Any] = None
    test_cases: List[TestCase] = None
    chunks: List[str] = None
    
    def __post_init__(self):
        if self.structured_content is None:
            self.structured_content = {}
        if self.test_cases is None:
            self.test_cases = []
        if self.chunks is None:
            self.chunks = []


class UnifiedDocumentParser:
    """Unified document parser for TestTeller ecosystem"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.markdown_parser = MarkdownTestCaseParser()
        
        # Document type detection patterns
        self.test_case_patterns = [
            r'### Test Case.*(?:\[(\d+)\]|(\w+))',
            r'## Test.*Case',
            r'Test Steps?:',
            r'Expected.*Result:',
            r'Scenario:.*Given.*When.*Then'
        ]
        
        self.requirements_patterns = [
            r'## Requirements?',
            r'### Functional.*Requirements?',
            r'User Story:',
            r'Acceptance Criteria:',
            r'As a.*I want.*So that'
        ]
        
        self.api_patterns = [
            r'## API.*',
            r'### Endpoints?',
            r'HTTP.*Methods?:',
            r'Request.*Body:',
            r'Response.*Schema:'
        ]
    
    async def parse_document(
        self, 
        file_path: Union[str, Path], 
        mode: ParseMode = ParseMode.RAG_INGESTION,
        chunk_size: Optional[int] = None
    ) -> ParsedDocument:
        """
        Parse a document based on the specified mode.
        
        Args:
            file_path: Path to the document
            mode: Parsing mode (RAG_INGESTION, AUTOMATION, ANALYSIS, METADATA_ONLY)
            chunk_size: Optional chunk size for text splitting
            
        Returns:
            ParsedDocument object with parsed content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing document: {file_path} in mode: {mode.value}")
        
        # Extract basic metadata
        metadata = await self._extract_metadata(file_path)
        
        # Load raw content
        raw_content = await self.document_loader.load_document(str(file_path))
        if not raw_content:
            raise ValueError(f"Failed to load content from: {file_path}")
        
        # Update metadata with content stats
        metadata.character_count = len(raw_content)
        metadata.word_count = len(raw_content.split())
        
        # Create parsed document
        parsed_doc = ParsedDocument(
            metadata=metadata,
            content=raw_content
        )
        
        # Mode-specific processing
        if mode == ParseMode.METADATA_ONLY:
            return parsed_doc
        
        elif mode == ParseMode.RAG_INGESTION:
            await self._process_for_rag(parsed_doc, chunk_size)
            
        elif mode == ParseMode.AUTOMATION:
            await self._process_for_automation(parsed_doc)
            
        elif mode == ParseMode.ANALYSIS:
            await self._process_for_analysis(parsed_doc)
        
        return parsed_doc
    
    async def parse_for_rag(
        self, 
        file_path: Union[str, Path], 
        chunk_size: int = 1000
    ) -> ParsedDocument:
        """Parse document for RAG ingestion with chunking."""
        return await self.parse_document(file_path, ParseMode.RAG_INGESTION, chunk_size)
    
    async def parse_for_automation(self, file_path: Union[str, Path]) -> ParsedDocument:
        """Parse document for test automation with structured test cases."""
        return await self.parse_document(file_path, ParseMode.AUTOMATION)
    
    async def parse_for_analysis(self, file_path: Union[str, Path]) -> ParsedDocument:
        """Parse document for analysis with detailed structure."""
        return await self.parse_document(file_path, ParseMode.ANALYSIS)
    
    async def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract only metadata from document."""
        parsed_doc = await self.parse_document(file_path, ParseMode.METADATA_ONLY)
        return parsed_doc.metadata
    
    async def batch_parse(
        self, 
        file_paths: List[Union[str, Path]], 
        mode: ParseMode = ParseMode.RAG_INGESTION,
        max_concurrency: int = 5
    ) -> List[ParsedDocument]:
        """Parse multiple documents concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def parse_single(file_path):
            async with semaphore:
                try:
                    return await self.parse_document(file_path, mode)
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")
                    return None
        
        tasks = [parse_single(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed parses
        return [result for result in results if result is not None]
    
    # Private methods
    
    async def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from document."""
        file_type = file_path.suffix.lower()
        
        metadata = DocumentMetadata(
            file_path=str(file_path),
            file_type=file_type,
            document_type=DocumentType.UNKNOWN
        )
        
        # Basic file info
        metadata.title = file_path.stem
        
        # For text-based files, do quick content analysis
        if file_type in ['.md', '.txt']:
            try:
                # Read first few lines for quick analysis
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_lines = [f.readline().strip() for _ in range(100)]
                    content_sample = '\n'.join(first_lines)
                
                # Detect document type
                metadata.document_type = self._detect_document_type(content_sample)
                
                # Extract title from content
                title = self._extract_title_from_content(content_sample)
                if title:
                    metadata.title = title
                    
            except Exception as e:
                logger.warning(f"Failed to analyze content for metadata: {e}")
        
        return metadata
    
    def _detect_document_type(self, content: str) -> DocumentType:
        """Detect the type of document based on content patterns."""
        import re
        
        content_lower = content.lower()
        
        # Check for test case patterns
        for pattern in self.test_case_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return DocumentType.TEST_CASES
        
        # Check for requirements patterns  
        for pattern in self.requirements_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return DocumentType.REQUIREMENTS
        
        # Check for API documentation patterns
        for pattern in self.api_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return DocumentType.API_DOCS
        
        # Check for specification keywords
        spec_keywords = ['specification', 'design document', 'architecture', 'technical spec']
        if any(keyword in content_lower for keyword in spec_keywords):
            return DocumentType.SPECIFICATIONS
        
        # Default to documentation
        return DocumentType.DOCUMENTATION
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from document content."""
        import re
        
        lines = content.split('\n')
        
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            
            # Markdown H1
            if line.startswith('# '):
                return line[2:].strip()
            
            # Markdown H2 as fallback
            if line.startswith('## '):
                return line[3:].strip()
            
            # First non-empty line that looks like a title
            if line and len(line) > 5 and not line.startswith('-'):
                # Remove common prefixes
                title = re.sub(r'^(title:|subject:|document:)\s*', '', line, flags=re.IGNORECASE)
                if title != line:  # Found a prefix
                    return title.strip()
        
        return None
    
    async def _process_for_rag(self, parsed_doc: ParsedDocument, chunk_size: Optional[int]):
        """Process document for RAG ingestion."""
        content = parsed_doc.content
        
        # Extract sections
        sections = self._extract_sections(content)
        parsed_doc.metadata.sections = sections
        
        # Create chunks if requested
        if chunk_size:
            parsed_doc.chunks = self._create_smart_chunks(content, chunk_size)
        
        # Create structured content for better RAG retrieval
        parsed_doc.structured_content = {
            'title': parsed_doc.metadata.title,
            'sections': {section: self._extract_section_content(content, section) 
                        for section in sections},
            'document_type': parsed_doc.metadata.document_type.value,
            'summary': self._create_summary(content)
        }
    
    async def _process_for_automation(self, parsed_doc: ParsedDocument):
        """Process document for test automation."""
        content = parsed_doc.content
        file_type = parsed_doc.metadata.file_type
        
        # Extract test cases if this is a markdown file
        if file_type == '.md' and parsed_doc.metadata.document_type == DocumentType.TEST_CASES:
            try:
                test_cases = self.markdown_parser.parse_content(content)
                parsed_doc.test_cases = test_cases
                parsed_doc.metadata.test_case_count = len(test_cases)
                
                logger.info(f"Extracted {len(test_cases)} test cases for automation")
                
            except Exception as e:
                logger.error(f"Failed to parse test cases: {e}")
        
        # For other document types, create structured content for automation context
        elif parsed_doc.metadata.document_type in [DocumentType.REQUIREMENTS, DocumentType.API_DOCS]:
            structured_content = self._extract_automation_context(content)
            parsed_doc.structured_content = structured_content
        
        # Create automation-friendly chunks
        parsed_doc.chunks = self._create_automation_chunks(content)
    
    async def _process_for_analysis(self, parsed_doc: ParsedDocument):
        """Process document for detailed analysis."""
        content = parsed_doc.content
        
        # Extract detailed structure
        structure_info = {
            'headings': self._extract_headings(content),
            'sections': self._extract_sections(content),
            'lists': self._extract_lists(content),
            'code_blocks': self._extract_code_blocks(content),
            'tables': self._extract_tables(content),
            'complexity_score': self._calculate_complexity_score(content)
        }
        
        parsed_doc.metadata.structure_info = structure_info
        
        # Create detailed structured content
        parsed_doc.structured_content = {
            'structure': structure_info,
            'readability_score': self._calculate_readability_score(content),
            'key_terms': self._extract_key_terms(content),
            'document_quality': self._assess_document_quality(content)
        }
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headings from content."""
        import re
        
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Markdown headings
            if re.match(r'^#{1,6}\s+(.+)', line):
                heading = re.sub(r'^#{1,6}\s+', '', line).strip()
                sections.append(heading)
            
            # Alternative heading styles (underlined with = or -)
            elif line and len(line) > 0:
                next_line_idx = i + 1
                if (next_line_idx < len(lines) and 
                    lines[next_line_idx].strip() and
                    all(c in '=-' for c in lines[next_line_idx].strip())):
                    sections.append(line)
        
        return sections
    
    def _extract_section_content(self, content: str, section: str) -> str:
        """Extract content for a specific section."""
        lines = content.split('\n')
        section_content = []
        in_section = False
        
        for i, line in enumerate(lines):
            if section.lower() in line.lower() and ('#' in line or 
                (i + 1 < len(lines) and lines[i + 1].strip() and 
                 all(c in '=-' for c in lines[i + 1].strip()))):
                in_section = True
                continue
            
            if in_section:
                # Stop at next section
                if (line.strip().startswith('#') or 
                    (i + 1 < len(lines) and lines[i + 1].strip() and
                     all(c in '=-' for c in lines[i + 1].strip()))):
                    break
                section_content.append(line)
        
        return '\n'.join(section_content).strip()
    
    def _create_smart_chunks(self, content: str, chunk_size: int) -> List[str]:
        """Create intelligent chunks that respect document structure."""
        chunks = []
        current_chunk = ""
        lines = content.split('\n')
        
        for line in lines:
            # If adding this line would exceed chunk size and we have content
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            current_chunk += line + '\n'
            
            # Break at section boundaries for better semantic chunking
            if line.strip().startswith('#') and len(current_chunk) > chunk_size * 0.5:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_automation_chunks(self, content: str) -> List[str]:
        """Create chunks optimized for automation context."""
        chunks = []
        
        # Split by test cases if they exist
        if 'test case' in content.lower():
            import re
            test_case_pattern = r'###?\s*Test Case.*?\n(.*?)(?=###?\s*Test Case|\Z)'
            matches = re.findall(test_case_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if matches:
                chunks.extend([match.strip() for match in matches if match.strip()])
            else:
                # Fallback to section-based chunking
                chunks = self._create_smart_chunks(content, 2000)
        else:
            # For non-test case documents, create semantic chunks
            chunks = self._create_smart_chunks(content, 1500)
        
        return chunks
    
    def _extract_automation_context(self, content: str) -> Dict[str, Any]:
        """Extract context useful for automation generation."""
        context = {
            'requirements': [],
            'user_stories': [],
            'acceptance_criteria': [],
            'api_endpoints': [],
            'business_rules': [],
            'workflows': []
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Identify sections
            if 'user story' in line_lower or 'as a' in line_lower:
                current_section = 'user_stories'
            elif 'acceptance criteria' in line_lower:
                current_section = 'acceptance_criteria'
            elif 'api' in line_lower and ('endpoint' in line_lower or 'method' in line_lower):
                current_section = 'api_endpoints'
            elif 'business rule' in line_lower or 'rule:' in line_lower:
                current_section = 'business_rules'
            elif 'workflow' in line_lower or 'process' in line_lower:
                current_section = 'workflows'
            elif 'requirement' in line_lower:
                current_section = 'requirements'
            
            # Collect content for current section
            if current_section and line.strip():
                context[current_section].append(line.strip())
        
        return context
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """Extract all headings with their levels."""
        import re
        
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Markdown style headings
            match = re.match(r'^(#{1,6})\s+(.+)', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append({
                    'level': level,
                    'text': text,
                    'line_number': i + 1
                })
        
        return headings
    
    def _extract_lists(self, content: str) -> List[str]:
        """Extract list items from content."""
        import re
        
        list_items = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Markdown list items
            if re.match(r'^[-*+]\s+(.+)', line) or re.match(r'^\d+\.\s+(.+)', line):
                list_items.append(line)
        
        return list_items
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content."""
        import re
        
        # Extract fenced code blocks
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
        
        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', content)
        
        return code_blocks + inline_code
    
    def _extract_tables(self, content: str) -> List[str]:
        """Extract table structures from content."""
        import re
        
        tables = []
        lines = content.split('\n')
        
        in_table = False
        current_table = []
        
        for line in lines:
            # Markdown table detection
            if '|' in line and line.strip():
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line.strip())
            else:
                if in_table:
                    tables.append('\n'.join(current_table))
                    in_table = False
        
        if in_table and current_table:
            tables.append('\n'.join(current_table))
        
        return tables
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate document complexity score."""
        # Simple complexity metrics
        lines = len(content.split('\n'))
        words = len(content.split())
        unique_words = len(set(content.lower().split()))
        
        # Vocabulary diversity
        diversity = unique_words / words if words > 0 else 0
        
        # Structure complexity (headings, lists, code blocks)
        headings = len(self._extract_headings(content))
        lists = len(self._extract_lists(content))
        code_blocks = len(self._extract_code_blocks(content))
        
        structure_score = (headings + lists + code_blocks) / lines if lines > 0 else 0
        
        # Combine metrics (0-1 scale)
        complexity = min(1.0, (diversity + structure_score) / 2)
        
        return round(complexity, 3)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate basic readability score."""
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Simple readability metric (lower is better)
        # Scale to 0-1 where 1 is most readable
        readability = max(0.0, min(1.0, 1 - (avg_sentence_length - 15) / 20))
        
        return round(readability, 3)
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content."""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Remove common words
        stop_words = {'that', 'this', 'with', 'have', 'will', 'from', 'they', 
                     'been', 'were', 'said', 'each', 'which', 'their', 'would'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get most common terms
        word_counts = Counter(filtered_words)
        key_terms = [word for word, count in word_counts.most_common(20)]
        
        return key_terms
    
    def _assess_document_quality(self, content: str) -> Dict[str, Any]:
        """Assess overall document quality."""
        return {
            'has_structure': len(self._extract_headings(content)) > 0,
            'has_examples': 'example' in content.lower() or '```' in content,
            'has_lists': len(self._extract_lists(content)) > 0,
            'length_appropriate': 100 < len(content.split()) < 10000,
            'quality_score': self._calculate_complexity_score(content)
        }
    
    def _create_summary(self, content: str, max_length: int = 200) -> str:
        """Create a brief summary of the document."""
        # Take first paragraph or first few sentences
        paragraphs = content.split('\n\n')
        first_paragraph = paragraphs[0] if paragraphs else content[:max_length]
        
        # Truncate to max_length
        if len(first_paragraph) > max_length:
            sentences = first_paragraph.split('.')
            summary = ""
            for sentence in sentences:
                if len(summary + sentence) < max_length:
                    summary += sentence + '.'
                else:
                    break
            return summary.strip()
        
        return first_paragraph.strip()


# Convenience functions for backward compatibility and ease of use

async def parse_document_for_rag(file_path: Union[str, Path], chunk_size: int = 1000) -> ParsedDocument:
    """Parse document for RAG ingestion."""
    parser = UnifiedDocumentParser()
    return await parser.parse_for_rag(file_path, chunk_size)


async def parse_document_for_automation(file_path: Union[str, Path]) -> ParsedDocument:
    """Parse document for test automation."""
    parser = UnifiedDocumentParser()
    return await parser.parse_for_automation(file_path)


async def extract_test_cases_from_document(file_path: Union[str, Path]) -> List[TestCase]:
    """Extract test cases from any supported document format."""
    parser = UnifiedDocumentParser()
    parsed_doc = await parser.parse_for_automation(file_path)
    return parsed_doc.test_cases


async def get_document_metadata(file_path: Union[str, Path]) -> DocumentMetadata:
    """Get metadata from any supported document format."""
    parser = UnifiedDocumentParser()
    return await parser.extract_metadata(file_path)