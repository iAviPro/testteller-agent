"""
Unit tests for UnifiedDocumentParser

Tests the unified document parsing functionality for multiple formats
including metadata extraction, document type detection, and parsing modes.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from testteller.core.data_ingestion.unified_document_parser import (
    UnifiedDocumentParser, ParseMode, DocumentType, DocumentMetadata, ParsedDocument,
    parse_document_for_rag, parse_document_for_automation, extract_test_cases_from_document
)
from testteller.automator_agent.parser.markdown_parser import TestCase


class TestUnifiedDocumentParser:
    """Test suite for UnifiedDocumentParser"""

    @pytest.fixture
    def parser(self):
        """Create a UnifiedDocumentParser instance"""
        return UnifiedDocumentParser()

    @pytest.fixture
    def sample_md_content(self):
        """Sample markdown content with test cases"""
        return """# Test Cases for Login Feature

## Overview
This document contains test cases for the user login functionality.

### Test Case [1]: Valid Login
**Description**: Test successful login with valid credentials
**Preconditions**: User account exists
**Test Steps**:
1. Navigate to login page
2. Enter valid username and password
3. Click login button
**Expected Result**: User is redirected to dashboard

### Test Case [2]: Invalid Password
**Description**: Test login with invalid password
**Test Steps**:
1. Navigate to login page
2. Enter valid username and invalid password
3. Click login button
**Expected Result**: Error message is displayed
"""

    @pytest.fixture
    def sample_requirements_content(self):
        """Sample requirements document content"""
        return """# User Authentication Requirements

## Functional Requirements

### User Story: Login
As a registered user
I want to log into the system
So that I can access my personal dashboard

### Acceptance Criteria:
- User must provide valid credentials
- System validates credentials against database
- Successful login redirects to dashboard
- Failed login shows appropriate error message

## Business Rules
- Password must be at least 8 characters
- Account locks after 5 failed attempts
"""

    @pytest.fixture
    def temp_md_file(self, sample_md_content):
        """Create temporary markdown file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_md_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_txt_file(self, sample_requirements_content):
        """Create temporary text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_requirements_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parse_document_basic(self, parser, temp_md_file):
        """Test basic document parsing"""
        result = await parser.parse_document(temp_md_file, ParseMode.RAG_INGESTION)
        
        assert isinstance(result, ParsedDocument)
        assert result.metadata.file_path == temp_md_file
        assert result.metadata.file_type == '.md'
        assert result.content is not None
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_document_type_detection(self, parser, temp_md_file, temp_txt_file):
        """Test document type detection"""
        # Test test cases document
        md_result = await parser.parse_document(temp_md_file, ParseMode.METADATA_ONLY)
        assert md_result.metadata.document_type == DocumentType.TEST_CASES
        
        # Test requirements document
        txt_result = await parser.parse_document(temp_txt_file, ParseMode.METADATA_ONLY)
        assert txt_result.metadata.document_type == DocumentType.REQUIREMENTS

    @pytest.mark.asyncio
    async def test_parse_for_rag_mode(self, parser, temp_md_file):
        """Test RAG ingestion parsing mode"""
        result = await parser.parse_for_rag(temp_md_file, chunk_size=500)
        
        assert result.metadata.document_type == DocumentType.TEST_CASES
        assert len(result.chunks) > 0
        assert result.structured_content is not None
        assert 'title' in result.structured_content
        assert 'sections' in result.structured_content

    @pytest.mark.asyncio
    async def test_parse_for_automation_mode(self, parser, temp_md_file):
        """Test automation parsing mode"""
        with patch.object(parser.markdown_parser, 'parse_content') as mock_parse:
            mock_test_cases = [
                Mock(id="1", description="Test case 1"),
                Mock(id="2", description="Test case 2")
            ]
            mock_parse.return_value = mock_test_cases
            
            result = await parser.parse_for_automation(temp_md_file)
            
            assert result.metadata.document_type == DocumentType.TEST_CASES
            assert len(result.test_cases) == 2
            assert result.metadata.test_case_count == 2

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, parser, temp_md_file):
        """Test metadata extraction functionality"""
        metadata = await parser.extract_metadata(temp_md_file)
        
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.file_path == temp_md_file
        assert metadata.file_type == '.md'
        assert metadata.document_type == DocumentType.TEST_CASES
        assert metadata.title is not None

    @pytest.mark.asyncio
    async def test_batch_parsing(self, parser, temp_md_file, temp_txt_file):
        """Test batch parsing of multiple documents"""
        files = [temp_md_file, temp_txt_file]
        results = await parser.batch_parse(files, ParseMode.METADATA_ONLY, max_concurrency=2)
        
        assert len(results) == 2
        assert all(isinstance(result, ParsedDocument) for result in results)
        
        # Check that different document types were detected
        doc_types = [result.metadata.document_type for result in results]
        assert DocumentType.TEST_CASES in doc_types
        assert DocumentType.REQUIREMENTS in doc_types

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, parser):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            await parser.parse_document("/non/existent/file.md")

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, parser):
        """Test handling of empty documents"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")  # Empty file
            empty_file = f.name
        
        try:
            with patch.object(parser.document_loader, 'load_document', return_value=""):
                with pytest.raises(ValueError, match="Failed to load content"):
                    await parser.parse_document(empty_file)
        finally:
            os.unlink(empty_file)

    def test_section_extraction(self, parser):
        """Test section heading extraction"""
        content = """# Main Title

## Section 1
Content for section 1

### Subsection 1.1
More content

## Section 2
Content for section 2
"""
        sections = parser._extract_sections(content)
        expected_sections = ["Main Title", "Section 1", "Subsection 1.1", "Section 2"]
        assert sections == expected_sections

    def test_smart_chunking(self, parser):
        """Test intelligent chunking that respects document structure"""
        content = "# Title\n" + "This is content. " * 50 + "\n## Section\n" + "More content. " * 30
        chunks = parser._create_smart_chunks(content, chunk_size=200)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 250 for chunk in chunks)  # Allow some flexibility
        
        # Check that sections are preserved
        section_chunk = next((chunk for chunk in chunks if "## Section" in chunk), None)
        assert section_chunk is not None

    def test_automation_context_extraction(self, parser, sample_requirements_content):
        """Test extraction of automation-relevant context"""
        context = parser._extract_automation_context(sample_requirements_content)
        
        assert isinstance(context, dict)
        assert 'user_stories' in context
        assert 'requirements' in context
        assert 'acceptance_criteria' in context
        
        # Check that content was properly categorized
        assert len(context['user_stories']) > 0
        assert len(context['requirements']) > 0

    def test_document_quality_assessment(self, parser, sample_md_content):
        """Test document quality assessment"""
        quality = parser._assess_document_quality(sample_md_content)
        
        assert isinstance(quality, dict)
        assert 'has_structure' in quality
        assert 'has_examples' in quality
        assert 'has_lists' in quality
        assert 'length_appropriate' in quality
        assert 'quality_score' in quality
        
        assert quality['has_structure'] is True  # Has headings
        assert isinstance(quality['quality_score'], float)

    @pytest.mark.asyncio
    async def test_convenience_functions(self, temp_md_file):
        """Test convenience functions for backward compatibility"""
        # Test parse_document_for_rag
        with patch('testteller.data_ingestion.unified_document_parser.UnifiedDocumentParser') as mock_parser:
            mock_instance = AsyncMock()
            mock_parser.return_value = mock_instance
            mock_result = Mock()
            mock_instance.parse_for_rag.return_value = mock_result
            
            result = await parse_document_for_rag(temp_md_file, chunk_size=500)
            
            mock_instance.parse_for_rag.assert_called_once_with(temp_md_file, 500)
            assert result == mock_result

        # Test parse_document_for_automation
        with patch('testteller.data_ingestion.unified_document_parser.UnifiedDocumentParser') as mock_parser:
            mock_instance = AsyncMock()
            mock_parser.return_value = mock_instance
            mock_result = Mock()
            mock_instance.parse_for_automation.return_value = mock_result
            
            result = await parse_document_for_automation(temp_md_file)
            
            mock_instance.parse_for_automation.assert_called_once_with(temp_md_file)
            assert result == mock_result

    def test_complexity_score_calculation(self, parser):
        """Test document complexity score calculation"""
        simple_content = "This is a simple document with just text."
        complex_content = """# Complex Document

## Section 1
- List item 1
- List item 2

```python
def function():
    pass
```

### Subsection
More text with **formatting** and `code`.

## Section 2
| Table | Header |
|-------|--------|
| Data  | Value  |
"""
        
        simple_score = parser._calculate_complexity_score(simple_content)
        complex_score = parser._calculate_complexity_score(complex_content)
        
        assert isinstance(simple_score, float)
        assert isinstance(complex_score, float)
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
        assert complex_score > simple_score

    @pytest.mark.asyncio
    async def test_error_handling_in_batch_parse(self, parser, temp_md_file):
        """Test error handling in batch parsing"""
        files = [temp_md_file, "/non/existent/file.md"]
        
        results = await parser.batch_parse(files, ParseMode.METADATA_ONLY)
        
        # Should return only successful parses
        assert len(results) == 1
        assert results[0].metadata.file_path == temp_md_file


@pytest.mark.asyncio
async def test_extract_test_cases_from_document_integration():
    """Integration test for extracting test cases from document"""
    content = """### Test Case [1]: Login Test
**Description**: Test login functionality
**Test Steps**:
1. Open login page
2. Enter credentials
3. Click login
**Expected Result**: User logged in
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name
    
    try:
        with patch('testteller.data_ingestion.unified_document_parser.UnifiedDocumentParser') as mock_parser:
            mock_instance = AsyncMock()
            mock_parser.return_value = mock_instance
            mock_parsed_doc = Mock()
            mock_test_cases = [Mock(id="1", description="Login Test")]
            mock_parsed_doc.test_cases = mock_test_cases
            mock_instance.parse_for_automation.return_value = mock_parsed_doc
            
            result = await extract_test_cases_from_document(temp_path)
            
            mock_instance.parse_for_automation.assert_called_once_with(temp_path)
            assert result == mock_test_cases
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])