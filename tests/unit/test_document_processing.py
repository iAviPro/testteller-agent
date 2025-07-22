"""
Unit tests for document processing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from testteller.core.data_ingestion.document_loader import DocumentLoader
from testteller.core.data_ingestion.unified_document_parser import UnifiedDocumentParser, DocumentType
from testteller.core.data_ingestion.text_splitter import TextSplitter


class TestDocumentLoader:
    """Test DocumentLoader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
    
    def test_init(self):
        """Test DocumentLoader initialization."""
        assert self.loader is not None
    
    def test_load_text_file(self):
        """Test loading a simple text file."""
        content = "This is a test document.\nIt has multiple lines.\nAnd some content."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            result = self.loader.load_text(temp_file)
            assert content in result
        finally:
            os.unlink(temp_file)
    
    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        content = """# Test Document
        
## Section 1
This is some content.

## Section 2
This is more content.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            result = self.loader.load_text(temp_file)
            assert "Test Document" in result
            assert "Section 1" in result
            assert "Section 2" in result
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_text("nonexistent_file.txt")


class TestTextSplitter:
    """Test TextSplitter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.splitter = TextSplitter()
    
    def test_init_default(self):
        """Test TextSplitter initialization with defaults."""
        assert self.splitter.chunk_size == 1000
        assert self.splitter.chunk_overlap == 200
    
    def test_init_custom(self):
        """Test TextSplitter initialization with custom values."""
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
    
    def test_split_simple_text(self):
        """Test splitting simple text."""
        text = "This is a simple text. " * 100  # Create text longer than default chunk size
        chunks = self.splitter.split_text(text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(len(chunk) <= self.splitter.chunk_size + 100 for chunk in chunks)  # Allow some flexibility
    
    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        text = "This is a short text."
        chunks = self.splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_with_custom_separators(self):
        """Test splitting with custom separators."""
        text = "Section 1\n\nSection 2\n\nSection 3\n\nSection 4"
        chunks = self.splitter.split_text(text, separators=["\n\n"])
        
        assert len(chunks) >= 1  # Should respect paragraph boundaries


class TestUnifiedDocumentParser:
    """Test UnifiedDocumentParser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = UnifiedDocumentParser()
    
    def test_init(self):
        """Test UnifiedDocumentParser initialization."""
        assert self.parser is not None
        assert hasattr(self.parser, 'document_loader')
        assert hasattr(self.parser, 'markdown_parser')
    
    def test_detect_document_type_test_cases(self):
        """Test document type detection for test cases."""
        content = """
### Test Case E2E_[1]
**Feature:** User Login
**Type:** Authentication

#### Objective
Test user login functionality.

#### Test Steps
1. Enter credentials
2. Click login
3. Verify success
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type == DocumentType.TEST_CASES
    
    def test_detect_document_type_documentation(self):
        """Test document type detection for documentation."""
        content = """
# API Documentation

## Overview
This document describes the API endpoints.

## Authentication
Use bearer tokens for authentication.

## Endpoints
- GET /users
- POST /users
- PUT /users/{id}
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type in [DocumentType.DOCUMENTATION, DocumentType.API_DOCS]
    
    def test_detect_document_type_requirements(self):
        """Test document type detection for requirements."""
        content = """
# Requirements Document

## Functional Requirements
1. The system must support user registration
2. The system must support user login
3. The system must support password reset

## Non-Functional Requirements
1. Response time must be under 2 seconds
2. System must support 1000 concurrent users
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type == DocumentType.REQUIREMENTS
    
    @pytest.mark.asyncio
    async def test_extract_metadata_basic(self):
        """Test basic metadata extraction."""
        content = """# Test Document

This is a test document with some content.

## Section 1
Content here.

## Section 2
More content here.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            metadata = await self.parser.extract_metadata(temp_file)
            
            assert metadata is not None
            assert metadata.file_type == '.md'
            assert metadata.title is not None
            assert metadata.word_count > 0
            assert metadata.character_count > 0
            assert len(metadata.sections) > 0
        finally:
            os.unlink(temp_file)
    
    def test_smart_chunking_preserves_structure(self):
        """Test that smart chunking preserves document structure."""
        content = """# Main Title

## Section 1
This is content for section 1.
It has multiple paragraphs.

### Subsection 1.1
More detailed content here.

## Section 2
This is content for section 2.

### Subsection 2.1
Even more content.
"""
        
        chunks = self.parser._smart_chunk_text(content, chunk_size=200)
        
        assert len(chunks) > 1
        # Verify that section headers are preserved
        section_chunks = [chunk for chunk in chunks if chunk.strip().startswith('#')]
        assert len(section_chunks) > 0
    
    def test_extract_sections(self):
        """Test section extraction from markdown."""
        content = """# Main Document

## Introduction
This is the introduction.

## Methods
This describes the methods.

### Method 1
Details about method 1.

### Method 2
Details about method 2.

## Conclusion
This is the conclusion.
"""
        
        sections = self.parser._extract_sections(content)
        
        assert len(sections) > 0
        assert any('Introduction' in section for section in sections)
        assert any('Methods' in section for section in sections)
        assert any('Conclusion' in section for section in sections)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "  This   has    extra    spaces   and\n\n\nmultiple\n\n\nnewlines.  "
        cleaned = self.parser._clean_text(text)
        
        assert "extra spaces" not in cleaned  # Multiple spaces should be reduced
        assert cleaned.strip() == cleaned  # Leading/trailing whitespace removed
        assert len(cleaned) < len(text)  # Should be shorter after cleaning


class TestDocumentTypeDetection:
    """Test document type detection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = UnifiedDocumentParser()
    
    def test_api_documentation_detection(self):
        """Test detection of API documentation."""
        content = """# API Reference

## Endpoints

### GET /api/users
Returns a list of users.

### POST /api/users
Creates a new user.

## Authentication
Use API keys for authentication.
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type == DocumentType.API_DOCS
    
    def test_test_cases_with_various_formats(self):
        """Test detection of test cases in various formats."""
        test_contents = [
            """### Test Case E2E_[1]
**Feature:** Login
**Steps:** Enter credentials, click login""",
            
            """## Test Scenario: User Registration
1. Navigate to signup page
2. Fill in form
3. Submit
Expected: User is registered""",
            
            """# Test Plan
Test ID: TC001
Test Description: Verify login functionality
Steps:
1. Open app
2. Enter username
3. Enter password
4. Click login
Expected Result: User logged in successfully"""
        ]
        
        for content in test_contents:
            doc_type = self.parser._detect_document_type(content)
            assert doc_type == DocumentType.TEST_CASES
    
    def test_specifications_detection(self):
        """Test detection of specification documents."""
        content = """# System Specification

## Architecture Overview
The system follows a microservices architecture.

## Database Design
The database consists of the following tables:
- users
- orders
- products

## API Specification
REST APIs are used for communication.
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type == DocumentType.SPECIFICATIONS
    
    def test_fallback_to_documentation(self):
        """Test fallback to generic documentation."""
        content = """# Some Document

This is just a regular document without specific indicators.
It has some content but no clear type indicators.

## Random Section
With some random content.
"""
        doc_type = self.parser._detect_document_type(content)
        assert doc_type == DocumentType.DOCUMENTATION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])