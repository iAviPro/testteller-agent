"""
Integration tests for multi-format document workflow

Tests the end-to-end workflow of processing different document formats
through both RAG ingestion and TestWriter automation pipelines.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from testteller.core.data_ingestion.unified_document_parser import (
    UnifiedDocumentParser, ParseMode, DocumentType
)


class TestMultiFormatWorkflow:
    """Integration tests for multi-format document processing"""

    @pytest.fixture
    def parser(self):
        """Create UnifiedDocumentParser instance"""
        return UnifiedDocumentParser()

    @pytest.fixture
    def sample_test_cases_md(self):
        """Sample markdown test cases file"""
        return """# Login Feature Test Cases

## Test Case [TC001]: Valid Login
**Test ID**: TC001
**Description**: Verify successful login with valid credentials
**Priority**: High
**Preconditions**: 
- User account exists in the system
- User is on the login page

**Test Steps**:
1. Enter valid username in the username field
2. Enter valid password in the password field
3. Click the "Login" button

**Expected Results**:
- User is redirected to the dashboard
- Welcome message is displayed
- User session is established

---

## Test Case [TC002]: Invalid Password
**Test ID**: TC002
**Description**: Verify error handling for invalid password
**Priority**: Medium
**Preconditions**: 
- User account exists in the system
- User is on the login page

**Test Steps**:
1. Enter valid username in the username field
2. Enter invalid password in the password field
3. Click the "Login" button

**Expected Results**:
- Error message "Invalid credentials" is displayed
- User remains on the login page
- Login attempt is logged for security
"""

    @pytest.fixture
    def sample_requirements_docx_content(self):
        """Sample DOCX requirements content (simulated as text)"""
        return """User Authentication System Requirements

Document Version: 1.0
Date: 2024-01-15
Author: Product Team

1. FUNCTIONAL REQUIREMENTS

1.1 User Registration
As a new user
I want to create an account
So that I can access the system features

Acceptance Criteria:
- User must provide valid email address
- Password must meet security requirements (8+ chars, special characters)
- Email verification required before account activation
- Duplicate email addresses not allowed

1.2 User Login
As a registered user
I want to log into the system
So that I can access my personal dashboard

Acceptance Criteria:
- User authentication via email and password
- Remember me functionality for 30 days
- Account lockout after 5 failed attempts
- Password reset functionality available

2. NON-FUNCTIONAL REQUIREMENTS

2.1 Security Requirements
- All passwords must be hashed using bcrypt
- HTTPS required for all authentication endpoints
- Session timeout after 24 hours of inactivity
- Multi-factor authentication support

2.2 Performance Requirements
- Login process must complete within 2 seconds
- System must support 1000 concurrent users
- 99.9% uptime requirement

3. BUSINESS RULES

3.1 Password Policy
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter  
- At least one number
- At least one special character

3.2 Account Management
- Accounts automatically disabled after 90 days of inactivity
- Users can update profile information
- Account deletion requires email confirmation
"""

    @pytest.fixture
    def sample_api_spec_content(self):
        """Sample API specification content"""
        return """# Authentication API Specification

## Base URL
https://api.example.com/v1

## Authentication Endpoints

### POST /auth/login
Authenticate user credentials and return access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123",
  "remember_me": false
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "12345",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "expires_in": 3600
}
```

**Error Responses:**
- 401 Unauthorized: Invalid credentials
- 422 Unprocessable Entity: Validation errors
- 429 Too Many Requests: Rate limit exceeded

### POST /auth/logout
Invalidate the current session token.

**Request Headers:**
```
Authorization: Bearer <access_token>
```

**Response (204 No Content)**

### POST /auth/refresh
Refresh an expired access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

## Rate Limiting
- 5 requests per minute for login endpoint
- 100 requests per hour for authenticated endpoints
- Rate limit headers included in all responses
"""

    @pytest.fixture
    def temp_files(self, sample_test_cases_md, sample_requirements_docx_content, sample_api_spec_content):
        """Create temporary files for testing"""
        files = {}
        
        # Create markdown test cases file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_test_cases_md)
            files['test_cases_md'] = f.name
        
        # Create text requirements file (simulating DOCX content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(sample_requirements_docx_content)
            files['requirements_txt'] = f.name
        
        # Create API specification file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_api_spec_content)
            files['api_spec_md'] = f.name
        
        yield files
        
        # Cleanup
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_rag_ingestion_workflow(self, parser, temp_files):
        """Test complete RAG ingestion workflow for multiple formats"""
        results = {}
        
        # Process each file for RAG ingestion
        for file_type, file_path in temp_files.items():
            result = await parser.parse_for_rag(file_path, chunk_size=800)
            results[file_type] = result
        
        # Verify all files were processed
        assert len(results) == 3
        
        # Test cases markdown file
        test_cases_result = results['test_cases_md']
        assert test_cases_result.metadata.document_type == DocumentType.TEST_CASES
        assert test_cases_result.metadata.test_case_count == 0  # Only counts when parsing for automation
        assert len(test_cases_result.chunks) > 0
        assert 'title' in test_cases_result.structured_content
        assert 'sections' in test_cases_result.structured_content
        
        # Requirements text file
        requirements_result = results['requirements_txt']
        assert requirements_result.metadata.document_type == DocumentType.REQUIREMENTS
        assert len(requirements_result.chunks) > 0
        assert requirements_result.structured_content['document_type'] == 'requirements'
        
        # API specification file
        api_result = results['api_spec_md']
        assert api_result.metadata.document_type == DocumentType.API_DOCS
        assert len(api_result.chunks) > 0

    @pytest.mark.asyncio
    async def test_automation_workflow(self, parser, temp_files):
        """Test complete automation workflow for multiple formats"""
        results = {}
        
        # Process each file for automation
        for file_type, file_path in temp_files.items():
            result = await parser.parse_for_automation(file_path)
            results[file_type] = result
        
        # Test markdown file with test cases
        test_cases_result = results['test_cases_md']
        assert test_cases_result.metadata.document_type == DocumentType.TEST_CASES
        # Note: test_cases will be empty unless markdown_parser is properly mocked
        assert isinstance(test_cases_result.test_cases, list)
        assert len(test_cases_result.chunks) > 0
        
        # Requirements file - should extract automation context
        requirements_result = results['requirements_txt']
        assert requirements_result.metadata.document_type == DocumentType.REQUIREMENTS
        assert requirements_result.structured_content is not None
        assert 'user_stories' in requirements_result.structured_content
        assert 'requirements' in requirements_result.structured_content
        
        # API specification - should be processed for automation context
        api_result = results['api_spec_md']
        assert api_result.metadata.document_type == DocumentType.API_DOCS
        assert len(api_result.chunks) > 0

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, parser, temp_files):
        """Test batch processing of multiple document formats"""
        file_paths = list(temp_files.values())
        
        # Test batch processing for RAG
        rag_results = await parser.batch_parse(file_paths, ParseMode.RAG_INGESTION, max_concurrency=3)
        assert len(rag_results) == 3
        
        # Verify all documents have chunks for RAG
        for result in rag_results:
            assert len(result.chunks) > 0
            assert result.structured_content is not None
        
        # Test batch processing for automation
        automation_results = await parser.batch_parse(file_paths, ParseMode.AUTOMATION, max_concurrency=3)
        assert len(automation_results) == 3
        
        # Verify automation-specific processing
        for result in automation_results:
            assert len(result.chunks) > 0
            if result.metadata.document_type == DocumentType.TEST_CASES:
                assert isinstance(result.test_cases, list)
            elif result.metadata.document_type in [DocumentType.REQUIREMENTS, DocumentType.API_DOCS]:
                assert result.structured_content is not None

    @pytest.mark.asyncio
    async def test_document_type_detection_accuracy(self, parser, temp_files):
        """Test accuracy of document type detection across formats"""
        expected_types = {
            'test_cases_md': DocumentType.TEST_CASES,
            'requirements_txt': DocumentType.REQUIREMENTS,
            'api_spec_md': DocumentType.API_DOCS
        }
        
        for file_type, file_path in temp_files.items():
            metadata = await parser.extract_metadata(file_path)
            expected_type = expected_types[file_type]
            assert metadata.document_type == expected_type, f"Failed for {file_type}: expected {expected_type}, got {metadata.document_type}"

    @pytest.mark.asyncio
    async def test_metadata_consistency_across_modes(self, parser, temp_files):
        """Test that metadata is consistent across different parsing modes"""
        for file_type, file_path in temp_files.items():
            # Parse with different modes
            rag_result = await parser.parse_document(file_path, ParseMode.RAG_INGESTION)
            automation_result = await parser.parse_document(file_path, ParseMode.AUTOMATION)
            analysis_result = await parser.parse_document(file_path, ParseMode.ANALYSIS)
            metadata_only = await parser.extract_metadata(file_path)
            
            # Core metadata should be consistent
            metadatas = [
                rag_result.metadata,
                automation_result.metadata,
                analysis_result.metadata,
                metadata_only
            ]
            
            for metadata in metadatas:
                assert metadata.file_path == file_path
                assert metadata.file_type == Path(file_path).suffix.lower()
                assert metadata.document_type == metadatas[0].document_type
                assert metadata.title == metadatas[0].title

    @pytest.mark.asyncio
    async def test_chunking_strategy_differences(self, parser, temp_files):
        """Test that different parsing modes use appropriate chunking strategies"""
        file_path = temp_files['requirements_txt']
        
        # RAG ingestion with small chunks
        rag_result = await parser.parse_for_rag(file_path, chunk_size=400)
        
        # Automation parsing
        automation_result = await parser.parse_for_automation(file_path)
        
        # RAG should have more, smaller chunks
        assert len(rag_result.chunks) >= len(automation_result.chunks)
        
        # Average RAG chunk size should be smaller
        if rag_result.chunks and automation_result.chunks:
            avg_rag_chunk_size = sum(len(chunk) for chunk in rag_result.chunks) / len(rag_result.chunks)
            avg_automation_chunk_size = sum(len(chunk) for chunk in automation_result.chunks) / len(automation_result.chunks)
            assert avg_rag_chunk_size <= avg_automation_chunk_size * 1.5  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_content_preservation(self, parser, temp_files):
        """Test that original content is preserved across all parsing modes"""
        for file_type, file_path in temp_files.items():
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Parse with different modes
            rag_result = await parser.parse_for_rag(file_path)
            automation_result = await parser.parse_for_automation(file_path)
            
            # Content should be identical to original
            assert rag_result.content == original_content
            assert automation_result.content == original_content
            
            # Chunks should contain all original content when combined
            rag_combined = '\n'.join(rag_result.chunks)
            automation_combined = '\n'.join(automation_result.chunks)
            
            # Allow for minor whitespace differences in chunking
            original_words = set(original_content.split())
            rag_words = set(rag_combined.split())
            automation_words = set(automation_combined.split())
            
            # Most words should be preserved (allowing for some processing differences)
            assert len(original_words & rag_words) >= len(original_words) * 0.9
            assert len(original_words & automation_words) >= len(original_words) * 0.9

    @pytest.mark.asyncio
    async def test_error_handling_mixed_formats(self, parser):
        """Test error handling when processing mixed valid and invalid files"""
        # Create mix of valid and invalid files
        valid_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
        valid_file.write("# Valid Document\nContent here.")
        valid_file.close()
        
        invalid_file = "/non/existent/file.txt"
        
        files = [valid_file.name, invalid_file]
        
        try:
            # Batch process should handle errors gracefully
            results = await parser.batch_parse(files, ParseMode.RAG_INGESTION)
            
            # Should return only successful parses
            assert len(results) == 1
            assert results[0].metadata.file_path == valid_file.name
            
        finally:
            os.unlink(valid_file.name)

    @pytest.mark.asyncio
    async def test_performance_with_large_batch(self, parser, temp_files):
        """Test performance characteristics with larger batches"""
        # Create multiple copies of files for batch processing
        file_paths = []
        for i in range(5):  # Create 5 copies of each file
            for original_path in temp_files.values():
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                    with open(original_path, 'r', encoding='utf-8') as orig:
                        f.write(orig.read())
                    file_paths.append(f.name)
        
        try:
            import time
            start_time = time.time()
            
            # Process batch with limited concurrency
            results = await parser.batch_parse(file_paths, ParseMode.RAG_INGESTION, max_concurrency=3)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process all files
            assert len(results) == len(file_paths)
            
            # Should complete within reasonable time (adjust based on system)
            assert processing_time < 30  # 30 seconds max for 15 files
            
            # Verify all results are valid
            for result in results:
                assert result.content is not None
                assert len(result.chunks) > 0
                
        finally:
            # Cleanup temporary files
            for file_path in file_paths:
                try:
                    os.unlink(file_path)
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])