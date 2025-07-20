"""
Integration tests for enhanced RAG ingestion capabilities

Tests the enhanced RAG pipeline including smart chunking, metadata extraction,
and improved retrieval accuracy with the unified document parser.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from testteller.generator_agent.agent.testteller_agent import TestTellerRagAgent
from testteller.core.data_ingestion.unified_document_parser import UnifiedDocumentParser


class TestEnhancedRAGIngestion:
    """Integration tests for enhanced RAG ingestion pipeline"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        settings = Mock()
        settings.llm.provider = "gemini"
        settings.llm.gemini_api_key = "test-key"
        settings.llm.gemini_generation_model = "gemini-pro"
        settings.llm.gemini_embedding_model = "embedding-001"
        settings.chromadb.host = "localhost"
        settings.chromadb.port = 8000
        settings.chromadb.use_remote = False
        settings.chromadb.persist_directory = "./test_chroma_data"
        settings.chromadb.default_collection_name = "test_collection"
        return settings

    @pytest.fixture
    def sample_complex_document(self):
        """Complex document with multiple sections and formats"""
        return """# E-Commerce Platform Test Strategy

## Overview
This document outlines the comprehensive testing strategy for our e-commerce platform.

## Test Scope

### Frontend Testing
- User interface components
- User experience flows
- Cross-browser compatibility
- Mobile responsiveness

### Backend Testing
- API endpoints validation
- Database operations
- Authentication and authorization
- Payment processing

### Integration Testing
- Third-party service integrations
- Microservices communication
- End-to-end user workflows

## Test Cases

### User Registration Tests

#### Test Case [REG001]: Valid Registration
**Priority**: High
**Description**: Verify successful user registration with valid data
**Preconditions**: 
- Registration page is accessible
- Database is available

**Test Steps**:
1. Navigate to registration page
2. Fill in valid user details:
   - Email: test@example.com
   - Password: SecurePass123!
   - Confirm Password: SecurePass123!
   - First Name: John
   - Last Name: Doe
3. Accept terms and conditions
4. Click "Register" button

**Expected Results**:
- User account is created successfully
- Confirmation email is sent
- User is redirected to login page
- Success message is displayed

#### Test Case [REG002]: Duplicate Email
**Priority**: Medium
**Description**: Verify error handling for duplicate email registration
**Test Steps**:
1. Attempt to register with existing email
2. Fill in all other valid details
3. Submit registration form

**Expected Results**:
- Error message: "Email already exists"
- Registration form remains open
- User account is not created

### Shopping Cart Tests

#### Test Case [CART001]: Add Product to Cart
**Priority**: High
**Description**: Verify adding products to shopping cart
**Test Steps**:
1. Browse to product catalog
2. Select a product
3. Choose quantity and options
4. Click "Add to Cart"

**Expected Results**:
- Product appears in cart
- Cart total updates correctly
- Cart icon shows item count

## API Test Scenarios

### Authentication API

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

**Expected Response (200 OK):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "12345",
    "email": "user@example.com",
    "role": "customer"
  }
}
```

### Product API

```http
GET /api/v1/products?category=electronics&limit=10
Authorization: Bearer <token>
```

## Performance Requirements

| Metric | Target | Critical |
|--------|---------|----------|
| Page Load Time | < 2 seconds | < 5 seconds |
| API Response Time | < 500ms | < 1 second |
| Concurrent Users | 1000 | 5000 |
| Database Queries | < 100ms | < 500ms |

## Security Test Cases

### Authentication Security
- SQL injection attempts
- XSS vulnerability tests
- CSRF protection validation
- Session management security

### Data Protection
- PCI DSS compliance for payments
- Personal data encryption
- Secure data transmission (HTTPS)
- Access control validation

## Automation Strategy

### Unit Tests
- 80% code coverage target
- Mock external dependencies
- Fast execution (< 10 minutes)

### Integration Tests
- Database integration
- External API integration
- Service-to-service communication

### End-to-End Tests
- Critical user journeys
- Cross-browser testing
- Mobile device testing

## Tools and Frameworks

### Testing Tools
- **Selenium WebDriver**: Browser automation
- **Pytest**: Python test framework
- **Jest**: JavaScript unit testing
- **Postman**: API testing
- **JMeter**: Performance testing

### CI/CD Integration
- GitHub Actions workflows
- Automated test execution
- Test result reporting
- Quality gate enforcement

## Risk Assessment

### High Risk Areas
1. Payment processing integration
2. User data security
3. Third-party API dependencies
4. Database performance under load

### Mitigation Strategies
- Comprehensive test coverage
- Security penetration testing
- Load testing scenarios
- Disaster recovery procedures

## Test Environment Setup

### Development Environment
- Local database setup
- Mock external services
- Test data generation
- Debug logging enabled

### Staging Environment
- Production-like configuration
- Real external service integration
- Performance monitoring
- Security scanning

### Production Environment
- Monitoring and alerting
- Error tracking
- Performance metrics
- User feedback collection
"""

    @pytest.fixture
    def temp_document_file(self, sample_complex_document):
        """Create temporary document file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_complex_document)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_enhanced_document_ingestion(self, temp_document_file):
        """Test enhanced document ingestion with metadata extraction"""
        parser = UnifiedDocumentParser()
        
        # Test enhanced parsing
        result = await parser.parse_for_rag(temp_document_file, chunk_size=1200)
        
        # Verify enhanced features
        assert result.metadata.document_type is not None
        assert result.metadata.word_count > 0
        assert result.metadata.character_count > 0
        assert len(result.metadata.sections) > 5  # Should detect multiple sections
        
        # Verify structured content
        assert 'title' in result.structured_content
        assert 'sections' in result.structured_content
        assert 'document_type' in result.structured_content
        assert 'summary' in result.structured_content
        
        # Verify smart chunking
        assert len(result.chunks) > 1
        for chunk in result.chunks:
            assert len(chunk) <= 1200 * 1.1  # Allow 10% flexibility
            assert len(chunk) > 100  # Should not be too small

    @pytest.mark.asyncio
    async def test_smart_chunking_preserves_context(self, temp_document_file):
        """Test that smart chunking preserves document context and structure"""
        parser = UnifiedDocumentParser()
        result = await parser.parse_for_rag(temp_document_file, chunk_size=800)
        
        # Test that section boundaries are respected
        chunks_with_headers = [chunk for chunk in result.chunks if chunk.strip().startswith('#')]
        assert len(chunks_with_headers) > 0, "Should preserve section headers in chunks"
        
        # Test that code blocks are preserved
        chunks_with_code = [chunk for chunk in result.chunks if '```' in chunk]
        assert len(chunks_with_code) > 0, "Should preserve code blocks in chunks"
        
        # Test that tables are preserved
        chunks_with_tables = [chunk for chunk in result.chunks if '|' in chunk and '---' in chunk]
        assert len(chunks_with_tables) > 0, "Should preserve table structure in chunks"

    @pytest.mark.asyncio
    async def test_metadata_extraction_accuracy(self, temp_document_file):
        """Test accuracy of metadata extraction"""
        parser = UnifiedDocumentParser()
        metadata = await parser.extract_metadata(temp_document_file)
        
        # Test document type detection
        assert metadata.document_type.value in ['test_cases', 'documentation']
        
        # Test title extraction
        assert metadata.title is not None
        assert len(metadata.title) > 0
        
        # Test section detection
        expected_sections = [
            'E-Commerce Platform Test Strategy',
            'Overview', 
            'Test Scope',
            'Frontend Testing',
            'Backend Testing',
            'Test Cases',
            'API Test Scenarios'
        ]
        
        # Should detect most major sections
        detected_sections = metadata.sections
        matches = sum(1 for expected in expected_sections if any(expected in section for section in detected_sections))
        assert matches >= len(expected_sections) * 0.7  # At least 70% accuracy

    @pytest.mark.asyncio
    async def test_structured_content_quality(self, temp_document_file):
        """Test quality of structured content extraction"""
        parser = UnifiedDocumentParser()
        result = await parser.parse_for_rag(temp_document_file)
        
        structured = result.structured_content
        
        # Test section content extraction
        sections = structured['sections']
        assert len(sections) > 0
        
        # Should extract content for major sections
        for section_name, section_content in sections.items():
            if section_content:  # Some sections might be empty
                assert len(section_content) > 10  # Should have meaningful content
                assert isinstance(section_content, str)
        
        # Test summary quality
        summary = structured['summary']
        assert len(summary) > 50  # Should be substantial
        assert len(summary) <= 300  # Should not be too long
        assert 'e-commerce' in summary.lower() or 'testing' in summary.lower()

    @pytest.mark.asyncio 
    async def test_chunking_strategies_comparison(self, temp_document_file):
        """Test different chunking strategies and their effectiveness"""
        parser = UnifiedDocumentParser()
        
        # Test different chunk sizes
        small_chunks = await parser.parse_for_rag(temp_document_file, chunk_size=500)
        medium_chunks = await parser.parse_for_rag(temp_document_file, chunk_size=1000)
        large_chunks = await parser.parse_for_rag(temp_document_file, chunk_size=2000)
        
        # Verify chunk count relationships
        assert len(small_chunks.chunks) > len(medium_chunks.chunks)
        assert len(medium_chunks.chunks) >= len(large_chunks.chunks)
        
        # Test chunk content overlap
        all_small_content = '\n'.join(small_chunks.chunks)
        all_medium_content = '\n'.join(medium_chunks.chunks) 
        all_large_content = '\n'.join(large_chunks.chunks)
        
        # All should contain the same essential content
        original_words = set(small_chunks.content.split())
        small_words = set(all_small_content.split())
        medium_words = set(all_medium_content.split())
        large_words = set(all_large_content.split())
        
        # All chunking strategies should preserve most content
        assert len(original_words & small_words) >= len(original_words) * 0.9
        assert len(original_words & medium_words) >= len(original_words) * 0.9
        assert len(original_words & large_words) >= len(original_words) * 0.9

    @pytest.mark.asyncio
    async def test_complex_document_structure_analysis(self, temp_document_file):
        """Test analysis of complex document structures"""
        parser = UnifiedDocumentParser()
        result = await parser.parse_document(temp_document_file, parser.ParseMode.ANALYSIS)
        
        structure_info = result.metadata.structure_info
        
        # Test heading extraction
        headings = structure_info['headings']
        assert len(headings) > 5
        
        # Should detect different heading levels
        heading_levels = [h['level'] for h in headings]
        assert min(heading_levels) == 1  # H1
        assert max(heading_levels) >= 3  # At least H3
        
        # Test list detection
        lists = structure_info['lists']
        assert len(lists) > 0  # Document contains lists
        
        # Test code block detection
        code_blocks = structure_info['code_blocks']
        assert len(code_blocks) > 0  # Document contains code
        
        # Test table detection
        tables = structure_info['tables']
        assert len(tables) > 0  # Document contains tables
        
        # Test complexity score
        complexity = structure_info['complexity_score']
        assert 0 <= complexity <= 1
        assert complexity > 0.3  # Should be reasonably complex

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, sample_complex_document):
        """Test performance of batch processing multiple documents"""
        # Create multiple temporary files
        temp_files = []
        variations = [
            sample_complex_document,
            sample_complex_document[:len(sample_complex_document)//2],  # Shorter version
            sample_complex_document.replace('E-Commerce', 'Banking'),  # Modified version
        ]
        
        for i, content in enumerate(variations * 2):  # 6 files total
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_files.append(f.name)
        
        try:
            parser = UnifiedDocumentParser()
            
            import time
            start_time = time.time()
            
            # Batch process with concurrency
            results = await parser.batch_parse(temp_files, parser.ParseMode.RAG_INGESTION, max_concurrency=3)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify results
            assert len(results) == len(temp_files)
            assert processing_time < 15  # Should complete within 15 seconds
            
            # Verify all results have proper structure
            for result in results:
                assert len(result.chunks) > 0
                assert result.structured_content is not None
                assert result.metadata.word_count > 0
                
        finally:
            # Cleanup
            for file_path in temp_files:
                try:
                    os.unlink(file_path)
                except FileNotFoundError:
                    pass

    @pytest.mark.asyncio
    async def test_document_quality_assessment(self, temp_document_file):
        """Test document quality assessment features"""
        parser = UnifiedDocumentParser()
        result = await parser.parse_document(temp_document_file, parser.ParseMode.ANALYSIS)
        
        structured = result.structured_content
        
        # Test readability score
        readability = structured['readability_score']
        assert 0 <= readability <= 1
        
        # Test key terms extraction
        key_terms = structured['key_terms']
        assert len(key_terms) > 0
        expected_terms = ['testing', 'api', 'user', 'test', 'security', 'performance']
        found_terms = sum(1 for term in expected_terms if any(term in kt.lower() for kt in key_terms))
        assert found_terms >= 3  # Should find at least 3 expected terms
        
        # Test document quality assessment
        quality = structured['document_quality']
        assert quality['has_structure'] is True  # Has headings
        assert quality['has_examples'] is True   # Has code examples
        assert quality['has_lists'] is True      # Has lists
        assert quality['length_appropriate'] is True  # Reasonable length
        assert isinstance(quality['quality_score'], float)

    @pytest.mark.asyncio
    async def test_content_preservation_in_processing(self, temp_document_file):
        """Test that content is preserved throughout processing pipeline"""
        parser = UnifiedDocumentParser()
        
        # Read original content
        with open(temp_document_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Process with different modes
        rag_result = await parser.parse_for_rag(temp_document_file)
        automation_result = await parser.parse_for_automation(temp_document_file)
        analysis_result = await parser.parse_for_analysis(temp_document_file)
        
        # All should preserve original content
        assert rag_result.content == original_content
        assert automation_result.content == original_content
        assert analysis_result.content == original_content
        
        # Test that critical sections are preserved in chunks
        critical_content = [
            "Test Case [REG001]",
            "POST /api/v1/auth/login",
            "Performance Requirements",
            "Security Test Cases"
        ]
        
        # All critical content should appear in at least one chunk
        for mode_result in [rag_result, automation_result]:
            all_chunks_text = '\n'.join(mode_result.chunks)
            for critical in critical_content:
                assert critical in all_chunks_text, f"Missing critical content: {critical}"

    @pytest.mark.asyncio
    async def test_semantic_chunking_quality(self, temp_document_file):
        """Test quality of semantic chunking that preserves meaning"""
        parser = UnifiedDocumentParser()
        result = await parser.parse_for_rag(temp_document_file, chunk_size=1000)
        
        # Test that test cases are not split inappropriately
        test_case_chunks = [chunk for chunk in result.chunks if 'Test Case [' in chunk]
        
        for chunk in test_case_chunks:
            # Each test case chunk should contain key elements
            if 'Test Steps' in chunk:
                assert 'Expected Result' in chunk or any('Expected Result' in other for other in result.chunks)
            
            # Should not split in middle of numbered lists
            lines = chunk.split('\n')
            numbered_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit() and '. ' in line]
            if numbered_lines:
                # If chunk contains numbered items, should start with 1 or be continuation
                first_num = numbered_lines[0].strip().split('.')[0]
                assert first_num == '1' or any('1. ' in prev_chunk for prev_chunk in result.chunks[:result.chunks.index(chunk)])

    @pytest.mark.asyncio 
    async def test_error_resilience_in_enhanced_processing(self):
        """Test error handling in enhanced processing pipeline"""
        parser = UnifiedDocumentParser()
        
        # Test with malformed document
        malformed_content = """# Incomplete Document
        
        ## Section with no content
        
        ### Test Case [BROKEN
        **Description**: Missing closing bracket
        **Steps**:
        1. Step without
        **Expected**: No result section
        
        ```json
        {
          "invalid": "json
        }
        
        | Broken | Table
        |--------|
        | Missing | Cell |
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(malformed_content)
            malformed_file = f.name
        
        try:
            # Should handle malformed content gracefully
            result = await parser.parse_for_rag(malformed_file)
            
            # Should still produce valid output
            assert result is not None
            assert result.content == malformed_content
            assert len(result.chunks) > 0
            assert result.metadata.document_type is not None
            
            # Structured content should handle errors gracefully
            assert result.structured_content is not None
            
        finally:
            os.unlink(malformed_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])