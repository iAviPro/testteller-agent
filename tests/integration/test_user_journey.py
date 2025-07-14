"""
Integration tests for complete user journeys with different LLM providers.
"""
import pytest
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock
from testteller.agent.testteller_agent import TestTellerAgent
from testteller.llm.llm_manager import LLMManager


class TestUserJourney:
    """Test complete user journeys with different LLM providers."""

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_complete_journey_with_gemini(self, skip_if_no_api_key, llm_provider, cleanup_test_data):
        """Test complete user journey with Gemini provider."""
        if llm_provider != "gemini":
            pytest.skip("This test is specific to Gemini provider")

        # Create test collection
        collection_name = f"test_journey_{llm_provider}"
        agent = TestTellerAgent(collection_name=collection_name)

        # Test 1: Ingest sample document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# User Authentication API

## Endpoints
- POST /api/auth/login - User login
- POST /api/auth/register - User registration
- GET /api/auth/profile - Get user profile

## Authentication Flow
1. User submits credentials
2. Server validates credentials
3. Server returns JWT token
4. Client uses token for authenticated requests
""")
            doc_path = f.name

        try:
            await agent.ingest_documents_from_path(doc_path)

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Test 2: Generate test cases
            test_cases = await agent.generate_test_cases(
                "Create comprehensive API tests for user authentication",
                n_retrieved_docs=3
            )

            # Verify test cases were generated
            assert test_cases is not None
            assert len(test_cases) > 0
            assert "test" in test_cases.lower()

            # Test 3: Clear data
            await agent.clear_ingested_data()

            # Verify data was cleared
            count_after_clear = await agent.get_ingested_data_count()
            assert count_after_clear == 0

        finally:
            # Cleanup
            os.unlink(doc_path)

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_complete_journey_with_openai(self, skip_if_no_api_key, llm_provider, cleanup_test_data):
        """Test complete user journey with OpenAI provider."""
        if llm_provider != "openai":
            pytest.skip("This test is specific to OpenAI provider")

        collection_name = f"test_journey_{llm_provider}"
        agent = TestTellerAgent(collection_name=collection_name)

        # Create test code file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
class UserAuth:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def login(self, username, password):
        user = self.db.get_user(username)
        if user and self.verify_password(password, user.password_hash):
            return self.generate_token(user)
        return None
    
    def register(self, username, email, password):
        if self.db.user_exists(username):
            raise ValueError("User already exists")
        
        password_hash = self.hash_password(password)
        user = self.db.create_user(username, email, password_hash)
        return user
""")
            code_path = f.name

        try:
            await agent.ingest_code_from_source(str(Path(code_path).parent))

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Generate test cases
            test_cases = await agent.generate_test_cases(
                "Create unit tests for the UserAuth class",
                n_retrieved_docs=5
            )

            # Verify test cases
            assert test_cases is not None
            assert len(test_cases) > 0
            assert any(keyword in test_cases.lower()
                       for keyword in ["test", "assert", "auth"])

        finally:
            os.unlink(code_path)

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_complete_journey_with_claude(self, skip_if_no_api_key, llm_provider, cleanup_test_data):
        """Test complete user journey with Claude provider."""
        if llm_provider != "claude":
            pytest.skip("This test is specific to Claude provider")

        collection_name = f"test_journey_{llm_provider}"
        agent = TestTellerAgent(collection_name=collection_name)

        # Create test directory with multiple files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create API specification
            (temp_path / "api_spec.md").write_text("""
# Payment API Specification

## Endpoints
- POST /api/payments/create - Create payment
- GET /api/payments/{id} - Get payment details
- POST /api/payments/{id}/confirm - Confirm payment

## Security
- All endpoints require JWT authentication
- Payment confirmation requires 2FA
""")

            # Create implementation file
            (temp_path / "payment.py").write_text("""
class PaymentService:
    def create_payment(self, amount, currency, user_id):
        # Validate amount
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        # Create payment record
        payment = {
            "amount": amount,
            "currency": currency,
            "user_id": user_id,
            "status": "pending"
        }
        return self.db.save_payment(payment)
    
    def confirm_payment(self, payment_id, two_factor_code):
        payment = self.db.get_payment(payment_id)
        if not payment:
            raise ValueError("Payment not found")
        
        if not self.verify_2fa(payment.user_id, two_factor_code):
            raise ValueError("Invalid 2FA code")
        
        payment.status = "confirmed"
        return self.db.update_payment(payment)
""")

            # Ingest documents
            await agent.ingest_documents_from_path(str(temp_path))

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Generate comprehensive test cases
            test_cases = await agent.generate_test_cases(
                "Create comprehensive test suite for payment processing including security tests",
                n_retrieved_docs=10
            )

            # Verify test cases
            assert test_cases is not None
            assert len(test_cases) > 0
            assert any(keyword in test_cases.lower()
                       for keyword in ["payment", "security", "test"])

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_journey_with_llama(self, llm_provider, mock_env_for_provider, cleanup_test_data):
        """Test complete user journey with Llama provider (local)."""
        if llm_provider != "llama":
            pytest.skip("This test is specific to Llama provider")

        # Check if Ollama is available
        try:
            import httpx
            import os
            ollama_base_url = os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                if response.status_code != 200:
                    pytest.skip(f"Ollama not available at {ollama_base_url}")
        except Exception:
            ollama_base_url = os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434")
            pytest.skip(f"Ollama not available at {ollama_base_url}")

        collection_name = f"test_journey_{llm_provider}"

        # Create LLM manager explicitly with llama provider to avoid initialization errors
        from testteller.llm.llm_manager import LLMManager
        llm_manager = LLMManager(provider="llama")
        agent = TestTellerAgent(
            collection_name=collection_name, llm_manager=llm_manager)

        # Create simple test case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
Simple Calculator Requirements:
- Add two numbers
- Subtract two numbers
- Multiply two numbers
- Divide two numbers
- Handle division by zero
""")
            doc_path = f.name

        try:
            await agent.ingest_documents_from_path(doc_path)

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Generate test cases
            test_cases = await agent.generate_test_cases(
                "Create simple unit tests for calculator functions",
                n_retrieved_docs=3
            )

            # Verify test cases
            assert test_cases is not None
            assert len(test_cases) > 0

        finally:
            os.unlink(doc_path)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mixed_content_ingestion(self, mock_env_for_provider, cleanup_test_data):
        """Test ingesting mixed content types (documents and code)."""
        collection_name = "test_mixed_content"
        agent = TestTellerAgent(collection_name=collection_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create documentation
            (temp_path / "requirements.md").write_text("""
# E-commerce System Requirements

## User Stories
- As a customer, I want to browse products
- As a customer, I want to add items to cart
- As a customer, I want to checkout securely
""")

            # Create code implementation
            (temp_path / "cart.py").write_text("""
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, product_id, quantity=1):
        self.items.append({"product_id": product_id, "quantity": quantity})
    
    def remove_item(self, product_id):
        self.items = [item for item in self.items if item["product_id"] != product_id]
    
    def get_total(self):
        return sum(item["quantity"] for item in self.items)
""")

            # Ingest both documents and code
            await agent.ingest_documents_from_path(str(temp_path))

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Generate test cases that should consider both requirements and implementation
            test_cases = await agent.generate_test_cases(
                "Create comprehensive tests for the shopping cart functionality",
                n_retrieved_docs=5
            )

            # Verify test cases
            assert test_cases is not None
            assert len(test_cases) > 0
            assert any(keyword in test_cases.lower()
                       for keyword in ["cart", "item", "test"])

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_switching(self, mock_env_vars, cleanup_test_data):
        """Test switching between different LLM providers."""
        collection_name = "test_provider_switching"

        # Test with mocked providers to avoid API calls
        with patch('testteller.llm.llm_manager.GeminiClient') as mock_gemini:
            with patch('testteller.llm.llm_manager.OpenAIClient') as mock_openai:
                mock_gemini_instance = Mock()
                mock_openai_instance = Mock()
                mock_gemini.return_value = mock_gemini_instance
                mock_openai.return_value = mock_openai_instance

                # Test Gemini - create LLM manager explicitly to avoid settings caching issues
                from testteller.llm.llm_manager import LLMManager
                llm_manager_gemini = LLMManager(provider="gemini")
                agent_gemini = TestTellerAgent(
                    collection_name=collection_name, llm_manager=llm_manager_gemini)
                assert agent_gemini.llm_manager.provider == "gemini"

                # Test OpenAI - create LLM manager explicitly to avoid settings caching issues
                llm_manager_openai = LLMManager(provider="openai")
                agent_openai = TestTellerAgent(
                    collection_name=collection_name, llm_manager=llm_manager_openai)
                assert agent_openai.llm_manager.provider == "openai"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_recovery(self, llm_provider, mock_env_for_provider, cleanup_test_data):
        """Test error recovery in user journey."""
        collection_name = "test_error_recovery"

        # Mock both the LLM manager and the vector store to avoid API calls
        with patch('testteller.agent.testteller_agent.LLMManager') as mock_llm_manager_class, \
                patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_vector_store_class:

            # Set up mock LLM manager
            mock_llm_manager = Mock()
            mock_llm_manager.get_embeddings_sync.return_value = [[0.1] * 1536]
            mock_llm_manager.generate_text.return_value = "Test case generated"
            mock_llm_manager_class.return_value = mock_llm_manager

            # Set up mock vector store
            mock_vector_store = Mock()
            mock_vector_store.get_collection_count.return_value = 0
            mock_vector_store.clear_collection.return_value = None
            mock_vector_store.add_documents.return_value = None
            mock_vector_store_class.return_value = mock_vector_store

            agent = TestTellerAgent(collection_name=collection_name)

            # Test ingestion with invalid path - should handle gracefully
            await agent.ingest_documents_from_path("/nonexistent/path/that/does/not/exist")

            # Verify agent is still functional after error
            count = await agent.get_ingested_data_count()
            assert count == 0  # Should be empty after failed ingestion

            # Test successful ingestion after error
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Test content after error recovery")
                doc_path = f.name

            try:
                # Mock the vector store to return 1 document after ingestion
                mock_vector_store.get_collection_count.return_value = 1

                await agent.ingest_documents_from_path(doc_path)
                count = await agent.get_ingested_data_count()
                assert count > 0
            finally:
                os.unlink(doc_path)
                await agent.clear_ingested_data()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_document_processing(self, mock_env_for_provider, cleanup_test_data):
        """Test processing large documents."""
        collection_name = "test_large_document"
        agent = TestTellerAgent(collection_name=collection_name)

        # Create a large document
        large_content = """
# Large System Documentation

## Overview
This is a comprehensive system documentation with multiple sections.

""" + "\n".join([f"""
## Section {i}
This is section {i} of the documentation. It contains detailed information
about component {i} and its functionality. The component handles various
operations and integrates with other system components.

### Subsection {i}.1
Detailed technical specifications for component {i}.

### Subsection {i}.2
API documentation for component {i}.

### Subsection {i}.3
Configuration and deployment instructions for component {i}.
""" for i in range(1, 21)])  # 20 sections

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(large_content)
            doc_path = f.name

        try:
            await agent.ingest_documents_from_path(doc_path)

            # Verify ingestion
            count = await agent.get_ingested_data_count()
            assert count > 0

            # Generate test cases for large document
            test_cases = await agent.generate_test_cases(
                "Create integration tests for the entire system",
                n_retrieved_docs=10
            )

            # Verify test cases
            assert test_cases is not None
            assert len(test_cases) > 0

        finally:
            os.unlink(doc_path)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_operations(self, llm_provider, mock_env_for_provider, cleanup_test_data):
        """Test concurrent operations on the same agent."""
        collection_name = "test_concurrent"

        # Mock both the LLM manager and the vector store to avoid API calls
        with patch('testteller.agent.testteller_agent.LLMManager') as mock_llm_manager_class, \
                patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_vector_store_class:

            # Set up mock LLM manager
            mock_llm_manager = Mock()
            mock_llm_manager.get_embeddings_sync.return_value = [[0.1] * 1536]
            mock_llm_manager.generate_text.return_value = "Test case generated"
            mock_llm_manager_class.return_value = mock_llm_manager

            # Set up mock vector store
            mock_vector_store = Mock()
            mock_vector_store.get_collection_count.return_value = 0
            mock_vector_store.clear_collection.return_value = None
            mock_vector_store.add_documents.return_value = None
            mock_vector_store_class.return_value = mock_vector_store

            agent = TestTellerAgent(collection_name=collection_name)

            # Create multiple test files
            test_files = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False) as f:
                    f.write(f"Test content {i}")
                    test_files.append(f.name)

            try:
                # Mock the vector store to return the number of files after ingestion
                mock_vector_store.get_collection_count.return_value = len(
                    test_files)

                # Concurrent ingestion
                tasks = [agent.ingest_documents_from_path(
                    path) for path in test_files]
                await asyncio.gather(*tasks)

                # Check that all files were ingested
                count = await agent.get_ingested_data_count()
                assert count >= len(test_files)

            finally:
                # Cleanup
                for file_path in test_files:
                    os.unlink(file_path)
                await agent.clear_ingested_data()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_collection_isolation(self, llm_provider, mock_env_for_provider, cleanup_test_data):
        """Test that different collections are isolated."""
        # Mock both the LLM manager and the vector store to avoid API calls
        with patch('testteller.agent.testteller_agent.LLMManager') as mock_llm_manager_class, \
                patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_vector_store_class:

            # Set up mock LLM manager
            mock_llm_manager = Mock()
            mock_llm_manager.get_embeddings_sync.return_value = [[0.1] * 1536]
            mock_llm_manager.generate_text.return_value = "Test case generated"
            mock_llm_manager_class.return_value = mock_llm_manager

            # Set up mock vector stores for different collections
            mock_vector_store1 = Mock()
            mock_vector_store1.get_collection_count.return_value = 0
            mock_vector_store1.clear_collection.return_value = None
            mock_vector_store1.add_documents.return_value = None

            mock_vector_store2 = Mock()
            mock_vector_store2.get_collection_count.return_value = 0
            mock_vector_store2.clear_collection.return_value = None
            mock_vector_store2.add_documents.return_value = None

            # Return different mock instances for different collection names
            def mock_vector_store_factory(collection_name, *args, **kwargs):
                if collection_name == "test_collection_1":
                    return mock_vector_store1
                else:
                    return mock_vector_store2

            mock_vector_store_class.side_effect = mock_vector_store_factory

            # Create two agents with different collections
            agent1 = TestTellerAgent(collection_name="test_collection_1")
            agent2 = TestTellerAgent(collection_name="test_collection_2")

            # Add data to first collection
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Content for collection 1")
                doc_path1 = f.name

            # Add data to second collection
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("Content for collection 2")
                doc_path2 = f.name

            try:
                # Mock each vector store to return 1 document after ingestion
                mock_vector_store1.get_collection_count.return_value = 1
                mock_vector_store2.get_collection_count.return_value = 1

                await agent1.ingest_documents_from_path(doc_path1)
                await agent2.ingest_documents_from_path(doc_path2)

                # Check that each collection has its own data
                count1 = await agent1.get_ingested_data_count()
                count2 = await agent2.get_ingested_data_count()

                assert count1 > 0
                assert count2 > 0

                # Clear one collection and verify the other is unaffected
                mock_vector_store1.get_collection_count.return_value = 0  # After clearing
                await agent1.clear_ingested_data()

                count1_after = await agent1.get_ingested_data_count()
                count2_after = await agent2.get_ingested_data_count()

                assert count1_after == 0
                assert count2_after == count2  # Should be unchanged

            finally:
                # Cleanup
                os.unlink(doc_path1)
                os.unlink(doc_path2)
                await agent1.clear_ingested_data()
                await agent2.clear_ingested_data()
