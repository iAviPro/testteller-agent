"""
TestTellerAgent implementation for test case generation.
"""
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from testteller.config import settings
from testteller.core.llm.llm_manager import LLMManager
from testteller.core.vector_store.chromadb_manager import ChromaDBManager
from testteller.core.data_ingestion.document_loader import DocumentLoader
from testteller.core.data_ingestion.code_loader import CodeLoader
from testteller.core.data_ingestion.unified_document_parser import UnifiedDocumentParser, ParseMode
from testteller.generator_agent.prompts import TEST_CASE_GENERATION_PROMPT_TEMPLATE, get_test_case_generation_prompt
import hashlib

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_COLLECTION_NAME = "test_documents_non_prod"


class TestTellerAgent:
    """Agent for generating test cases using RAG approach."""
    __test__ = False  # Tell pytest this is not a test class

    def __init__(
        self,
        collection_name: Optional[str] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        """
        Initialize the TestTellerAgent.

        Args:
            collection_name: Name of the ChromaDB collection (optional)
            llm_manager: Instance of LLMManager (optional)
        """
        self.collection_name = collection_name or self._get_collection_name()
        self.llm_manager = llm_manager or LLMManager()
        self.vector_store = ChromaDBManager(
            llm_manager=self.llm_manager,
            collection_name=self.collection_name
        )
        self.document_loader = DocumentLoader()
        self.code_loader = CodeLoader()
        self.unified_parser = UnifiedDocumentParser()
        logger.info(
            "Initialized TestTellerAgent with collection '%s' and LLM provider '%s'",
            self.collection_name, self.llm_manager.provider)

    def _get_collection_name(self) -> str:
        """Get collection name from settings or use default."""
        try:
            if settings and settings.chromadb:
                return settings.chromadb.__dict__.get('default_collection_name', DEFAULT_COLLECTION_NAME)
        except Exception as e:
            logger.debug("Could not get collection name from settings: %s", e)
        return DEFAULT_COLLECTION_NAME

    async def ingest_documents_from_path(self, path: str, enhanced_parsing: bool = True, chunk_size: int = 1000) -> None:
        """
        Ingest documents from a file or directory with enhanced parsing.
        
        Args:
            path: File or directory path
            enhanced_parsing: Use unified parser for enhanced metadata and chunking
            chunk_size: Size of text chunks for better retrieval
        """
        try:
            if os.path.isfile(path):
                await self._ingest_single_document(path, enhanced_parsing, chunk_size)
            elif os.path.isdir(path):
                await self._ingest_directory(path, enhanced_parsing, chunk_size)
            else:
                raise ValueError(f"Path not found: {path}")
            
            logger.info("Ingested documents from path: %s", path)
        except Exception as e:
            logger.error("Error ingesting documents: %s", e)
            raise
    
    async def _ingest_single_document(self, file_path: str, enhanced_parsing: bool, chunk_size: int) -> None:
        """Ingest a single document with optional enhanced parsing."""
        if enhanced_parsing:
            # Use unified parser for enhanced ingestion
            try:
                parsed_doc = await self.unified_parser.parse_for_rag(file_path, chunk_size)
                
                if parsed_doc.chunks:
                    # Ingest document chunks with rich metadata
                    contents = parsed_doc.chunks
                    metadatas = []
                    ids = []
                    
                    for i, chunk in enumerate(contents):
                        # Enhanced metadata
                        metadata = {
                            "source": file_path,
                            "type": "document",
                            "document_type": parsed_doc.metadata.document_type.value,
                            "title": parsed_doc.metadata.title or "",
                            "chunk_index": i,
                            "total_chunks": len(contents),
                            "word_count": len(chunk.split()),
                            "file_type": parsed_doc.metadata.file_type
                        }
                        
                        # Add section info if available
                        if parsed_doc.metadata.sections:
                            metadata["sections"] = ";".join(parsed_doc.metadata.sections[:5])  # Limit size
                        
                        metadatas.append(metadata)
                        
                        # Generate unique ID for each chunk
                        chunk_id = hashlib.sha256(f"doc:{file_path}:chunk:{i}".encode()).hexdigest()
                        ids.append(chunk_id)
                    
                    # Add to vector store (run in thread pool to avoid blocking)
                    await asyncio.to_thread(self.vector_store.add_documents, contents, metadatas, ids)
                    
                    logger.info(
                        "Enhanced ingestion: %s (%d chunks, %s, %d words)",
                        file_path, len(contents), parsed_doc.metadata.document_type.value,
                        parsed_doc.metadata.word_count
                    )
                else:
                    # Fallback to raw content if no chunks
                    await self._ingest_document_fallback(file_path)
                    
            except Exception as e:
                logger.warning("Enhanced parsing failed for %s, falling back to basic parsing: %s", file_path, e)
                await self._ingest_document_fallback(file_path)
        else:
            # Use basic document loader
            await self._ingest_document_fallback(file_path)
    
    async def _ingest_document_fallback(self, file_path: str) -> None:
        """Fallback document ingestion using basic document loader."""
        content = await self.document_loader.load_document(file_path)
        if content:
            # Generate unique ID for the document
            doc_id = hashlib.sha256(f"doc:{file_path}".encode()).hexdigest()
            # Add to vector store (run in thread pool to avoid blocking)
            await asyncio.to_thread(
                self.vector_store.add_documents,
                [content],
                [{"source": file_path, "type": "document"}],
                [doc_id]
            )
        else:
            logger.warning("No content loaded from document: %s", file_path)
    
    async def _ingest_directory(self, dir_path: str, enhanced_parsing: bool, chunk_size: int) -> None:
        """Ingest all documents from a directory."""
        from pathlib import Path
        
        supported_extensions = {'.md', '.txt', '.pdf', '.docx', '.xlsx', '.py', '.js', '.java', '.html', '.css', '.json', '.yaml', '.log'}
        
        dir_path_obj = Path(dir_path)
        file_paths = []
        
        # Collect all supported files
        for file_path in dir_path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_paths.append(str(file_path))
        
        if not file_paths:
            logger.warning("No supported documents found in directory: %s", dir_path)
            return
        
        # Process files with enhanced parsing if requested
        if enhanced_parsing:
            # Use batch processing for efficiency
            try:
                parsed_docs = await self.unified_parser.batch_parse(
                    file_paths, ParseMode.RAG_INGESTION, max_concurrency=3
                )
                
                # Process each parsed document
                for parsed_doc in parsed_docs:
                    if parsed_doc and parsed_doc.chunks:
                        await self._add_parsed_document_to_store(parsed_doc)
                    elif parsed_doc:
                        # Fallback for documents without chunks
                        await self._ingest_document_fallback(parsed_doc.metadata.file_path)
                
                logger.info("Enhanced directory ingestion completed: %d documents from %s", 
                           len(parsed_docs), dir_path)
                
            except Exception as e:
                logger.warning("Batch enhanced parsing failed for directory %s, falling back: %s", dir_path, e)
                await self._ingest_directory_fallback(file_paths)
        else:
            # Use basic document loader for all files
            await self._ingest_directory_fallback(file_paths)
    
    async def _add_parsed_document_to_store(self, parsed_doc) -> None:
        """Add a parsed document to the vector store."""
        contents = parsed_doc.chunks
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(contents):
            metadata = {
                "source": parsed_doc.metadata.file_path,
                "type": "document",
                "document_type": parsed_doc.metadata.document_type.value,
                "title": parsed_doc.metadata.title or "",
                "chunk_index": i,
                "total_chunks": len(contents),
                "word_count": len(chunk.split()),
                "file_type": parsed_doc.metadata.file_type
            }
            
            if parsed_doc.metadata.sections:
                metadata["sections"] = ";".join(parsed_doc.metadata.sections[:5])
            
            metadatas.append(metadata)
            
            chunk_id = hashlib.sha256(f"doc:{parsed_doc.metadata.file_path}:chunk:{i}".encode()).hexdigest()
            ids.append(chunk_id)
        
        await asyncio.to_thread(self.vector_store.add_documents, contents, metadatas, ids)
    
    async def _ingest_directory_fallback(self, file_paths: List[str]) -> None:
        """Fallback directory ingestion using basic document loader."""
        docs = []
        for file_path in file_paths:
            try:
                content = await self.document_loader.load_document(file_path)
                if content:
                    docs.append((content, file_path))
            except Exception as e:
                logger.warning("Failed to load document %s: %s", file_path, e)
        
        if docs:
            contents, paths = zip(*docs)
            ids = [hashlib.sha256(f"doc:{p}".encode()).hexdigest() for p in paths]
            await asyncio.to_thread(
                self.vector_store.add_documents,
                list(contents),
                [{"source": p, "type": "document"} for p in paths],
                ids
            )

    async def ingest_code_from_source(self, source_path: str, cleanup_github_after: bool = True) -> None:
        """Ingest code from GitHub repository or local folder."""
        try:
            is_remote = "://" in source_path or source_path.startswith("git@")
            if is_remote:
                code_files = await self.code_loader.load_code_from_repo(source_path)
                if cleanup_github_after:
                    await self.code_loader.cleanup_repo(source_path)
            else:
                code_files = await self.code_loader.load_code_from_local_folder(source_path)

            if code_files:
                contents, paths = zip(*code_files)
                # Generate unique IDs based on source path and file path
                ids = [
                    hashlib.sha256(
                        f"{source_path}:{str(p)}".encode()).hexdigest()
                    for p in paths
                ]
                # Add to vector store (run in thread pool to avoid blocking)
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    list(contents),
                    [{"source": p, "type": "code"} for p in paths],
                    ids
                )
                logger.info("Ingested code from source: %s", source_path)
            else:
                logger.warning(
                    "No code files loaded from source: %s", source_path)
        except Exception as e:
            logger.error("Error ingesting code: %s", e)
            raise

    async def get_ingested_data_count(self) -> int:
        """Get count of ingested documents."""
        return await self.vector_store.get_collection_count_async()

    async def clear_ingested_data(self) -> None:
        """Clear all ingested data."""
        try:
            self.vector_store.clear_collection()
            await self.code_loader.cleanup_all_repos()
            logger.info("Cleared all ingested data")
        except Exception as e:
            logger.error("Error clearing data: %s", e)
            raise

    async def generate_test_cases(
        self,
        code_context: str,
        n_retrieved_docs: int = 5
    ) -> str:
        """
        Generate test cases for given code context.

        Args:
            code_context: Code to generate tests for
            n_retrieved_docs: Number of similar documents to retrieve

        Returns:
            Generated test cases as string
        """
        try:
            logger.info("Starting test case generation for query: '%.50s...'", code_context)
            
            # Query similar test cases using async method to avoid blocking
            logger.debug("Querying vector store for similar documents...")
            results = await asyncio.to_thread(
                self.vector_store.query_similar,
                query_text=code_context,
                n_results=n_retrieved_docs
            )
            similar_tests = results.get('documents', [[]])[0]
            logger.debug("Retrieved %d similar documents from vector store", len(similar_tests))

            # Get provider-optimized prompt
            current_provider = self.llm_manager.get_current_provider()
            logger.debug("Using LLM provider: %s", current_provider)
            prompt = get_test_case_generation_prompt(
                provider=current_provider,
                context="\n\n".join(similar_tests),
                query=code_context
            )

            # Generate test cases using LLM Manager
            logger.debug("Sending request to LLM for test case generation...")
            response_text = await self.llm_manager.generate_text_async(prompt)
            logger.info("Generated test cases for code context using %s provider with optimized prompt",
                        self.llm_manager.provider)
            logger.debug("Test case generation completed successfully")
            return response_text

        except Exception as e:
            logger.error("Error generating test cases: %s", e)
            raise

    def add_test_cases(
        self,
        test_cases: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add test cases to the vector store.

        Args:
            test_cases: List of test case texts
            metadatas: Optional metadata for each test case
            ids: Optional IDs for each test case
        """
        try:
            # Keep synchronous for this utility method (mainly used in tests)
            self.vector_store.add_documents(
                documents=test_cases,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Added %d test cases to the vector store",
                        len(test_cases))
        except Exception as e:
            logger.error("Error adding test cases: %s", e)
            raise

    def clear_test_cases(self) -> None:
        """Clear all test cases from the vector store."""
        try:
            self.vector_store.clear_collection()
            logger.info("Cleared all test cases from the vector store")
        except Exception as e:
            logger.error("Error clearing test cases: %s", e)
            raise

    def _calculate_quality_score(self, test_cases_content: str) -> float:
        """
        Calculate quality score (0.0 to 1.0) for generated test cases.
        
        Args:
            test_cases_content: The generated test cases content
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 1.0
        
        # Check for completeness - presence of TODOs indicates incomplete generation
        if "TODO" in test_cases_content or "FIXME" in test_cases_content:
            score -= 0.3
            
        # Check for structure - should have test cases with proper formatting
        if not re.search(r'Test Case \d+:', test_cases_content, re.IGNORECASE):
            score -= 0.2
            
        # Check for detail level - too short indicates poor quality
        if len(test_cases_content) < 500:
            score -= 0.2
            
        # Check for test steps - good test cases have clear steps
        step_count = len(re.findall(r'Step \d+:', test_cases_content, re.IGNORECASE))
        if step_count < 3:
            score -= 0.1
            
        # Check for expected results
        if not re.search(r'Expected Result[s]?:', test_cases_content, re.IGNORECASE):
            score -= 0.1
            
        return max(0.0, score)
    
    def _parse_test_cases(self, test_cases_content: str) -> List[str]:
        """
        Parse test cases content into individual chunks.
        
        Args:
            test_cases_content: The full test cases content
            
        Returns:
            List of individual test case chunks
        """
        # Split by test case markers
        test_case_pattern = r'Test Case \d+:|Test Case ID:|Test Case:'
        test_cases = re.split(test_case_pattern, test_cases_content, flags=re.IGNORECASE)
        
        # Filter out empty chunks and clean up
        chunks = []
        for i, test_case in enumerate(test_cases[1:], 1):  # Skip first empty split
            if test_case.strip():
                # Add the test case marker back
                chunk = f"Test Case {i}: {test_case.strip()}"
                chunks.append(chunk)
                
        # If no clear test case markers, treat as single chunk
        if not chunks:
            chunks = [test_cases_content]
            
        return chunks
    
    async def store_generated_test_cases(
        self, 
        test_cases_content: str, 
        query: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store generated test cases back into vector store for future reference.
        
        Args:
            test_cases_content: The generated test cases content
            query: Original query used for generation
            metadata: Additional metadata about generation
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Calculate quality score
            quality_score = self._calculate_quality_score(test_cases_content)
            
            # Only store high-quality test cases
            min_quality = float(os.getenv('MIN_QUALITY_SCORE_FOR_STORAGE', '0.7'))
            if quality_score < min_quality:
                logger.info(
                    "Skipping storage of test cases due to low quality score: %.2f < %.2f",
                    quality_score, min_quality
                )
                return False
            
            # Parse test cases into chunks
            test_case_chunks = self._parse_test_cases(test_cases_content)
            
            # Check for duplicates before storing
            stored_count = 0
            skipped_count = 0
            
            for i, chunk in enumerate(test_case_chunks):
                # Check if similar test case already exists
                if await self._is_duplicate_test_case(chunk, query):
                    logger.debug(f"Skipping duplicate test case chunk {i}")
                    skipped_count += 1
                    continue
                
                # Prepare metadata
                base_metadata = {
                    "type": "generated_test_case",
                    "source": "testteller_generator",
                    "generation_query": query,
                    "generation_timestamp": datetime.now().isoformat(),
                    "llm_provider": self.llm_manager.provider,
                    "quality_score": quality_score,
                    "document_type": "test_cases"
                }
                
                if metadata:
                    base_metadata.update(metadata)
                
                # Generate unique ID for this chunk
                chunk_id = hashlib.sha256(
                    f"generated:{query}:{i}:{datetime.now().isoformat()}".encode()
                ).hexdigest()
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(test_case_chunks)
                
                # Add to vector store (run in thread pool to avoid blocking)
                await asyncio.to_thread(
                    self.vector_store.add_documents,
                    documents=[chunk],
                    metadatas=[chunk_metadata],
                    ids=[chunk_id]
                )
                stored_count += 1
            
            logger.info(
                "Stored %d generated test case chunks (skipped %d duplicates, quality: %.2f)",
                stored_count, skipped_count, quality_score
            )
            return stored_count > 0
            
        except Exception as e:
            logger.error("Error storing generated test cases: %s", e)
            return False
    
    async def _is_duplicate_test_case(self, test_case: str, query: str) -> bool:
        """
        Check if a similar test case already exists in the vector store.
        
        Args:
            test_case: The test case content to check
            query: The original query used for generation
            
        Returns:
            True if a very similar test case exists, False otherwise
        """
        try:
            # Query for similar test cases
            results = self.vector_store.query_similar(
                query_text=test_case[:500],  # Use first 500 chars for similarity check
                n_results=3,
                metadata_filter={
                    "type": "generated_test_case",
                    "generation_query": query  # Look for test cases from same query
                }
            )
            
            if results and results.get('documents'):
                # Check similarity of results
                for doc in results['documents'][0]:
                    # Simple similarity check - if >80% of content matches, consider duplicate
                    if self._calculate_similarity(test_case, doc) > 0.8:
                        return True
                        
        except Exception as e:
            logger.warning(f"Error checking for duplicate test cases: {e}")
            
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def close(self):
        """Clean up resources and connections."""
        try:
            logger.debug("Closing TestTellerAgent resources")
            
            # Close vector store if it has a close method
            if hasattr(self.vector_store, 'close'):
                self.vector_store.close()
                
            # Clear references to help garbage collection
            if hasattr(self, 'vector_store'):
                self.vector_store = None
                
            logger.debug("TestTellerAgent resources closed successfully")
        except Exception as e:
            logger.debug("Error during TestTellerAgent cleanup: %s", e)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()


# Create an alias for backward compatibility
TestTellerRagAgent = TestTellerAgent
