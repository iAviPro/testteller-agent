"""
TestTellerAgent implementation for test case generation.
"""
import logging
import os
from typing import List, Dict, Any, Optional
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
                    
                    # Add to vector store
                    self.vector_store.add_documents(contents, metadatas, ids)
                    
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
            # Use sync call without await
            self.vector_store.add_documents(
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
        
        self.vector_store.add_documents(contents, metadatas, ids)
    
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
            self.vector_store.add_documents(
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
                # Use sync call without await
                self.vector_store.add_documents(
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
        return self.vector_store.get_collection_count()

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
            # Query similar test cases
            results = self.vector_store.query_similar(
                query_text=code_context,
                n_results=n_retrieved_docs
            )
            similar_tests = results.get('documents', [[]])[0]

            # Get provider-optimized prompt
            current_provider = self.llm_manager.get_current_provider()
            prompt = get_test_case_generation_prompt(
                provider=current_provider,
                context="\n\n".join(similar_tests),
                query=code_context
            )

            # Generate test cases using LLM Manager
            response_text = await self.llm_manager.generate_text_async(prompt)
            logger.info("Generated test cases for code context using %s provider with optimized prompt",
                        self.llm_manager.provider)
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


# Create an alias for backward compatibility
TestTellerRagAgent = TestTellerAgent
