"""
Text splitting utilities for document ingestion.
"""
import logging
import os
import time
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from testteller.config import settings
from testteller.core.data_ingestion.code_loader import CodeLoader
from testteller.core.data_ingestion.document_loader import DocumentLoader
from testteller.core.llm.gemini_client import GeminiClient
from testteller.core.vector_store.chromadb_manager import ChromaDBManager
from testteller.generator_agent.prompts import TEST_CASE_GENERATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_LENGTH_FUNCTION = len
DEFAULT_ADD_START_INDEX = True


class TextSplitter:
    """Handles text splitting for document ingestion."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        length_function: Optional[callable] = None,
        add_start_index: Optional[bool] = None
    ):
        """
        Initialize text splitter with configuration.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            length_function: Function to measure text length
            add_start_index: Whether to add start index to chunk metadata
        """
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        self.length_function = length_function or DEFAULT_LENGTH_FUNCTION
        self.add_start_index = add_start_index if add_start_index is not None else DEFAULT_ADD_START_INDEX

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            add_start_index=self.add_start_index
        )

        logger.info(
            "Initialized TextSplitter with chunk_size=%d, chunk_overlap=%d",
            self.chunk_size,
            self.chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        try:
            chunks = self.splitter.split_text(text)
            logger.info("Split text into %d chunks", len(chunks))
            return chunks
        except Exception as e:
            logger.error("Error splitting text: %s", e)
            raise

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        Split multiple texts into chunks.

        Args:
            texts: List of texts to split

        Returns:
            List of text chunks from all texts
        """
        try:
            all_chunks = []
            for text in texts:
                chunks = self.split_text(text)
                all_chunks.extend(chunks)
            logger.info("Split %d texts into %d total chunks",
                        len(texts), len(all_chunks))
            return all_chunks
        except Exception as e:
            logger.error("Error splitting texts: %s", e)
            raise


class TestTellerAgent:
    def __init__(self, collection_name: str = settings.chromadb.default_collection_name):
        # Initialize clients. These are lightweight and can be created per agent instance.
        # For a long-running service, you might make them singletons or manage them differently.
        self.gemini_client = GeminiClient()
        self.vector_store = ChromaDBManager(
            gemini_client=self.gemini_client, collection_name=collection_name)
        self.document_loader = DocumentLoader()
        self.code_loader = CodeLoader()
        self.text_splitter = TextSplitter()
        self.collection_name = collection_name
        logger.info(
            "TestTellerRAGAgent initialized for collection: %s", collection_name)

    async def _ingest_content(self, contents_with_paths: list[tuple[str, str]], source_type: str):
        if not contents_with_paths:
            logger.info(
                "No content provided for ingestion from %s.", source_type)
            return

        all_chunks = []
        all_metadatas = []
        all_ids = []

        total_content_items = len(contents_with_paths)
        processed_count = 0
        start_time_ingestion_prep = time.monotonic()

        for item_path, item_content in contents_with_paths:
            processed_count += 1
            log_prefix = f"[{processed_count}/{total_content_items}]"
            if not item_content or not item_content.strip():
                logger.warning(
                    "%s Skipping empty content from %s", log_prefix, item_path)
                continue

            start_time_splitting = time.monotonic()
            chunks = self.text_splitter.split_text(
                item_content)  # This is synchronous
            splitting_duration = time.monotonic() - start_time_splitting

            if not chunks:
                logger.warning(
                    "%s No chunks generated for %s (split_duration: %.3fs).", log_prefix, item_path, splitting_duration)
                continue

            logger.debug(
                "%s Processed %s into %d chunks (split_duration: %.3fs).",
                log_prefix, item_path, len(chunks), splitting_duration)

            for chunk_idx, chunk in enumerate(chunks):
                unique_id = self.vector_store.generate_id_from_text_and_source(
                    chunk, item_path)
                metadata = {
                    "source": item_path,
                    "type": source_type,
                    "original_length": len(item_content),
                    "chunk_index": chunk_idx,
                    "total_chunks_for_source": len(chunks)
                }
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(unique_id)

        ingestion_prep_duration = time.monotonic() - start_time_ingestion_prep
        logger.info(
            "Content preparation for %d chunks from %d sources took %.2fs.",
            len(all_chunks), total_content_items, ingestion_prep_duration)

        if all_chunks:
            await self.vector_store.add_documents(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
            # Final count logged by vector_store.add_documents
        else:
            logger.info("No valid chunks to ingest from %s.", source_type)

    async def ingest_documents_from_path(self, path: str):
        logger.info("Starting ingestion of documents from path: %s", path)
        start_time = time.monotonic()
        if os.path.isfile(path):  # Sync check, acceptable for CLI startup
            content = await self.document_loader.load_document(path)
            if content:
                await self._ingest_content([(path, content)], source_type="document_file")
            else:
                logger.warning(
                    "Could not load document content from file: %s", path)
        elif os.path.isdir(path):
            docs_with_content = await self.document_loader.load_from_directory(path)
            if docs_with_content:
                await self._ingest_content(docs_with_content, source_type="document_directory")
            else:
                logger.warning("No documents loaded from directory: %s", path)
        else:
            logger.error(
                "Path does not exist or is not a file/directory: %s", path)
        duration = time.monotonic() - start_time
        logger.info(
            "Document ingestion from path '%s' completed in %.2fs.", path, duration)

    async def ingest_code_from_github(self, repo_url: str, cleanup_after: bool = True):
        logger.info("Starting ingestion of code from GitHub repo: %s", repo_url)
        start_time = time.monotonic()
        code_files_content = await self.code_loader.load_code_from_repo(repo_url)
        if code_files_content:
            await self._ingest_content(code_files_content, source_type="github_code")
        else:
            logger.warning(
                "No code files loaded from GitHub repo: %s", repo_url)

        if cleanup_after:
            await self.code_loader.cleanup_repo(repo_url)
        duration = time.monotonic() - start_time
        logger.info(
            "Code ingestion from repo '%s' completed in %.2fs.", repo_url, duration)

    async def get_ingested_data_count(self) -> int:
        return await self.vector_store.get_collection_count_async()

    async def clear_ingested_data(self):
        logger.info(
            "Clearing all ingested data from collection '%s'.", self.collection_name)
        start_time = time.monotonic()
        await self.vector_store.clear_collection_async()
        await self.code_loader.cleanup_all_repos()
        duration = time.monotonic() - start_time
        logger.info(
            "Data cleared for collection '%s' in %.2fs.", self.collection_name, duration)

    async def generate_test_cases(self, query: str, n_retrieved_docs: int = 5) -> str:
        logger.info(
            "Generating test cases for query: '%s' using collection '%s'", query, self.collection_name)
        start_time_total = time.monotonic()

        ingested_count = await self.get_ingested_data_count()
        if ingested_count == 0:
            logger.warning(
                "No data ingested in collection '%s'. Test case generation might be suboptimal.", self.collection_name)
            context_str = "No specific context documents were found in the knowledge base for this query."
        else:
            start_time_retrieval = time.monotonic()
            retrieved_docs = await self.vector_store.query_collection(query_text=query, n_results=n_retrieved_docs)
            retrieval_duration = time.monotonic() - start_time_retrieval
            logger.info(
                "Context retrieval took %.2fs, found %d documents.", retrieval_duration, len(retrieved_docs))

            if not retrieved_docs:
                logger.warning(
                    "No relevant documents found for query: '%s'", query)
                context_str = "No relevant context documents were found in the knowledge base for this query."
            else:
                context_parts = []
                for i, doc_data in enumerate(retrieved_docs):
                    source = doc_data.get('metadata', {}).get(
                        'source', 'Unknown source')
                    doc_content = doc_data.get('document', '')
                    distance = doc_data.get('distance', 'N/A')
                    context_parts.append(
                        f"--- Context Document {i+1} (Source: {source}, Distance: {distance:.4f}) ---\n{doc_content}\n--- End Context Document {i+1} ---")
                    logger.debug(
                        "Retrieved doc %d: Source: %s, Distance: %.4f, Preview: %s...",
                        i+1, source, distance, doc_content[:100])
                context_str = "\n\n".join(context_parts)

        prompt = TEST_CASE_GENERATION_PROMPT_TEMPLATE.format(
            context=context_str, query=query)
        logger.debug(
            "Constructed prompt for Gemini (first 300 chars): \n%s...", prompt[:300])

        start_time_llm = time.monotonic()
        response_text = await self.gemini_client.generate_text_async(prompt)
        llm_duration = time.monotonic() - start_time_llm
        logger.info("LLM generation took %.2fs.", llm_duration)

        total_duration = time.monotonic() - start_time_total
        logger.info("Total test case generation took %.2fs.", total_duration)
        return response_text
