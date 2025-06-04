import logging
import os
import time  # For performance metrics logging
from pathlib import Path  # For checking if path is likely a URL
from .prompts import TEST_CASE_GENERATION_PROMPT_TEMPLATE
from .llm.gemini_client import GeminiClient
from .vector_store.chromadb_manager import ChromaDBManager
from .data_ingestion.document_loader import DocumentLoader
from .data_ingestion.code_loader import CodeLoader
from .data_ingestion.text_splitter import TextSplitter
from .config import settings


logger = logging.getLogger(__name__)


class TestTellerRagAgent:
    def __init__(self, collection_name: str = settings.chroma_db.default_collection_name):
        self.gemini_client = GeminiClient()
        self.vector_store = ChromaDBManager(
            gemini_client=self.gemini_client, collection_name=collection_name)
        self.document_loader = DocumentLoader()
        self.code_loader = CodeLoader()
        self.text_splitter = TextSplitter()
        self.collection_name = collection_name
        logger.info(
            "TestCaseAgent initialized for collection: %s", collection_name)

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

        for item_path, item_content in contents_with_paths:  # item_path is the full source_identifier here
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

            # The item_path here is already the unique identifier like "https://github.com/user/repo/file.py"
            # or "local:/abs/path/to/folder/file.py" which comes from CodeLoader._read_code_files_from_path
            logger.debug(
                "%s Processed source '%s' into %d chunks (split_duration: %.3fs).",
                log_prefix, item_path, len(chunks), splitting_duration)

            for chunk_idx, chunk in enumerate(chunks):
                # Generate ID based on the unique source path of the file and the chunk content
                unique_id = self.vector_store.generate_id_from_text_and_source(
                    chunk, item_path)
                metadata = {
                    "source": item_path,  # This is the full unique path to the source file
                    "type": source_type,
                    "original_length": len(item_content),
                    "chunk_index": chunk_idx,
                    # Number of chunks from this specific file
                    "total_chunks_for_source": len(chunks)
                }
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(unique_id)

        ingestion_prep_duration = time.monotonic() - start_time_ingestion_prep
        logger.info(
            "Content preparation for %d chunks from %d original source files took %.2fs.",
            len(all_chunks), total_content_items, ingestion_prep_duration)

        if all_chunks:
            await self.vector_store.add_documents(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
        else:
            logger.info("No valid chunks to ingest from %s.", source_type)

    async def ingest_documents_from_path(self, path: str):
        logger.info("Starting ingestion of documents from path: %s", path)
        start_time = time.monotonic()
        if os.path.isfile(path):
            content = await self.document_loader.load_document(path)
            if content:
                # For single doc, item_path is the file path, source_type is 'document_file'
                await self._ingest_content([(path, content)], source_type="document_file")
            else:
                logger.warning(
                    "Could not load document content from file: %s", path)
        elif os.path.isdir(path):
            docs_with_content = await self.document_loader.load_from_directory(path)
            if docs_with_content:
                # docs_with_content is list of (file_path, content)
                await self._ingest_content(docs_with_content, source_type="document_directory")
            else:
                logger.warning("No documents loaded from directory: %s", path)
        else:
            logger.error(
                "Path does not exist or is not a file/directory: %s", path)
        duration = time.monotonic() - start_time
        logger.info(
            "Document ingestion from path '%s' completed in %.2fs.", path, duration)

    async def ingest_code_from_source(self, source_path: str, cleanup_github_after: bool = True):
        """
        Ingests code from either a GitHub repository URL or a local folder path.
        """
        logger.info("Starting ingestion of code from source: %s", source_path)
        start_time = time.monotonic()
        # This will be list of (full_source_path_identifier, content)
        code_files_content = []

        is_url = "://" in source_path or source_path.startswith("git@")

        if is_url:
            logger.info(
                "Source '%s' identified as a remote repository URL.", source_path)
            code_files_content = await self.code_loader.load_code_from_repo(source_path)
            if cleanup_github_after:
                await self.code_loader.cleanup_repo(source_path)
            source_type_log = "github_code"
        else:
            local_path = Path(source_path)
            if not local_path.exists():
                logger.error("Local code path does not exist: %s", source_path)
                print(f"Error: Local code path '{source_path}' not found.")
                return
            if not local_path.is_dir():
                logger.error(
                    "Local code path is not a directory: %s", source_path)
                print(
                    f"Error: Local code path '{source_path}' is not a directory.")
                return

            logger.info(
                "Source '%s' identified as a local folder path.", source_path)
            # Pass the original source_path, CodeLoader will resolve and use absolute path for its identifier
            code_files_content = await self.code_loader.load_code_from_local_folder(source_path)
            source_type_log = "local_folder_code"

        if code_files_content:
            # code_files_content is already a list of (unique_file_identifier, content)
            await self._ingest_content(code_files_content, source_type=source_type_log)
        else:
            logger.warning("No code files loaded from source: %s", source_path)

        duration = time.monotonic() - start_time
        logger.info(
            "Code ingestion from source '%s' completed in %.2fs.", source_path, duration)

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
                    distance = doc_data.get('distance')
                    distance_str = f"{distance:.4f}" if distance is not None else "N/A"
                    context_parts.append(
                        f"--- Context Document {i+1} (Source: {source}, Distance: {distance_str}) ---\n{doc_content}\n--- End Context Document {i+1} ---")
                    logger.debug(
                        "Retrieved doc %d: Source: %s, Distance: %s, Preview: %s...",
                        i+1, source, distance_str, doc_content[:100])
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
