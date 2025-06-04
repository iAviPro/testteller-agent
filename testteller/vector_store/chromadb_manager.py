import logging
import os
from typing import List, Dict, Any
import functools
import hashlib
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings  # Added
from chromadb import HttpClient  # Added
from testteller.config import settings
from testteller.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class ChromaDBManager:
    def __init__(self, gemini_client: GeminiClient, collection_name: str = settings.chroma_db.default_collection_name):
        self.gemini_client = gemini_client
        self.collection_name = collection_name
        self.db_path = settings.chroma_db.chroma_db_path  # db_path is still defined

        chroma_host = os.getenv("CHROMA_DB_HOST")
        chroma_port_str = os.getenv("CHROMA_DB_PORT")

        if chroma_host and chroma_port_str:
            try:
                chroma_port = int(chroma_port_str)
                logger.info(
                    "Connecting to ChromaDB server at %s:%d", chroma_host, chroma_port)
                self.client = HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=Settings(anonymized_telemetry=False)
                )
            except ValueError:
                logger.error(
                    "Invalid CHROMA_DB_PORT: %s. Falling back to PersistentClient.", chroma_port_str)
                try:
                    self.client = chromadb.PersistentClient(
                        path=self.db_path, settings=Settings(anonymized_telemetry=False))
                except Exception as e:
                    logger.error(
                        "Failed to initialize ChromaDB PersistentClient (fallback) at %s: %s", self.db_path, e, exc_info=True)
                    raise
            except Exception as e:  # Catch other HttpClient connection errors
                logger.error(
                    "Failed to connect to ChromaDB HttpClient at %s:%s: %s. Falling back to PersistentClient.", chroma_host, chroma_port_str, e, exc_info=True)
                try:
                    self.client = chromadb.PersistentClient(
                        path=self.db_path, settings=Settings(anonymized_telemetry=False))
                except Exception as e_fallback:
                    logger.error(
                        "Failed to initialize ChromaDB PersistentClient (fallback after HttpClient error) at %s: %s", self.db_path, e_fallback, exc_info=True)
                    raise
        else:
            logger.info(
                "CHROMA_DB_HOST or CHROMA_DB_PORT not set. Using PersistentClient with path: %s", self.db_path)
            try:
                self.client = chromadb.PersistentClient(
                    path=self.db_path, settings=Settings(anonymized_telemetry=False))
            except Exception as e:
                logger.error(
                    "Failed to initialize ChromaDB PersistentClient at %s: %s", self.db_path, e, exc_info=True)
                raise

        # Ensure client is set before proceeding
        if not hasattr(self, 'client') or self.client is None:
            # This case should ideally be prevented by the raise statements above,
            # but as a safeguard:
            logger.critical(
                "ChromaDB client could not be initialized. Aborting initialization.")
            # Depending on application structure, might want to raise a specific error here
            # or ensure that later code handles a None client gracefully (though raising is better).
            raise ConnectionError("Failed to initialize any ChromaDB client.")

        class GeminiChromaEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, gem_client: GeminiClient):
                self.gem_client = gem_client

            def __call__(self, input_texts: List[str]) -> List[List[float]]:
                raw_embeddings = self.gem_client.get_embeddings_sync(
                    input_texts)
                valid_embeddings: List[List[float]] = []
                # TODO: Make embedding_dim configurable or fetch from model info
                # For "models/embedding-001", the dimension is 768.
                embedding_dim = 768

                for i, emb in enumerate(raw_embeddings):
                    if emb is None:
                        logger.warning(
                            "Sync embedding for input text at index %d ('%s...') was None. Using zero vector.",
                            i, input_texts[i][:30])
                        valid_embeddings.append([0.0] * embedding_dim)
                    else:
                        if len(emb) != embedding_dim:
                            logger.error(
                                "Embedding for input text at index %d has incorrect dimension %d, expected %d. Using zero vector.",
                                i, len(emb), embedding_dim)
                            valid_embeddings.append([0.0] * embedding_dim)
                        else:
                            valid_embeddings.append(emb)

                if not valid_embeddings and input_texts:  # Should not happen if we use zero vectors
                    logger.error(
                        "No valid embeddings generated for any input texts, even with zero vector fallback.")
                return valid_embeddings

        self.embedding_function = GeminiChromaEmbeddingFunction(
            self.gemini_client)

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(
                "ChromaDBManager initialized. Collection: '%s'. Path: '%s'. Client: %s. Count: %d",
                self.collection_name, self.db_path, type(self.client).__name__, self.collection.count())
        except Exception as e:
            logger.error(
                "Error getting or creating Chroma collection '%s' with client %s: %s",
                self.collection_name, type(self.client).__name__, e, exc_info=True)
            raise

    async def _run_collection_method(self, method_name: str, *pos_args, **kw_args) -> Any:
        """
        Helper to run a synchronous method of self.collection in a thread executor.
        It correctly uses functools.partial to bind all arguments.
        """
        loop = asyncio.get_running_loop()
        method_to_call = getattr(self.collection, method_name)

        # Create a partial function that has all arguments (positional and keyword) bound to it.
        # This partial function will then be called by run_in_executor without any additional arguments.
        func_with_bound_args = functools.partial(
            method_to_call, *pos_args, **kw_args)

        # DEBUG: Print what's being prepared for the executor
        # logger.debug(f"Executing in thread: {method_name} with pos_args={pos_args}, kw_args={kw_args}")
        # logger.debug(f"Partial function details: {func_with_bound_args.func}, {func_with_bound_args.args}, {func_with_bound_args.keywords}")

        return await loop.run_in_executor(None, func_with_bound_args)

    def generate_id_from_text_and_source(self, text: str, source: str) -> str:
        return hashlib.md5((text + source).encode('utf-8')).hexdigest()[:16]

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return

        start_time = asyncio.get_event_loop().time()
        try:
            # Call the helper, passing arguments as keyword arguments for `collection.add`
            await self._run_collection_method(
                'add',  # method_name
                # No positional arguments for collection.add, all are keyword.
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                "Added/updated %d documents to collection '%s' in %.2fs.",
                len(documents), self.collection_name, duration)
        except Exception as e:
            logger.error(
                "Error adding documents to ChromaDB: %s", e, exc_info=True)

    async def query_collection(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided, returning empty list.")
            return []

        start_time = asyncio.get_event_loop().time()
        try:
            current_count = await self.get_collection_count_async()
            if current_count == 0:
                logger.warning(
                    "Querying empty collection '%s'. Returning no results.", self.collection_name)
                return []

            actual_n_results = min(n_results, current_count)
            if actual_n_results <= 0:
                # ensure n_results is positive if querying
                actual_n_results = 1 if current_count > 0 else 0

            if actual_n_results == 0:  # if collection is empty or n_results forced to 0
                logger.info(
                    "Query for '%.50s...' resulted in 0 n_results. Returning empty list.", query_text)
                return []

            results = await self._run_collection_method(
                'query',  # method_name
                # No positional arguments for collection.query, all are keyword.
                query_texts=[query_text],
                n_results=actual_n_results,
                include=['documents', 'metadatas', 'distances']
            )
            duration = asyncio.get_event_loop().time() - start_time

            formatted_results = []
            # ChromaDB query results structure: results['ids'] is list of lists, etc.
            # Check if the inner list is not None
            if results and results.get('ids') and results['ids'][0] is not None:
                for i in range(len(results['ids'][0])):
                    res = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i] if results.get('documents') and results['documents'][0] else None,
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else None,
                        'distance': results['distances'][0][i] if results.get('distances') and results['distances'][0] else None,
                    }
                    formatted_results.append(res)

            logger.info(
                "Query '%.50s...' returned %d results in %.2fs.",
                query_text, len(formatted_results), duration)
            return formatted_results
        except Exception as e:
            logger.error(
                "Error querying ChromaDB collection: %s", e, exc_info=True)
            return []

    async def get_collection_count_async(self) -> int:
        try:
            # collection.count() takes no arguments
            return await self._run_collection_method('count')
        except Exception as e:
            logger.error("Error getting collection count: %s",
                         e, exc_info=True)
            return 0

    async def clear_collection_async(self):
        logger.warning(
            "Clearing collection '%s'. This will delete and recreate it.", self.collection_name)
        try:
            # These client operations are synchronous and not part of the _run_collection_method helper
            await asyncio.to_thread(self.client.delete_collection, name=self.collection_name)

            new_collection_instance = await asyncio.to_thread(
                self.client.get_or_create_collection,
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.collection = new_collection_instance

            new_count = await self.get_collection_count_async()
            logger.info(
                "Collection '%s' cleared and recreated. New count: %d", self.collection_name, new_count)
        except Exception as e:
            logger.error(
                "Error clearing collection '%s': %s", self.collection_name, e, exc_info=True)
