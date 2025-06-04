import logging
from typing import List, Dict, Any
import functools
import hashlib
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from config import settings
from llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


# Define GeminiChromaEmbeddingFunction at the module level
class GeminiChromaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, gem_client: GeminiClient):
        self.gem_client = gem_client

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        if not input_texts:
            return [] # Return empty list early if no input texts

        raw_embeddings = self.gem_client.get_embeddings_sync(input_texts)
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


class ChromaDBManager:
    def __init__(self, gemini_client: GeminiClient, collection_name: str = settings.chroma_db.default_collection_name):
        self.gemini_client = gemini_client
        self.collection_name = collection_name
        self.db_path = settings.chroma_db.chroma_db_path

        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
        except Exception as e:
            logger.error(
                "Failed to initialize ChromaDB PersistentClient at %s: %s", self.db_path, e, exc_info=True)
            raise

        self.embedding_function = GeminiChromaEmbeddingFunction(self.gemini_client)

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(
                "ChromaDBManager initialized. Collection: '%s' at path: '%s'. Count: %d",
                self.collection_name, self.db_path, self.collection.count())
        except Exception as e:
            logger.error(
                "Error getting or creating Chroma collection '%s': %s", self.collection_name, e, exc_info=True)
            raise

    async def _run_collection_method(self, method_name: str, *pos_args, **kw_args) -> Any:
        """
        Helper to run a synchronous method of self.collection in a thread executor.
        It correctly uses functools.partial to bind all arguments.
        """
        loop = asyncio.get_running_loop()
        method_to_call = getattr(self.collection, method_name)

        func_with_bound_args = functools.partial(
            method_to_call, *pos_args, **kw_args)

        return await loop.run_in_executor(None, func_with_bound_args)

    def generate_id_from_text_and_source(self, text: str, source: str) -> str:
        return hashlib.md5((text + source).encode('utf-8')).hexdigest()[:16]

    async def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return

        start_time = asyncio.get_event_loop().time()
        try:
            await self._run_collection_method(
                'add',
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

    async def query_collection(self, query_texts: List[str] | str, n_results: int = 5) -> List[Dict[str, Any]]:
        if isinstance(query_texts, list):
            if not query_texts:
                logger.warning("Empty query texts list provided, returning empty list.")
                return []
            current_query_text = query_texts[0]
            if len(query_texts) > 1:
                logger.warning("Multiple query texts provided, but only processing the first one: '%s'", current_query_text)
        else:
            current_query_text = query_texts

        if not current_query_text or not current_query_text.strip():
            logger.warning("Empty query text provided, returning empty list.")
            return []

        start_time = asyncio.get_event_loop().time()
        formatted_results = []
        try:
            current_count = await self.get_collection_count_async()
            if current_count == 0:
                logger.warning(
                    "Querying empty collection '%s'. Returning no results.", self.collection_name)
                return []

            actual_n_results = min(n_results, current_count)
            if actual_n_results <= 0:
                actual_n_results = 1 if current_count > 0 else 0

            if actual_n_results == 0:
                logger.info(
                    "Query for '%.50s...' resulted in 0 n_results. Returning empty list.", current_query_text)
                return []

            results = await self._run_collection_method(
                'query',
                query_texts=[current_query_text],
                n_results=actual_n_results,
                include=['documents', 'metadatas', 'distances']
            )

            if results and results.get('ids') and results['ids'] and results['ids'][0] is not None:
                expected_keys = ['ids', 'documents', 'metadatas', 'distances']
                valid_structure = True
                for key in expected_keys:
                    if not (results.get(key) and isinstance(results[key], list) and len(results[key]) > 0 and isinstance(results[key][0], list)):
                        logger.warning(f"Query results missing or malformed for key: {key}. Results: {results}")
                        valid_structure = False
                        break

                if valid_structure:
                    num_ids = len(results['ids'][0])
                    num_docs = len(results['documents'][0])
                    num_metadatas = len(results['metadatas'][0])
                    num_distances = len(results['distances'][0])

                    if not (num_ids == num_docs == num_metadatas == num_distances):
                        logger.warning(
                            "Query results for query '%s' had mismatched lengths. IDs: %d, Docs: %d, Metadatas: %d, Distances: %d. Skipping result processing.",
                            current_query_text, num_ids, num_docs, num_metadatas, num_distances
                        )
                    else:
                        for i in range(num_ids):
                            res = {
                                'id': results['ids'][0][i],
                                'document': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i],
                            }
                            formatted_results.append(res)
            elif results:
                 logger.warning(f"Query results were present but malformed (missing 'ids'[0] or it was None). Results: {results}")

        except Exception as e:
            logger.error("Error querying ChromaDB collection: %s", e, exc_info=True)
            # Fall through to return empty formatted_results

        duration = asyncio.get_event_loop().time() - start_time
        logger.info(
            "Query '%.50s...' returned %d results in %.2fs.",
            current_query_text, len(formatted_results), duration)
        return formatted_results

    async def get_collection_count_async(self) -> int:
        try:
            return await self._run_collection_method('count')
        except Exception as e:
            logger.error("Error getting collection count: %s", e, exc_info=True)
            return 0

    async def clear_collection_async(self):
        logger.warning(
            "Clearing collection '%s'. This will delete and recreate it.", self.collection_name)
        try:
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
