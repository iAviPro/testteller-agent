# Moved import after potential env var set
from utils.retry_helpers import api_retry_async, api_retry_sync
import google.generativeai as genai
import os
from config import settings  # Ensure settings is imported
import logging
import asyncio
import functools

logger = logging.getLogger(__name__)

# Module-level configuration and debug prints removed.
# Configuration will now happen in GeminiClient.__init__

class GeminiClient:
    def __init__(self):
        try:
            # Configure genai here, ensuring it's done after mocks can be applied for tests
            # and only when a client is instantiated.
            api_key_value = settings.api_keys.google_api_key.get_secret_value()
            if not api_key_value: # Check if the key is empty after getting secret
                logger.error("Google API Key is not set or is empty.")
                raise ValueError("Google API Key is not set or is empty.")

            genai.configure(api_key=api_key_value)
            logger.info("Google Generative AI client configured successfully.")

            self.embedding_model_name = settings.gemini_model.gemini_embedding_model
            self.generation_model_name = settings.gemini_model.gemini_generation_model
            self.generation_model = genai.GenerativeModel(
                self.generation_model_name)
            logger.info(
                "Gemini client initialized with generation model: %s and embedding model: %s",
                self.generation_model_name, self.embedding_model_name)
        except ValueError as ve: # Catch specific ValueError for API key
            logger.error("Failed to initialize Gemini client due to API key issue: %s", ve, exc_info=True)
            raise
        except Exception as e: # Catch other potential errors during init
            logger.error(
                "Failed to initialize Gemini client with models: %s", e, exc_info=True)
            raise

    # ... (rest of GeminiClient methods - get_embedding_async, get_embedding_sync, etc. - REMAIN THE SAME as the last correct version)
    @api_retry_async
    async def get_embedding_async(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for embedding, returning None.")
            return None
        try:
            loop = asyncio.get_running_loop()
            func_to_run = functools.partial(
                genai.embed_content,
                model=self.embedding_model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            result = await loop.run_in_executor(None, func_to_run)
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    @api_retry_sync
    def get_embedding_sync(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for sync embedding, returning None.")
            return None
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(
                "Error generating sync embedding for text: '%s...': %s", text[:50], e, exc_info=True)
            return None

    async def get_embeddings_async(self, texts: list[str]) -> list[list[float] | None]:
        tasks = [self.get_embedding_async(text_chunk) for text_chunk in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        processed_embeddings = []
        for i, emb_or_exc in enumerate(embeddings):
            if isinstance(emb_or_exc, Exception):
                logger.error(
                    "Failed to get embedding for text chunk %d after retries: %s", i, emb_or_exc)
                processed_embeddings.append(None)
            else:
                processed_embeddings.append(emb_or_exc)
        return processed_embeddings

    def get_embeddings_sync(self, texts: list[str]) -> list[list[float] | None]:
        embeddings = []
        for i, text_chunk in enumerate(texts):
            emb = self.get_embedding_sync(text_chunk)
            embeddings.append(emb)
        return embeddings

    @api_retry_async
    async def generate_text_async(self, prompt: str, safety_settings=None, generation_config=None) -> str:
        try:
            if safety_settings is None:
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"},
                ]
            if generation_config is None:
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=8000, temperature=0.7)

            loop = asyncio.get_running_loop()

            func_to_run = functools.partial(
                self.generation_model.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            response = await loop.run_in_executor(None, func_to_run)

            if not response.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    msg = f"Content generation blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                    logger.error(msg)
                    return f"Error: {msg}"
                else:
                    logger.error(
                        "Content generation failed: No parts in response and no block reason provided.")
                    return "Error: Content generation failed for an unknown reason."
            return response.text
        except ValueError as ve:
            logger.error(
                "ValueError during text generation: %s", ve, exc_info=True)
            return f"Error: Configuration issue for text generation. {ve}"
        except Exception as e:
            logger.error("Error generating text: %s", e, exc_info=True)
            if "API key not valid" in str(e):
                return "Error: Invalid Google API Key."
            return f"Error: An unexpected error occurred during text generation. {e}"
