import asyncio
import logging
import os

import typer
from typing_extensions import Annotated

from .agent import TestTellerRagAgent
from .config import settings
from .constants import (
    DEFAULT_LOG_LEVEL, DEFAULT_GEMINI_EMBEDDING_MODEL, DEFAULT_GEMINI_GENERATION_MODEL,
    DEFAULT_OUTPUT_FILE, DEFAULT_COLLECTION_NAME, DEFAULT_LLM_PROVIDER, SUPPORTED_LLM_PROVIDERS,
    DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_OPENAI_GENERATION_MODEL,
    DEFAULT_CLAUDE_GENERATION_MODEL, DEFAULT_CLAUDE_EMBEDDING_PROVIDER,
    DEFAULT_LLAMA_EMBEDDING_MODEL, DEFAULT_LLAMA_GENERATION_MODEL,
    DEFAULT_CHROMA_PERSIST_DIRECTORY
)
from .utils.helpers import setup_logging
from .utils.loader import with_spinner
from ._version import __version__
from .utils.exceptions import EmbeddingGenerationError


setup_logging()
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Callback for version option."""
    if value:
        print(f"TestTeller RAG Agent version: {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="TestTeller: RAG Agent for AI Test Case Generation. Configure the agent via .env file.")

# Default .env template with descriptions
ENV_TEMPLATE = {
    "LLM_PROVIDER": {
        "value": DEFAULT_LLM_PROVIDER,
        "required": False,
        "description": f"LLM provider to use ({', '.join(SUPPORTED_LLM_PROVIDERS)})",
        "options": SUPPORTED_LLM_PROVIDERS
    },
    "GOOGLE_API_KEY": {
        "value": "",
        "required": False,
        "description": "Your Google Gemini API key (required for Gemini)",
        "conditional": "gemini"
    },
    "OPENAI_API_KEY": {
        "value": "",
        "required": False,
        "description": "Your OpenAI API key (required for OpenAI)",
        "conditional": "openai"
    },
    "CLAUDE_API_KEY": {
        "value": "",
        "required": False,
        "description": "Your Anthropic Claude API key (required for Claude)",
        "conditional": "claude"
    },
    "CLAUDE_EMBEDDING_PROVIDER": {
        "value": DEFAULT_CLAUDE_EMBEDDING_PROVIDER,
        "required": False,
        "description": "Embedding provider for Claude (google, openai)",
        "conditional": "claude",
        "options": ["google", "openai"]
    },
    "GEMINI_EMBEDDING_MODEL": {
        "value": DEFAULT_GEMINI_EMBEDDING_MODEL,
        "required": False,
        "description": f"Gemini embedding model (optional, default: {DEFAULT_GEMINI_EMBEDDING_MODEL})",
        "conditional": "gemini"
    },
    "GEMINI_GENERATION_MODEL": {
        "value": DEFAULT_GEMINI_GENERATION_MODEL,
        "required": False,
        "description": f"Gemini generation model (optional, default: {DEFAULT_GEMINI_GENERATION_MODEL})",
        "conditional": "gemini"
    },
    "OPENAI_EMBEDDING_MODEL": {
        "value": DEFAULT_OPENAI_EMBEDDING_MODEL,
        "required": False,
        "description": f"OpenAI embedding model (optional, default: {DEFAULT_OPENAI_EMBEDDING_MODEL})",
        "conditional": "openai"
    },
    "OPENAI_GENERATION_MODEL": {
        "value": DEFAULT_OPENAI_GENERATION_MODEL,
        "required": False,
        "description": f"OpenAI generation model (optional, default: {DEFAULT_OPENAI_GENERATION_MODEL})",
        "conditional": "openai"
    },
    "CLAUDE_GENERATION_MODEL": {
        "value": DEFAULT_CLAUDE_GENERATION_MODEL,
        "required": False,
        "description": f"Claude generation model (optional, default: {DEFAULT_CLAUDE_GENERATION_MODEL})",
        "conditional": "claude"
    },
    "LLAMA_EMBEDDING_MODEL": {
        "value": DEFAULT_LLAMA_EMBEDDING_MODEL,
        "required": False,
        "description": f"Llama embedding model (optional, default: {DEFAULT_LLAMA_EMBEDDING_MODEL})",
        "conditional": "llama"
    },
    "LLAMA_GENERATION_MODEL": {
        "value": DEFAULT_LLAMA_GENERATION_MODEL,
        "required": False,
        "description": f"Llama generation model (optional, default: {DEFAULT_LLAMA_GENERATION_MODEL})",
        "conditional": "llama"
    },
    "GITHUB_TOKEN": {
        "value": "",
        "required": False,
        "description": "GitHub Personal Access Token for private repos (optional)"
    },
    "LOG_LEVEL": {
        "value": DEFAULT_LOG_LEVEL,
        "required": False,
        "description": "Logging level (DEBUG, INFO, WARNING, ERROR)"
    },
    "CHROMA_DB_PATH": {
        "value": DEFAULT_CHROMA_PERSIST_DIRECTORY,
        "required": False,
        "description": "Path to ChromaDB persistent storage"
    },
    "DEFAULT_COLLECTION_NAME": {
        "value": DEFAULT_COLLECTION_NAME,
        "required": False,
        "description": "Default ChromaDB collection name"
    },
    "OUTPUT_FILE_PATH": {
        "value": DEFAULT_OUTPUT_FILE,
        "required": False,
        "description": "Default path to save generated test cases"
    }
}


def get_collection_name(provided_name: str | None = None) -> str:
    """
    Get the collection name to use, with the following priority:
    1. User-provided name
    2. Name from settings
    3. Default fallback name
    """
    if provided_name:
        return provided_name

    default_name = DEFAULT_COLLECTION_NAME

    try:
        if settings and settings.chromadb:
            settings_dict = settings.chromadb.__dict__
            if settings_dict.get('default_collection_name'):
                name = settings_dict['default_collection_name']
                logger.info(
                    "Using default collection name from settings: %s", name)
                return name
    except Exception as e:
        logger.warning("Failed to get collection name from settings: %s", e)

    logger.info("Using fallback default collection name: %s", default_name)
    return default_name


def check_settings():
    """Check if required settings are available and provide guidance if not."""
    if settings is None:
        env_path = os.path.join(os.getcwd(), '.env')
        print("\n⚠️  Configuration Error: Missing or invalid .env file")
        print("\nTo configure TestTeller, you have two options:")
        print("\n1. Run the configuration wizard:")
        print("   testteller configure")
        print("\n2. Manually create a .env file at:")
        print(f"   {env_path}")
        print("\nMinimum required configuration:")
        print('   GOOGLE_API_KEY="your-api-key-here"')
        print("\nFor more information about configuration, visit:")
        print("   https://github.com/yourusername/testteller#configuration")
        raise typer.Exit(code=1)
    return True


def _get_agent(collection_name: str) -> TestTellerRagAgent:
    check_settings()  # Ensure settings are available
    try:
        return TestTellerRagAgent(collection_name=collection_name)
    except Exception as e:
        logger.error(
            "Failed to initialize TestCaseAgent for collection '%s': %s", collection_name, e, exc_info=True)
        print(
            f"Error: Could not initialize agent. Check logs and GOOGLE_API_KEY. Details: {e}")
        raise typer.Exit(code=1)


async def ingest_docs_async(path: str, collection_name: str):
    agent = _get_agent(collection_name)

    async def _ingest_task():
        await agent.ingest_documents_from_path(path)
        return await agent.get_ingested_data_count()

    count = await with_spinner(_ingest_task(), f"Ingesting documents from '{path}'...")
    print(
        f"Successfully ingested documents. Collection '{collection_name}' now contains {count} items.")


async def ingest_code_async(source_path: str, collection_name: str, no_cleanup_github: bool):
    agent = _get_agent(collection_name)

    async def _ingest_task():
        await agent.ingest_code_from_source(source_path, cleanup_github_after=not no_cleanup_github)
        return await agent.get_ingested_data_count()

    count = await with_spinner(_ingest_task(), f"Ingesting code from '{source_path}'...")
    print(
        f"Successfully ingested code from '{source_path}'. Collection '{collection_name}' now contains {count} items.")


async def generate_async(query: str, collection_name: str, num_retrieved: int, output_file: str | None):
    agent = _get_agent(collection_name)

    current_count = await agent.get_ingested_data_count()
    if current_count == 0:
        print(
            f"Warning: Collection '{collection_name}' is empty. Generation will rely on LLM's general knowledge.")
        if not typer.confirm("Proceed anyway?", default=True):
            print("Generation aborted.")
            return

    async def _generate_task():
        return await agent.generate_test_cases(query, n_retrieved_docs=num_retrieved)

    test_cases = await with_spinner(_generate_task(), f"Generating test cases for query...")
    print("\n--- Generated Test Cases ---")
    print(test_cases)
    print("--- End of Test Cases ---\n")

    if output_file:
        if "Error:" in test_cases[:20]:
            logger.warning(
                "LLM generation resulted in an error, not saving to file: %s", test_cases)
            print(
                f"Warning: Test case generation seems to have failed. Not saving to {output_file}.")
        else:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(test_cases)
                print(f"Test cases saved to: {output_file}")
            except Exception as e:
                logger.error(
                    "Failed to save test cases to %s: %s", output_file, e, exc_info=True)
                print(
                    f"Error: Could not save test cases to {output_file}: {e}")


async def status_async(collection_name: str):
    """Check status of a collection asynchronously."""
    agent = _get_agent(collection_name)
    count = await agent.get_ingested_data_count()
    print(f"\nCollection '{collection_name}' contains {count} ingested items.")

    # Print ChromaDB connection info
    if agent.vector_store.use_remote:
        print(
            f"ChromaDB connection: Remote at {agent.vector_store.host}:{agent.vector_store.port}")
    else:
        print(f"ChromaDB persistent path: {agent.vector_store.db_path}")


async def clear_data_async(collection_name: str, force: bool):
    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to clear all data from collection '{collection_name}' and remove related cloned repositories?")
        if not confirm:
            print("Operation cancelled.")
            return False  # Return False to indicate cancellation

    agent = _get_agent(collection_name)

    async def _clear_task():
        await agent.clear_ingested_data()

    await with_spinner(_clear_task(), f"Clearing data from collection '{collection_name}'...")
    print(f"Successfully cleared data from collection '{collection_name}'.")
    return True  # Return True to indicate success


@app.command()
def ingest_docs(
    path: Annotated[str, typer.Argument(help="Path to a document file or a directory.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None
):
    """Ingests documents from a file or directory into a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Ingesting documents from '%s' into collection '%s'",
                path, collection_name)

    if not os.path.exists(path):
        logger.error(
            "Document source path does not exist or is not accessible: %s", path)
        print(
            f"Error: Document source path '{path}' not found or not accessible.")
        raise typer.Exit(code=1)

    try:
        asyncio.run(ingest_docs_async(path, collection_name))
    except EmbeddingGenerationError as e:
        logger.error(
            "CLI: Embedding generation failed during document ingestion. Error: %s", e, exc_info=True)
        print(f"\n❌ Embedding Generation Failed:")
        print(f"   {e}")
        print("\n💡 Potential Solutions:")
        print("   1. Verify your API key in the .env file is correct and has sufficient quota.")
        print("   2. For Claude, ensure CLAUDE_API_KEY and your selected embedding provider's API key are set.")
        print("   3. For Llama, ensure the Ollama service is running and accessible.")
        print("   4. Check your network connection and firewall settings.")
        print("\nRun 'testteller configure' to re-check your settings.")
        raise typer.Exit(code=1)
    except typer.Exit:
        # Re-raise typer.Exit exceptions to avoid catching them
        raise
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during document ingestion from '%s': %s", path, e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def ingest_code(
    source_path: Annotated[str, typer.Argument(help="URL of the GitHub repository OR path to a local code folder.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None,
    no_cleanup_github: Annotated[bool, typer.Option(
        help="Do not delete cloned GitHub repo after ingestion (no effect for local folders).")] = False
):
    """Ingests code from a GitHub repository or local folder into a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Ingesting code from '%s' into collection '%s'",
                source_path, collection_name)

    # For local paths, check if they exist
    if not source_path.startswith(('http://', 'https://', 'git@')) and not os.path.exists(source_path):
        logger.error(
            "Local source path does not exist or is not accessible: %s", source_path)
        print(
            f"Error: Local source path '{source_path}' not found or not accessible.")
        raise typer.Exit(code=1)

    try:
        asyncio.run(ingest_code_async(
            source_path, collection_name, no_cleanup_github))
    except EmbeddingGenerationError as e:
        logger.error(
            "CLI: Embedding generation failed during code ingestion. Error: %s", e, exc_info=True)
        print(f"\n❌ Embedding Generation Failed:")
        print(f"   {e}")
        print("\n💡 Potential Solutions:")
        print("   1. Verify your API key in the .env file is correct and has sufficient quota.")
        print("   2. For Claude, ensure CLAUDE_API_KEY and your selected embedding provider's API key are set.")
        print("   3. For Llama, ensure the Ollama service is running and accessible.")
        print("   4. Check your network connection and firewall settings.")
        print("\nRun 'testteller configure' to re-check your settings.")
        raise typer.Exit(code=1)
    except typer.Exit:
        # Re-raise typer.Exit exceptions to avoid catching them
        raise
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during code ingestion from '%s': %s", source_path, e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    query: Annotated[str, typer.Argument(help="Query for test case generation.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None,
    num_retrieved: Annotated[int, typer.Option(
        min=0, max=20, help="Number of docs for context.")] = 5,
    output_file: Annotated[str, typer.Option(
        help=f"Optional: Save test cases to this file. If not provided, uses OUTPUT_FILE_PATH from .env or defaults to {DEFAULT_OUTPUT_FILE}")] = None
):
    """Generates test cases based on query and knowledge base."""
    logger.info(
        "CLI: Generating test cases for query: '%s...', Collection: %s", query[:50], collection_name)

    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    # Determine output file path
    final_output_file = output_file
    if not final_output_file:
        try:
            if settings and settings.output:
                settings_dict = settings.output.__dict__
                if settings_dict.get('output_file_path'):
                    final_output_file = settings_dict['output_file_path']
                    logger.info(
                        "Using output file path from settings: %s", final_output_file)
        except Exception as e:
            logger.warning(
                "Failed to get output file path from settings: %s", e)

        if not final_output_file:
            final_output_file = DEFAULT_OUTPUT_FILE
            logger.info("Using default output file path: %s",
                        final_output_file)

    try:
        asyncio.run(generate_async(
            query, collection_name, num_retrieved, final_output_file))
    except typer.Exit:
        # Re-raise typer.Exit exceptions to avoid catching them
        raise
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during test case generation: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def status(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = None
):
    """Checks status of a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Checking status for collection: %s", collection_name)
    try:
        asyncio.run(status_async(collection_name))
    except typer.Exit:
        # Re-raise typer.Exit exceptions to avoid catching them
        raise
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during status check: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def clear_data(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection to clear.")] = None,
    force: Annotated[bool, typer.Option(
        help="Force clear without confirmation.")] = False
):
    """Clears ingested data from a collection."""
    # Get collection name from settings if not provided
    collection_name = get_collection_name(collection_name)

    logger.info("CLI: Clearing data for collection: %s", collection_name)
    try:
        result = asyncio.run(clear_data_async(collection_name, force))
        if result is False:
            # Operation was cancelled by user
            raise typer.Exit(code=0)
    except typer.Exit:
        # Re-raise typer.Exit exceptions to avoid catching them
        raise
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during data clearing: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def configure():
    """Interactive configuration wizard to set up TestTeller."""
    env_path = os.path.join(os.getcwd(), '.env')
    env_example_path = os.path.join(os.getcwd(), '.env.example')

    # Check if .env already exists
    if os.path.exists(env_path):
        overwrite = typer.confirm(
            "⚠️  A .env file already exists. Do you want to reconfigure it?", default=False)
        if not overwrite:
            print("Configuration cancelled.")
            raise typer.Exit()

    print("\n🔧 TestTeller Configuration Wizard")
    print("==================================")

    env_values = {}

    # First, handle LLM provider selection
    llm_provider_config = ENV_TEMPLATE["LLM_PROVIDER"]
    print(f"\n{llm_provider_config['description']}")
    print("Available providers:")
    for i, provider in enumerate(llm_provider_config["options"], 1):
        print(f"  {i}. {provider}")

    while True:
        try:
            choice = typer.prompt(
                "\nSelect LLM provider (enter number)", type=int)
            if 1 <= choice <= len(llm_provider_config["options"]):
                selected_provider = llm_provider_config["options"][choice - 1]
                env_values["LLM_PROVIDER"] = selected_provider
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except (ValueError, typer.Abort):
            print("Invalid input. Please enter a number.")

    print(f"\n✅ Selected LLM provider: {selected_provider}")

    # Special handling for Llama provider - ask for URL and Port separately
    if selected_provider == "llama":
        print("\n📝 Configuring Ollama connection:")
        print("   - No API key is required for local Llama models")
        print("   - Make sure Ollama is installed and running")
        print(
            f"   - Install required models: ollama pull {DEFAULT_LLAMA_EMBEDDING_MODEL} && ollama pull {DEFAULT_LLAMA_GENERATION_MODEL}")

        # Ask for URL first
        while True:
            try:
                url = typer.prompt(
                    "\nOllama server URL (optional, default: localhost)",
                    default="localhost",
                    show_default=False
                )
                if url:
                    break
            except typer.Abort:
                print("Configuration cancelled.")
                raise typer.Exit()

        # Ask for Port second
        while True:
            try:
                port = typer.prompt(
                    "\nOllama server Port (optional, default: 11434)",
                    default="11434",
                    show_default=False
                )
                if port:
                    try:
                        # Validate port is a number
                        int(port)
                        break
                    except ValueError:
                        print("Port must be a valid number.")
                        continue
            except typer.Abort:
                print("Configuration cancelled.")
                raise typer.Exit()

        # Form complete URL:PORT
        if url == "localhost":
            url = "http://localhost"
        elif not url.startswith(('http://', 'https://')):
            url = f"http://{url}"

        ollama_base_url = f"{url}:{port}"
        env_values["OLLAMA_BASE_URL"] = ollama_base_url
        print(f"\n✅ Ollama URL configured: {ollama_base_url}")

        # Ask for Llama models right after URL configuration
        print("\n🔧 Llama Model Configuration:")

        # Llama embedding model
        embedding_prompt = f"\nLlama embedding model (optional, default: {DEFAULT_LLAMA_EMBEDDING_MODEL})"
        try:
            embedding_model = typer.prompt(
                embedding_prompt,
                default=DEFAULT_LLAMA_EMBEDDING_MODEL,
                show_default=False
            )
            env_values["LLAMA_EMBEDDING_MODEL"] = embedding_model
        except typer.Abort:
            print("Configuration cancelled.")
            raise typer.Exit()

        # Llama generation model
        generation_prompt = f"\nLlama generation model (optional, default: {DEFAULT_LLAMA_GENERATION_MODEL})"
        try:
            generation_model = typer.prompt(
                generation_prompt,
                default=DEFAULT_LLAMA_GENERATION_MODEL,
                show_default=False
            )
            env_values["LLAMA_GENERATION_MODEL"] = generation_model
        except typer.Abort:
            print("Configuration cancelled.")
            raise typer.Exit()

    # Handle Claude configuration separately
    if selected_provider == "claude":
        print("\n🔧 Claude Configuration:")

        # Ask for Claude API key first
        while True:
            try:
                claude_api_key = typer.prompt(
                    "\nYour Anthropic Claude API key (required)",
                    hide_input=True
                )
                if claude_api_key:
                    env_values["CLAUDE_API_KEY"] = claude_api_key
                    break
                print("Claude API key is required. Please provide a value.")
            except typer.Abort:
                print("Configuration cancelled.")
                raise typer.Exit()

        # Ask for embedding provider
        print("\n📝 Embedding Provider Selection:")
        print("Claude needs an embedding provider for RAG functionality:")
        print("  1. google (Google Gemini - free, suitable for most use cases)")
        print("  2. openai (OpenAI - paid, high quality)")

        while True:
            try:
                choice = typer.prompt(
                    "\nSelect embedding provider (enter number)", type=int)
                if choice == 1:
                    embedding_provider = "google"
                    env_values["CLAUDE_EMBEDDING_PROVIDER"] = embedding_provider
                    break
                elif choice == 2:
                    embedding_provider = "openai"
                    env_values["CLAUDE_EMBEDDING_PROVIDER"] = embedding_provider
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except (ValueError, typer.Abort):
                print("Invalid input. Please enter a number.")

        # Ask for embedding provider's API key
        if embedding_provider == "google":
            while True:
                try:
                    google_api_key = typer.prompt(
                        "\nYour Google API key (required for Claude embeddings)",
                        hide_input=True
                    )
                    if google_api_key:
                        env_values["GOOGLE_API_KEY"] = google_api_key
                        break
                    print("Google API key is required. Please provide a value.")
                except typer.Abort:
                    print("Configuration cancelled.")
                    raise typer.Exit()
        elif embedding_provider == "openai":
            while True:
                try:
                    openai_api_key = typer.prompt(
                        "\nYour OpenAI API key (required for Claude embeddings)",
                        hide_input=True
                    )
                    if openai_api_key:
                        env_values["OPENAI_API_KEY"] = openai_api_key
                        break
                    print("OpenAI API key is required. Please provide a value.")
                except typer.Abort:
                    print("Configuration cancelled.")
                    raise typer.Exit()

        # Ask for Claude generation model
        try:
            claude_generation_model = typer.prompt(
                f"\nClaude generation model (optional, default: {DEFAULT_CLAUDE_GENERATION_MODEL})",
                default=DEFAULT_CLAUDE_GENERATION_MODEL,
                show_default=False
            )
            env_values["CLAUDE_GENERATION_MODEL"] = claude_generation_model
        except typer.Abort:
            print("Configuration cancelled.")
            raise typer.Exit()

        print(f"\n✅ Claude configuration complete!")
        print(f"   • Claude API key: configured")
        print(f"   • Embedding provider: {embedding_provider}")
        print(f"   • Generation model: {claude_generation_model}")

    # Collect values for each setting (for non-Claude providers)
    elif selected_provider != "claude":
        for key, config in ENV_TEMPLATE.items():
            if key == "LLM_PROVIDER":  # Already handled above
                continue

            description = config["description"]
            default = config["value"]
            required = config["required"]
            conditional = config.get("conditional")

            # Skip conditional fields if they don't match the selected provider
            if conditional and conditional != selected_provider:
                continue

            # For API keys, make them required if they match the selected provider
            # Keep model configurations optional (embedding/generation models)
            if conditional and conditional == selected_provider and ("API_KEY" in key or "TOKEN" in key):
                required = True

            # Format prompt based on whether the field is required
            prompt = f"\n{description}"
            if required:
                prompt += " (required)"
            elif default:
                prompt += f" (default: {default})"

            # Special handling for Llama provider
            if selected_provider == "llama" and key in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "CLAUDE_API_KEY", "LLAMA_EMBEDDING_MODEL", "LLAMA_GENERATION_MODEL"]:
                continue  # Skip API keys and model configs for Llama since they're handled separately

            # Get user input
            while True:
                try:
                    # Hide input for API keys to keep them secure
                    hide_input = "API_KEY" in key or "TOKEN" in key
                    value = typer.prompt(
                        prompt, default=default if not required else None, show_default=bool(default), hide_input=hide_input)
                    if value or not required:
                        break
                    print("This field is required. Please provide a value.")
                except typer.Abort:
                    print("Configuration cancelled.")
                    raise typer.Exit()

            if value:
                env_values[key] = value

    # Try to read additional non-critical configs from .env.example
    additional_configs = {}
    provider_specific_configs = {}

    if os.path.exists(env_example_path):
        try:
            with open(env_example_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        # Skip placeholder values
                        if 'your_' in value.lower() and '_here' in value.lower():
                            continue

                        # Only add if it's not already in env_values and not in ENV_TEMPLATE
                        if key not in env_values and key not in ENV_TEMPLATE:
                            # Categorize provider-specific vs general configs
                            if any(provider in key.lower() for provider in ['gemini', 'openai', 'claude', 'llama', 'ollama']):
                                provider_specific_configs[key] = value
                            else:
                                additional_configs[key] = value

            # Handle general additional configs
            if additional_configs:
                print(
                    f"\n📋 Found {len(additional_configs)} additional configuration(s) in .env.example:")
                for key, value in additional_configs.items():
                    print(f"  - {key}={value}")

                if typer.confirm("\nWould you like to include these additional configurations?", default=True):
                    env_values.update(additional_configs)
                    print("✅ Additional configurations included.")
                else:
                    print("⏭️  Additional configurations skipped.")

            # Handle provider-specific configs (auto-include relevant ones)
            relevant_provider_configs = {}
            for key, value in provider_specific_configs.items():
                key_lower = key.lower()
                if (selected_provider == "gemini" and "gemini" in key_lower) or \
                   (selected_provider == "openai" and "openai" in key_lower) or \
                   (selected_provider == "claude" and ("claude" in key_lower or
                    (env_values.get("CLAUDE_EMBEDDING_PROVIDER") == "openai" and "openai" in key_lower) or
                    (env_values.get("CLAUDE_EMBEDDING_PROVIDER") == "google" and "google" in key_lower))) or \
                   (selected_provider == "llama" and ("llama" in key_lower or "ollama" in key_lower)):
                    relevant_provider_configs[key] = value

            if relevant_provider_configs:
                print(
                    f"\n🔧 Found {len(relevant_provider_configs)} {selected_provider}-specific configuration(s):")
                for key, value in relevant_provider_configs.items():
                    print(f"  - {key}={value}")

                if typer.confirm(f"\nInclude {selected_provider}-specific configurations?", default=True):
                    env_values.update(relevant_provider_configs)
                    print(f"✅ {selected_provider} configurations included.")
                else:
                    print(f"⏭️  {selected_provider} configurations skipped.")

        except Exception as e:
            logger.warning("Could not read .env.example: %s", e)
            # Silently continue without additional configs

    # Write to .env file
    try:
        with open(env_path, 'w') as f:
            # Write critical configs first
            for key, config in ENV_TEMPLATE.items():
                if key in env_values:
                    f.write(f'{key}="{env_values[key]}"\n')

            # Write additional configs with a separator comment
            if any(key not in ENV_TEMPLATE for key in env_values):
                f.write('\n# Additional configurations\n')
                for key, value in env_values.items():
                    if key not in ENV_TEMPLATE:
                        f.write(f'{key}="{value}"\n')

        print("\n✅ Configuration complete!")
        print(f"Configuration saved to: {env_path}")
        print(f"LLM Provider: {selected_provider}")

        # Report what was copied from .env.example
        copied_from_example = []
        for key in env_values:
            if key not in ENV_TEMPLATE:
                copied_from_example.append(key)

        if copied_from_example:
            print(
                f"\n📋 Copied {len(copied_from_example)} additional configuration(s) from .env.example:")
            for key in copied_from_example:
                print(f"  - {key}={env_values[key]}")
            print(
                f"\n💡 You can modify these settings directly in {env_path} anytime:")
            for key in copied_from_example:
                print(f"   {key}")

        # Validate the configuration
        try:
            from testteller.llm.llm_manager import LLMManager
            is_valid, error_msg = LLMManager.validate_provider_config(
                selected_provider)
            if is_valid:
                print("\n✅ Configuration validated successfully!")
            else:
                print(f"\n⚠️  Configuration validation warning: {error_msg}")
        except Exception as e:
            print(f"\n⚠️  Could not validate configuration: {e}")

        print(f"\n🚀 Setup Summary:")
        print(f"  • LLM Provider: {selected_provider}")
        print(f"  • Configuration file: {env_path}")
        if copied_from_example:
            print(
                f"  • Additional settings: {len(copied_from_example)} copied from .env.example")
        print(f"  • Ready to use!")

        print("\n📚 Next steps:")
        print("  testteller --help                    # View all commands")
        print("  testteller ingest-docs <path>        # Add documents")
        print("  testteller ingest-code <repo_url>    # Add code")
        print("  testteller generate \"<query>\"        # Generate tests")

    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}")
        raise typer.Exit(code=1)


@app.callback()
def main(
    _: Annotated[bool, typer.Option(
        "--version", "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True
    )] = False
):
    """TestTeller: RAG Agent for AI Test Case Generation. Configure the agent via your .env file."""
    pass


def app_runner():
    """
    This function is the entry point for the CLI script defined in pyproject.toml.
    It ensures logging is set up and then runs the Typer application.
    """
    try:
        app()
    except Exception as e:
        logger.error("Unhandled error in CLI: %s", e, exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app_runner()
