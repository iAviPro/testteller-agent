import logging
import os
import asyncio
from pathlib import Path
from typing_extensions import Annotated
import typer
from .utils.helpers import setup_logging
from .agent import TestTellerRagAgent
from .config import settings


setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="TestTeller: RAG Agent for AI Test Case Generation. Configure the agent via .env file.")


def _get_agent(collection_name: str) -> TestTellerRagAgent:
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
    await agent.ingest_documents_from_path(path)
    count = await agent.get_ingested_data_count()
    print(
        f"Successfully ingested documents. Collection '{collection_name}' now contains {count} items.")


async def ingest_code_async(source_path: str, collection_name: str, no_cleanup_github: bool):
    agent = _get_agent(collection_name)
    await agent.ingest_code_from_source(source_path, cleanup_github_after=not no_cleanup_github)
    count = await agent.get_ingested_data_count()
    print(
        f"Successfully ingested code from '{source_path}'. Collection '{collection_name}' now contains {count} items.")


async def generate_async(query: str, collection_name: str, num_retrieved: int, output_file: str | None):
    agent = _get_agent(collection_name)

    if await agent.get_ingested_data_count() == 0:
        print(
            f"Warning: Collection '{collection_name}' is empty. Generation will rely on LLM's general knowledge.")
        if not typer.confirm("Proceed anyway?", default=True):
            print("Generation aborted.")
            raise typer.Exit()

    test_cases = await agent.generate_test_cases(query, n_retrieved_docs=num_retrieved)
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
    agent = _get_agent(collection_name)
    count = await agent.get_ingested_data_count()
    print(f"Collection '{collection_name}' contains {count} ingested items.")
    print(f"ChromaDB persistent path: {agent.vector_store.db_path}")


async def clear_data_async(collection_name: str, force: bool):
    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to clear all data from collection '{collection_name}' and remove related cloned repositories?")
        if not confirm:
            print("Operation cancelled.")
            raise typer.Exit()

    agent = _get_agent(collection_name)
    await agent.clear_ingested_data()
    print(f"Successfully cleared data from collection '{collection_name}'.")


@app.command()
def ingest_docs(
    path: Annotated[str, typer.Argument(help="Path to a document file or a directory.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = settings.chroma_db.default_collection_name
):
    """Ingests documents into the knowledge base."""
    logger.info(
        "CLI: Ingesting documents from: %s, Collection: %s", path, collection_name)
    if not os.path.exists(path):
        logger.error("Path does not exist: %s", path)
        print(f"Error: Path does not exist: {path}")
        raise typer.Exit(code=1)
    try:
        asyncio.run(ingest_docs_async(path, collection_name))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during document ingestion: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def ingest_code(
    source_path: Annotated[str, typer.Argument(help="URL of the GitHub repository OR path to a local code folder.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = settings.chroma_db.default_collection_name,
    no_cleanup_github: Annotated[bool, typer.Option(
        help="Do not delete cloned GitHub repo after ingestion (no effect for local folders).")] = False
):
    """Clones GitHub repo or reads local folder, extracts code, and ingests it."""
    logger.info(
        "CLI: Ingesting code from source: %s, Collection: %s", source_path, collection_name)
    is_url_heuristic = "://" in source_path or source_path.startswith("git@")
    # Quick check for local path
    if not is_url_heuristic and not Path(source_path).exists():
        logger.error(
            "Local source path does not exist or is not accessible: %s", source_path)
        print(
            f"Error: Local source path '{source_path}' not found or not accessible.")
        raise typer.Exit(code=1)

    try:
        asyncio.run(ingest_code_async(
            source_path, collection_name, no_cleanup_github))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during code ingestion from '%s': %s", source_path, e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate(
    query: Annotated[str, typer.Argument(help="Query for test case generation.")],
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = settings.chroma_db.default_collection_name,
    num_retrieved: Annotated[int, typer.Option(
        min=0, max=20, help="Number of docs for context.")] = 5,
    output_file: Annotated[str, typer.Option(
        help="Optional: Save test cases to this file.")] = None
):
    """Generates test cases based on query and knowledge base."""
    logger.info(
        "CLI: Generating test cases for query: '%s...', Collection: %s", query[:50], collection_name)
    try:
        asyncio.run(generate_async(
            query, collection_name, num_retrieved, output_file))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during test case generation: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def status(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection name.")] = settings.chroma_db.default_collection_name
):
    """Checks status of a collection."""
    logger.info("CLI: Checking status for collection: %s", collection_name)
    try:
        asyncio.run(status_async(collection_name))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during status check: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def clear_data(
    collection_name: Annotated[str, typer.Option(
        help="ChromaDB collection to clear.")] = settings.chroma_db.default_collection_name,
    force: Annotated[bool, typer.Option(
        help="Force clear without confirmation.")] = False
):
    """Clears ingested data from a collection."""
    logger.info("CLI: Clearing data for collection: %s", collection_name)
    try:
        asyncio.run(clear_data_async(collection_name, force))
    except Exception as e:
        logger.error(
            "CLI: Unhandled error during data clearing: %s", e, exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


def app_runner():
    """
    This function is the entry point for the CLI script defined in pyproject.toml.
    It ensures logging is set up and then runs the Typer application.
    """
    # setup_logging() # Called at module level now
    app()


if __name__ == "__main__":
    # This allows running main.py directly for development/testing
    # (e.g., python -m testteller_rag_agent.main generate "query")
    app_runner()
