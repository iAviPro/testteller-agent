import asyncio
import logging
import os

import typer
from typing_extensions import Annotated

from .generator_agent.agent import TestTellerRagAgent
from .config import settings
from .core.constants import (
    DEFAULT_OUTPUT_FILE, DEFAULT_COLLECTION_NAME, SUPPORTED_LLM_PROVIDERS,
    DEFAULT_CHROMA_PERSIST_DIRECTORY
)
from pathlib import Path
# Import config modules inside function to avoid circular imports
from .core.utils.helpers import setup_logging
from .core.utils.loader import with_spinner
from ._version import __version__
from .core.utils.exceptions import EmbeddingGenerationError

# Import automation command functionality
try:
    from testteller.automator_agent.parser import MarkdownTestCaseParser
    from testteller.automator_agent.generators import PythonTestGenerator, JavaScriptTestGenerator, JavaTestGenerator, TypeScriptTestGenerator
    from testteller.core.data_ingestion.unified_document_parser import UnifiedDocumentParser, DocumentType
    from testteller.core.constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS
    HAS_TESTWRITER = True
except ImportError:
    HAS_TESTWRITER = False


setup_logging()
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Callback for version option."""
    if value:
        print(f"TestTeller Agent version: {__version__}")
        raise typer.Exit()


app = typer.Typer(
    help="TestTeller: Complete AI Test Agent for Generation and Automation. Configure the agent via .env file.",
    context_settings={"help_option_names": ["--help", "-h"]})



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


async def ingest_docs_async(path: str, collection_name: str, enhanced: bool = True, chunk_size: int = 1000):
    agent = _get_agent(collection_name)

    async def _ingest_task():
        await agent.ingest_documents_from_path(path, enhanced_parsing=enhanced, chunk_size=chunk_size)
        return await agent.get_ingested_data_count()

    ingestion_mode = "enhanced" if enhanced else "basic"
    spinner_text = f"Ingesting documents from '{path}' ({ingestion_mode} mode)"
    count = await with_spinner(_ingest_task(), spinner_text)
    
    print(f"Successfully ingested documents. Collection '{collection_name}' now contains {count} items.")
    if enhanced:
        print(f"💡 Enhanced parsing enabled: Documents chunked ({chunk_size} chars) with metadata extraction")


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
    path: Annotated[str, typer.Argument(help="Path to a document file or directory (supports .md, .txt, .pdf, .docx, .xlsx).")],
    collection_name: Annotated[str, typer.Option(
        "--collection-name", "-c", help="ChromaDB collection name.")] = None,
    enhanced: Annotated[bool, typer.Option(
        "--enhanced", "-e", help="Use enhanced parsing with chunking and metadata extraction")] = True,
    chunk_size: Annotated[int, typer.Option(
        "--chunk-size", "-s", help="Text chunk size for better retrieval (100-5000)")] = 1000
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

    # Validate chunk_size
    if not (100 <= chunk_size <= 5000):
        print("❌ Error: chunk-size must be between 100 and 5000 characters")
        raise typer.Exit(code=1)

    # Show ingestion mode
    mode = "enhanced" if enhanced else "basic"
    print(f"\n📄 Document Ingestion ({mode} mode)")
    if enhanced:
        print(f"  • Chunk size: {chunk_size} characters")
        print(f"  • Metadata extraction: enabled")
        print(f"  • Supported formats: .md, .txt, .pdf, .docx, .xlsx")
    else:
        print(f"  • Basic parsing mode")

    try:
        asyncio.run(ingest_docs_async(path, collection_name, enhanced, chunk_size))
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
        "--collection-name", "-c", help="ChromaDB collection name.")] = None,
    no_cleanup_github: Annotated[bool, typer.Option(
        "--no-cleanup-github", "-nc", help="Do not delete cloned GitHub repo after ingestion (no effect for local folders).")] = False
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
        "--collection-name", "-c", help="ChromaDB collection name.")] = None,
    num_retrieved: Annotated[int, typer.Option(
        "--num-retrieved", "-n", min=0, max=20, help="Number of docs for context.")] = 5,
    output_file: Annotated[str, typer.Option(
        "--output-file", "-o", help=f"Optional: Save test cases to this file. If not provided, uses OUTPUT_FILE_PATH from .env or defaults to {DEFAULT_OUTPUT_FILE}")] = None
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
        "--collection-name", "-c", help="ChromaDB collection name.")] = None
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
        "--collection-name", "-c", help="ChromaDB collection to clear.")] = None,
    force: Annotated[bool, typer.Option(
        "--force", "-f", help="Force clear without confirmation.")] = False
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
def configure(
    provider: Annotated[str, typer.Option(
        "--provider", "-p", help="Quick setup for specific provider (gemini, openai, claude, llama)")] = None,
    testwriter: Annotated[bool, typer.Option(
        "--testwriter", "-tw", help="Configure TestWriter automation settings only")] = False
):
    """Interactive configuration wizard to set up TestTeller."""
    from testteller.core.config import ConfigurationWizard, run_provider_only_setup, run_automation_only_setup
    from testteller.core.config import UIMode
    
    env_path = Path.cwd() / ".env"
    
    try:
        # Handle different configuration modes
        if testwriter:
            # Configure TestWriter (Automation) settings only
            success = run_automation_only_setup(UIMode.CLI)
            if not success:
                print("❌ TestWriter configuration failed.")
                raise typer.Exit(code=1)
            return
            
        elif provider:
            # Quick setup for specific provider
            if provider not in SUPPORTED_LLM_PROVIDERS:
                print(f"❌ Unsupported provider: {provider}")
                print(f"Supported providers: {', '.join(SUPPORTED_LLM_PROVIDERS)}")
                raise typer.Exit(code=1)
            
            success = run_provider_only_setup(provider, UIMode.CLI)
            if not success:
                print(f"❌ {provider.title()} configuration failed.")
                raise typer.Exit(code=1)
            return
        
        # Full configuration wizard
        wizard = ConfigurationWizard(UIMode.CLI)
        success = wizard.run(env_path)
        
        if success:
            print("\n🚀 TestTeller is now ready to use!")
            print("\n📚 Next steps:")
            print("  testteller --help                    # View all commands")
            print("  testteller ingest-docs <path>        # Add documents")
            print("  testteller ingest-code <repo_url>    # Add code") 
            print("  testteller generate \"<query>\"        # Generate tests")
            if HAS_TESTWRITER:
                print("  testteller automate test-cases.md    # Generate automation code")
        else:
            print("❌ Configuration failed.")
            raise typer.Exit(code=1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Configuration cancelled by user.")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error("Configuration wizard failed: %s", e, exc_info=True)
        print(f"❌ Configuration failed: {e}")
        raise typer.Exit(code=1)


# TestWriter automation command (if available)
if HAS_TESTWRITER:
    
    def get_generator(language: str, framework: str, output_dir: Path):
        """Get the appropriate generator based on language and framework."""
        if language == 'python':
            return PythonTestGenerator(framework, output_dir)
        elif language == 'javascript':
            return JavaScriptTestGenerator(framework, output_dir)
        elif language == 'typescript':
            return TypeScriptTestGenerator(framework, output_dir)
        elif language == 'java':
            return JavaTestGenerator(framework, output_dir)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def validate_framework(language: str, framework: str) -> bool:
        """Validate that the framework is supported for the language."""
        return framework in SUPPORTED_FRAMEWORKS.get(language, [])

    def interactive_select_tests(test_cases):
        """Interactive test case selection."""
        print("\n📋 Available test cases:")
        for i, tc in enumerate(test_cases, 1):
            print(f"{i:3d}. [{tc.id}] {tc.objective[:60]}...")
        
        print("\nSelect test cases to automate:")
        print("  • Enter numbers separated by commas (e.g., 1,3,5)")
        print("  • Enter ranges (e.g., 1-5)")
        print("  • Enter 'all' to select all tests")
        print("  • Enter 'none' to cancel")
        
        selection = typer.prompt("\nYour selection").strip().lower()
        
        if selection == 'none':
            return []
        elif selection == 'all':
            return test_cases
        else:
            selected_indices = parse_selection(selection, len(test_cases))
            return [test_cases[i-1] for i in selected_indices]

    def parse_selection(selection: str, max_index: int) -> list:
        """Parse user selection string into list of indices."""
        indices = set()
        
        for part in selection.split(','):
            part = part.strip()
            if '-' in part:
                # Range
                try:
                    start, end = map(int, part.split('-'))
                    indices.update(range(max(1, start), min(max_index + 1, end + 1)))
                except ValueError:
                    print(f"⚠️  Invalid range: {part}")
            else:
                # Single number
                try:
                    num = int(part)
                    if 1 <= num <= max_index:
                        indices.add(num)
                    else:
                        print(f"⚠️  Index out of range: {num}")
                except ValueError:
                    print(f"⚠️  Invalid number: {part}")
        
        return sorted(indices)

    def print_next_steps(language: str, framework: str, output_dir: Path):
        """Print next steps based on language and framework."""
        print("\n📚 Next steps:")
        
        if language == 'python':
            print(f"  1. cd {output_dir}")
            print("  2. pip install -r requirements.txt")
            if framework == 'pytest':
                print("  3. pytest")
            else:
                print("  3. python -m unittest discover")
        
        elif language == 'javascript':
            print(f"  1. cd {output_dir}")
            print("  2. npm install")
            print("  3. npm test")
        
        elif language == 'java':
            print(f"  1. cd {output_dir}")
            print("  2. mvn clean install")
            print("  3. mvn test")
        
        print("\n💡 Tips:")
        print("  • Review and customize the generated tests")
        print("  • Update test data and assertions")
        print("  • Configure test environment settings")
        print("  • Add missing test implementations marked with TODO")

    @app.command()
    def automate(
        input_file: Annotated[str, typer.Argument(help="Path to test cases file (supports .md, .txt, .pdf, .docx, .xlsx)")],
        language: Annotated[str, typer.Option(
            "--language", "-l", help="Programming language for test automation")] = None,
        framework: Annotated[str, typer.Option(
            "--framework", "-F", help="Test framework to use")] = None,
        output_dir: Annotated[str, typer.Option(
            "--output-dir", "-o", help="Output directory for generated tests")] = "./generated_tests",
        interactive: Annotated[bool, typer.Option(
            "--interactive", "-i", help="Interactive mode to select test cases")] = False,
        enhance: Annotated[bool, typer.Option(
            "--enhance", "-E", help="Use LLM to enhance generated test code")] = False,
        llm_provider: Annotated[str, typer.Option(
            "--llm-provider", "-p", help="LLM provider for enhancement (gemini, openai, claude, llama)")] = None
    ):
        """Generate automation code from TestTeller test cases."""
        
        # Validate input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            print(f"❌ Error: Test cases file '{input_file}' not found.")
            raise typer.Exit(code=1)
        
        # Interactive language/framework selection if not provided
        if not language:
            print("\n🔧 Test Automation Configuration")
            print("================================")
            print("\nSelect programming language:")
            languages = list(SUPPORTED_FRAMEWORKS.keys())
            for i, lang in enumerate(languages, 1):
                print(f"  {i}. {lang}")
            
            while True:
                try:
                    choice = typer.prompt("\nEnter number", type=int)
                    if 1 <= choice <= len(languages):
                        language = languages[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except (ValueError, typer.Abort):
                    print("Invalid input. Please enter a number.")
        
        if not framework:
            frameworks = SUPPORTED_FRAMEWORKS[language]
            print(f"\nSelect test framework for {language}:")
            for i, fw in enumerate(frameworks, 1):
                print(f"  {i}. {fw}")
            
            while True:
                try:
                    choice = typer.prompt("\nEnter number", type=int)
                    if 1 <= choice <= len(frameworks):
                        framework = frameworks[choice - 1]
                        break
                    print("Invalid choice. Please try again.")
                except (ValueError, typer.Abort):
                    print("Invalid input. Please enter a number.")
        
        # Validate framework
        if not validate_framework(language, framework):
            print(f"❌ Error: Framework '{framework}' is not supported for language '{language}'.")
            print(f"Supported frameworks for {language}: {', '.join(SUPPORTED_FRAMEWORKS[language])}")
            raise typer.Exit(code=1)
        
        print(f"\n✅ Configuration:")
        print(f"  • Language: {language}")
        print(f"  • Framework: {framework}")
        print(f"  • Output: {output_dir}")
        
        # LLM Enhancement Configuration (if not already specified)
        if not enhance:
            try:
                from testteller.automator_agent.llm_enhancer import is_llm_enhancement_available
                
                if is_llm_enhancement_available():
                    print(f"\n🤖 LLM Enhancement Available")
                    print("=============================")
                    print("TestTeller can use AI to enhance the generated test code with:")
                    print("  • Improved error handling and edge cases")
                    print("  • Better assertions and validations")
                    print("  • Enhanced code quality and best practices")
                    print("  • Framework-specific optimizations")
                    
                    enhance = typer.confirm("\nWould you like to enable LLM enhancement?", default=False)
                    
                    if enhance:
                        if not llm_provider:
                            print("\nAvailable LLM providers: gemini, openai, claude, llama")
                            llm_provider = typer.prompt(
                                "Select LLM provider (leave empty for default)", 
                                default="", 
                                show_default=False
                            ).strip()
                            llm_provider = llm_provider if llm_provider else None
                        
                        print(f"✅ LLM enhancement enabled" + (f" with {llm_provider}" if llm_provider else ""))
                    else:
                        print("⏭️  LLM enhancement disabled")
                else:
                    print(f"\n💡 Tip: Configure TestTeller with an LLM provider to enable AI-powered test enhancement")
            except ImportError:
                pass  # LLM enhancement not available
            except Exception as e:
                logger.warning(f"Failed to check LLM availability: {e}")
        
        try:
            # Parse test cases using unified parser
            print(f"\n📖 Parsing test cases from: {input_file}")
            file_extension = input_path.suffix.lower()
            
            if file_extension not in ['.md', '.txt', '.pdf', '.docx', '.xlsx']:
                print(f"❌ Unsupported file format: {file_extension}")
                print("Supported formats: .md, .txt, .pdf, .docx, .xlsx")
                raise typer.Exit(code=1)
            
            # Use unified parser for all formats
            print(f"  📄 Detected format: {file_extension}")
            unified_parser = UnifiedDocumentParser()
            
            # Parse document for automation
            parsed_doc = asyncio.run(unified_parser.parse_for_automation(input_path))
            
            # Extract test cases
            test_cases = parsed_doc.test_cases
            
            # If no structured test cases found, try to use content for context
            if not test_cases:
                print(f"  📝 No structured test cases found. Document type: {parsed_doc.metadata.document_type.value}")
                
                if parsed_doc.metadata.document_type in [DocumentType.TEST_CASES, DocumentType.REQUIREMENTS]:
                    # For markdown files, try the legacy parser as fallback
                    if file_extension == '.md':
                        print("  🔄 Trying markdown-specific parser...")
                        md_parser = MarkdownTestCaseParser()
                        test_cases = md_parser.parse_file(input_path)
                    
                    if not test_cases:
                        print("❌ No test cases found in the file.")
                        print("💡 Tip: Ensure the file contains structured test cases in the expected format.")
                        print("   For supported formats, see: https://github.com/testteller/docs")
                        raise typer.Exit(code=1)
                else:
                    print("❌ This document doesn't appear to contain test cases.")
                    print(f"   Detected type: {parsed_doc.metadata.document_type.value}")
                    print("💡 Try using a file that contains structured test cases.")
                    raise typer.Exit(code=1)
            
            print(f"✅ Found {len(test_cases)} test cases")
            
            # Show document info
            if parsed_doc.metadata.title:
                print(f"  📋 Document: {parsed_doc.metadata.title}")
            if parsed_doc.metadata.sections:
                print(f"  📑 Sections: {len(parsed_doc.metadata.sections)}")
            print(f"  📊 Content: {parsed_doc.metadata.word_count} words, {parsed_doc.metadata.character_count} characters")
            
            # Interactive selection if requested
            if interactive:
                selected_cases = interactive_select_tests(test_cases)
                if not selected_cases:
                    print("❌ No test cases selected.")
                    raise typer.Exit(code=1)
                test_cases = selected_cases
            
            # Generate automation code
            print(f"\n🚀 Generating {language} automation code...")
            output_path = Path(output_dir)
            generator = get_generator(language, framework, output_path)
            
            generated_files = generator.generate(test_cases)
            
            # LLM Enhancement (optional)
            if enhance:
                print(f"\n🤖 Enhancing generated code with LLM...")
                try:
                    from testteller.automator_agent.llm_enhancer import create_enhancer, is_llm_enhancement_available
                    
                    if not is_llm_enhancement_available():
                        print("⚠️  LLM enhancement not available. Ensure LLM is configured in TestTeller.")
                        print("   Generated code will be used without enhancement.")
                    else:
                        enhancer = create_enhancer(provider=llm_provider)
                        if enhancer.is_available():
                            enhanced_files = enhancer.enhance_generated_tests(
                                generated_files, language, framework
                            )
                            
                            # Count enhanced files
                            enhanced_count = sum(1 for filename, content in enhanced_files.items() 
                                               if content != generated_files.get(filename, ""))
                            
                            if enhanced_count > 0:
                                generated_files = enhanced_files
                                print(f"✅ Enhanced {enhanced_count} test files using {enhancer.provider}")
                                
                                # Show optimization suggestions
                                suggestions = enhancer.get_optimization_suggestions(language, framework)
                                if suggestions:
                                    print(f"\n💡 Optimization suggestions for {language} + {framework}:")
                                    for i, suggestion in enumerate(suggestions[:5], 1):  # Show top 5
                                        print(f"  {i}. {suggestion}")
                            else:
                                print("⚠️  No files were enhanced. Using original generated code.")
                        else:
                            print("⚠️  LLM enhancer initialization failed. Using original generated code.")
                            
                except ImportError:
                    print("⚠️  LLM enhancement dependencies not available. Using original generated code.")
                except Exception as e:
                    logger.warning(f"LLM enhancement failed: {e}")
                    print(f"⚠️  LLM enhancement failed: {e}")
                    print("   Using original generated code.")
            
            # Write files
            print(f"\n📝 Writing generated files to: {output_dir}")
            generator.write_files(generated_files)
            
            # Summary
            print(f"\n✅ Successfully generated {len(generated_files)} files:")
            for file_name in generated_files:
                print(f"  • {output_path / file_name}")
            
            # Next steps
            print_next_steps(language, framework, output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate automation code: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
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
    """TestTeller: Complete AI Test Agent for Generation and Automation. Configure the agent via your .env file."""
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
