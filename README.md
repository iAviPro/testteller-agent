# TestTeller RAG Agent

**TestTeller RAG Agent** is a versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases. It leverages Google's Gemini LLM and ChromaDB as a vector store. The agent can process various input sources, including PRD documentation, API contracts, technical design documents (HLD/LLD), and code from GitHub repositories or local folders.

The agent aims to produce both:
1.  **Technical Test Cases**: Focusing on individual components, APIs, and system architecture.
2.  **User Journey Test Cases**: Driven by customer-backward scenarios and end-to-end flows.

## Features

*   **Multi-Source Ingestion**:
    *   Documents: `.docx`, `.pdf`, `.xlsx`, `.txt`, `.md`
    *   Code: Clones public/private GitHub repositories or reads from local folders (supports various programming languages via file extensions).
*   **RAG Pipeline**:
    *   Uses Google Gemini for generating embeddings and for text generation.
    *   Utilizes ChromaDB for efficient similarity search and retrieval of relevant context.
    *   Text chunking for effective processing of large documents and code files.
*   **Comprehensive Test Case Generation**:
    *   Generates both technical component-level and user-journey-driven test cases.
    *   Prompt-engineered to guide the LLM for specific test case formats and considerations.
*   **Command-Line Interface (CLI)**:
    *   User-friendly CLI built with Typer for all operations (ingestion, generation, status, clearing data).

## Project Structure

```
testteller-rag-agent/
├── main.py                 # CLI entry point
├── agent.py                # Core RAG Agent logic
├── config.py               # Configuration (Pydantic settings)
├── data_ingestion/
│   ├── __init__.py
│   ├── document_loader.py  # Handles .docx, .pdf, .xlsx, .txt
│   ├── code_loader.py      # Handles GitHub/local code loading
│   └── text_splitter.py    # Text chunking logic
├── vector_store/
│   ├── __init__.py
│   └── chromadb_manager.py # ChromaDB interactions
├── llm/
│   ├── __init__.py
│   └── gemini_client.py    # Gemini LLM and embedding interactions
├── prompts.py              # Prompt templates for test case generation
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # Logging setup
│   └── retry_utils.py      # Tenacity retry decorators
├── .env.example            # Example .env file
├── .env                    # Local environment configurations (Gitignored)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Prerequisites

*   Python 3.9+
*   Access to Google Gemini API (requires an API key from [Google AI Studio](https://aistudio.google.com/)).
*   (Optional) GitHub Personal Access Token (PAT) if you intend to clone private repositories. The token needs `repo` scope.

## Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd testteller_rag_agent
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your Google Gemini API key:
        ```env
        GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
        # Optional: For private GitHub repos
        # GITHUB_TOKEN="YOUR_GITHUB_PAT"

        # You can override other default settings from config.py here:
        # LOG_LEVEL="DEBUG" # For more verbose logging
        # CHROMA_DB_PATH="./my_vector_data"
        # DEFAULT_COLLECTION_NAME="my_project_kb"
        ```
    *   **Important**: Replace `"YOUR_GEMINI_API_KEY"` with your actual key.

## Usage (CLI)

The main interface to the agent is through `main.py`.

```bash
python main.py --help
```

### 1. Ingesting Data

You need to ingest relevant documents and code into a ChromaDB collection before generating test cases.

**Ingest Documents:**
*   From a directory (processes all supported files recursively):
    ```bash
    python main.py ingest-docs ./path/to/your/documents/ --collection-name project_alpha_docs
    ```
*   From a single document file:
    ```bash
    python main.py ingest-docs ./path/to/your/prd.pdf --collection-name project_alpha_docs
    ```

**Ingest Code:**
*   From a GitHub repository:
    ```bash
    python main.py ingest-code https://github.com/owner/repo.git --collection-name project_alpha_code
    ```
*   From a local code folder:
    ```bash
    python main.py ingest-code ./path/to/your/local_codebase/ --collection-name project_alpha_code
    ```
*   To prevent deletion of a cloned GitHub repository after ingestion (useful for debugging):
    ```bash
    python main.py ingest-code https://github.com/owner/repo.git --collection-name project_alpha_code --no-cleanup-github
    ```

*Note: You can use the same collection name for both documents and code, or separate them.*

### 2. Generating Test Cases

Once data is ingested, you can ask the agent to generate test cases.

```bash
python main.py generate "Generate test cases for the user login feature based on the PRD and API docs." --collection-name project_alpha_docs

# Specify number of retrieved context documents and output file
python main.py generate "Create API tests for the /users endpoint considering success and failure scenarios." \
    --collection-name project_alpha_code \
    --num-retrieved 7 \
    --output-file user_api_tests.md
```

### 3. Checking Collection Status

To see how many items are in a specific collection:

```bash
python main.py status --collection-name project_alpha_docs
```

### 4. Clearing Data

To remove all data from a collection and associated temporary files (like cloned repos):

```bash
# Will ask for confirmation
python main.py clear-data --collection-name project_alpha_docs

# Force clear without confirmation
python main.py clear-data --collection-name project_alpha_docs --force
```

## Configuration

Key configurations can be managed via:
*   The `.env` file for sensitive keys and common overrides.
*   The `config.py` file for default values and Pydantic settings schema.

Refer to `config.py` for all available settings (e.g., chunk size, model names, log level).

## Logging

*   Logs are output to the console.
*   Log format can be set to `text` (default) or `json` via the `LOG_FORMAT` environment variable or in `config.py`. JSON logs are recommended for production environments for easier parsing by log management systems.
*   Log level can be controlled by `LOG_LEVEL` (e.g., `INFO`, `DEBUG`).

## Troubleshooting

*   **`TypeError: Expected str, not <class 'pydantic.types.SecretStr'>`**:
    *   Ensure your `GOOGLE_API_KEY` is correctly set in `.env`.
    *   Make sure `llm/gemini_client.py` is calling `settings.google_api_key.get_secret_value()` when configuring `genai`.
    *   Delete all `__pycache__` directories and `*.pyc` files in your project and try again.
*   **`TypeError: BaseEventLoop.run_in_executor() got an unexpected keyword argument '...'`**:
    *   This usually means `functools.partial` was not used correctly to bind arguments for functions run in the thread executor. Ensure the wrapper methods (like `_run_collection_method` in `chromadb_manager.py` or the pattern in `gemini_client.py`) correctly bind all keyword arguments to the target function.
*   **Authentication Issues with GitHub**:
    *   Ensure your `GITHUB_TOKEN` (if used) has the correct `repo` scope for private repositories.
    *   For public repositories, no token is usually needed.
    *   Consider setting up SSH keys for Git if HTTPS token authentication is problematic.
*   **ChromaDB Issues**:
    *   Ensure the `CHROMA_DB_PATH` is writable.
    *   If you encounter persistent issues, try deleting the ChromaDB storage directory and re-ingesting.
*   **Gemini API Errors**:
    *   Check your API key and ensure it has the necessary permissions.
    *   If you hit rate limits, consider implementing exponential backoff or retry logic in your calls.
    *   Ensure the `google_genai` package is up-to-date.
*   **Document Ingestion Issues**:
    *   Ensure the file formats are supported and not corrupted.
    *   For large documents, consider increasing the `CHUNK_SIZE` in `config.py`.
    *   If you encounter memory issues, try processing smaller batches of files.
*   **Code Ingestion Issues**:
    *   Ensure the local folder or GitHub repository is accessible.
    *   For large codebases, consider increasing the `CHUNK_SIZE` or processing files in smaller batches.
    *   If cloning a GitHub repo fails, check your network connection and GitHub access permissions.
