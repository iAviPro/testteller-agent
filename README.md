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
./
├── .dockerignore
├── .env.example
├── .github/                  # GitHub Actions, CODEOWNERS, etc.
│   └── workflows/
│       └── python-app.yml
├── .gitignore
├── Dockerfile
├── LICENSE
├── MANIFEST.in
├── README.md                 # This file
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── setup.py
└── testteller/               # Main application package
    ├── __init__.py
    ├── main.py               # CLI entry point
    ├── agent.py              # Core RAG Agent logic
    ├── config.py             # Configuration (Pydantic settings)
    ├── data_ingestion/
    │   ├── __init__.py
    │   ├── code_loader.py      # Handles GitHub/local code loading
    │   ├── document_loader.py  # Handles .docx, .pdf, .xlsx, .txt
    │   └── text_splitter.py    # Text chunking logic
    ├── llm/
    │   ├── __init__.py
    │   └── gemini_client.py    # Gemini LLM and embedding interactions
    ├── prompts.py              # Prompt templates for test case generation
    ├── utils/
    │   ├── __init__.py
    │   ├── helpers.py          # Logging setup
    │   └── retry_helpers.py  # Tenacity retry decorators
    └── vector_store/
        ├── __init__.py
        └── chromadb_manager.py # ChromaDB interactions
```

## Installation

You can install TestTeller directly from PyPI using pip:

```bash
pip install testteller
```

This will download and install the latest stable version of TestTeller along with its dependencies. After installation, you can run the CLI using the `testteller` command.

## Prerequisites

*   Python 3.9+
*   Access to Google Gemini API (requires an API key from [Google AI Studio](https://aistudio.google.com/)).
*   (Optional) GitHub Personal Access Token (PAT) if you intend to clone private repositories. The token needs `repo` scope.
*   Docker and Docker Compose (if running TestTeller using Docker).

## Development Setup (from Source)

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
    *   Set up your environment variables as described in the main "Configuration" section below. If you have cloned the repository, you can copy `.env.example` to `.env` and modify it.

## Usage (CLI)

The main interface to the agent is through the `testteller` command-line tool.

```bash
testteller --help
```

### 1. Ingesting Data

You need to ingest relevant documents and code into a ChromaDB collection before generating test cases.

**Ingest Documents:**
*   From a directory (processes all supported files recursively):
    ```bash
    testteller ingest-docs ./path/to/your/documents/ --collection-name project_alpha_docs
    ```
*   From a single document file:
    ```bash
    testteller ingest-docs ./path/to/your/prd.pdf --collection-name project_alpha_docs
    ```

**Ingest Code:**
*   From a GitHub repository:
    ```bash
    testteller ingest-code https://github.com/owner/repo.git --collection-name project_alpha_code
    ```
*   From a local code folder:
    ```bash
    testteller ingest-code ./path/to/your/local_codebase/ --collection-name project_alpha_code
    ```
*   To prevent deletion of a cloned GitHub repository after ingestion (useful for debugging):
    ```bash
    testteller ingest-code https://github.com/owner/repo.git --collection-name project_alpha_code --no-cleanup-github
    ```

*Note: You can use the same collection name for both documents and code, or separate them.*

### 2. Generating Test Cases

Once data is ingested, you can ask the agent to generate test cases.

```bash
testteller generate "Generate test cases for the user login feature based on the PRD and API docs." --collection-name project_alpha_docs

# Specify number of retrieved context documents and output file
testteller generate "Create API tests for the /users endpoint considering success and failure scenarios." \
    --collection-name project_alpha_code \
    --num-retrieved 7 \
    --output-file user_api_tests.md
```

### 3. Checking Collection Status

To see how many items are in a specific collection:

```bash
testteller status --collection-name project_alpha_docs
```

### 4. Clearing Data

To remove all data from a collection and associated temporary files (like cloned repos):

```bash
# Will ask for confirmation
testteller clear-data --collection-name project_alpha_docs

# Force clear without confirmation
testteller clear-data --collection-name project_alpha_docs --force
```

## Configuration

TestTeller uses environment variables for configuration. These can be set directly in your shell or, more conveniently, by placing them in a `.env` file.

When you run the `testteller` command, it will automatically look for and load a file named `.env` in the directory from which you are running the command.

**Key Environment Variables:**

*   `GOOGLE_API_KEY`: **Required.** Your API key for Google Gemini.
    ```env
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    ```
*   `GITHUB_TOKEN`: **Optional.** A GitHub Personal Access Token with `repo` scope if you need to ingest code from private repositories.
    ```env
    GITHUB_TOKEN="YOUR_GITHUB_PAT"
    ```
*   Other settings (like `LOG_LEVEL`, `CHROMA_DB_PATH`, `DEFAULT_COLLECTION_NAME`) can also be overridden. Refer to `testteller/config.py` for all available settings and their default values (if you have the source code). Key configurable variables beyond `GOOGLE_API_KEY` and `GITHUB_TOKEN` include settings for model names (e.g., `GEMINI_MODEL_NAME`, `EMBEDDING_MODEL_NAME`), text processing (e.g., `CHUNK_SIZE`, `CHUNK_OVERLAP`), and database configuration (e.g., `CHROMA_DB_PATH`, `DEFAULT_COLLECTION_NAME`). You can set these in your `.env` file to override defaults. If you have cloned the repository, you can copy the `.env.example` file to `.env` as a template. Otherwise, simply create a new `.env` file in your working directory with the variables you need.

**Important:**
*   Replace placeholder values like `"YOUR_GEMINI_API_KEY"` with your actual credentials.
*   Ensure the `.env` file is not committed to version control if it contains sensitive information, especially if you are also developing and using a local Git repository. The provided `.gitignore` file (if you cloned the repo) should already include `.env`.

## Running with Docker

TestTeller can be run as a Docker container for an isolated and consistent execution environment. For managing TestTeller along with its dependent services like ChromaDB, using `docker-compose` is the recommended method as it simplifies configuration, networking, and data persistence. Running with `docker run` is an alternative for simpler, standalone execution or specific use cases.

### 1. Using `docker-compose` (Recommended)

The repository includes a `docker-compose.yml` file that defines and manages two main services:
*   `app`: The TestTeller application itself.
*   `chromadb`: A dedicated ChromaDB service that the `app` service connects to.

This setup is ideal for development and more stable deployments.

**Setup:**

1.  **Install Docker Compose:** Ensure you have Docker Compose installed on your system.
2.  **Environment Configuration (`.env` file):**
    *   Create a `.env` file in the root of the project directory (the same directory as `docker-compose.yml`).
    *   You can copy the contents of `.env.example` to your new `.env` file.
    *   **Crucially, populate `GOOGLE_API_KEY` in this `.env` file.**
    *   Other configurations like `GITHUB_TOKEN`, `LOG_LEVEL`, etc., can also be set here. The `docker-compose.yml` is configured to read these variables and pass them to the `app` service.

**Running TestTeller Commands:**

Use `docker-compose run --rm app <testteller_command_and_args>` to execute TestTeller. The `--rm` flag ensures the container is removed after the command finishes, which is suitable for CLI operations.

*   **Show Help:**
    ```bash
    docker-compose run --rm app --help
    ```
*   **Example: Ingesting Data and Generating Tests:**
    To ingest data or generate tests, you might need to make local files accessible to the `app` container.
    *   **Managing Input/Output Data:** The provided `docker-compose.yml` focuses on service orchestration and ChromaDB persistence. To map your local directories (for input documents/code or output files) to the `app` service, you'll need to add or modify the `volumes` section for the `app` service in your `docker-compose.yml`.
        For example, to make a local directory `./my_project_docs` available inside the container at `/app/input_data` and an output directory `./test_results` available at `/app/output_data`, modify the `app` service in `docker-compose.yml`:
        ```yaml
        services:
          app:
            build: .
            # ... other app configurations ...
            volumes:
              - ./my_project_docs:/app/input_data   # Map local input docs
              - ./test_results:/app/output_data     # Map local output directory
        ```
    *   **Ingest Command Example (after adding volume mounts):**
        ```bash
        docker-compose run --rm app ingest-docs /app/input_data --collection-name my_collection
        ```
    *   **Generate Command Example (after adding volume mounts):**
        ```bash
        docker-compose run --rm app generate "Query for tests" --collection-name my_collection --output-file /app/output_data/generated_tests.md
        ```

**Data Persistence (ChromaDB):**
*   The `docker-compose.yml` defines a service `chromadb` and a named volume (e.g., `chroma_data_db_persistent`) to persist ChromaDB data. This means your vector database will survive container restarts and removals.
*   The TestTeller `app` service is configured to connect to this `chromadb` service.

### 2. Using `docker run` (Alternative)

This method is for running the TestTeller agent as a standalone container. It requires manual management of data persistence and linking if a separate database container were used (though these examples assume TestTeller uses its internal, local file-based ChromaDB).

**Building the Docker Image:**
If you have cloned the repository and haven't built the image yet:
```bash
docker build -t testteller-agent .
```
(If you plan to pull a pre-built image from a registry in the future, you can skip this step.)

**Basic Command:**
To see the help message:
```bash
docker run -it --rm testteller-agent --help
```
*   `-it` runs the container in interactive mode with a TTY.
*   `--rm` automatically removes the container when it exits.

**Environment Variables:**
Provide configurations like `GOOGLE_API_KEY`.
*   **Using `--env`:**
    ```bash
    docker run -it --rm --env GOOGLE_API_KEY="YOUR_GEMINI_API_KEY" testteller-agent <testteller_command_here>
    ```
*   **Using an `--env-file` (Recommended for multiple variables):**
    Create a `.env` file on your host machine (e.g., `/path/to/my.env` or `my.env` in your current directory):
    ```env
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    GITHUB_TOKEN="YOUR_GITHUB_PAT" # Optional
    ```
    Then run:
    ```bash
    docker run -it --rm --env-file ./my.env testteller-agent <testteller_command_here>
    ```

**Data Persistence (ChromaDB with `docker run`):**
When using `docker run`, TestTeller defaults to storing ChromaDB data at `/app/chroma_db_data` inside the container. To persist this data, mount a host volume:
```bash
# Create a directory on your host, e.g., ./my_persistent_chroma_data
docker run -it --rm \
    --env-file ./my.env \
    -v "$(pwd)/my_persistent_chroma_data:/app/chroma_db_data" \
    testteller-agent ingest-docs /path/inside_container/to/docs --collection-name my_docs
```
*   Replace `$(pwd)/my_persistent_chroma_data` with your desired host path.

**Mounting Input/Output Data with `docker run`:**
To provide input files and retrieve output files:
```bash
# Host directories: ./my_local_inputs, ./my_test_outputs
# Put documents into ./my_local_inputs

# Ingest:
docker run -it --rm \
    --env-file ./my.env \
    -v "$(pwd)/my_persistent_chroma_data:/app/chroma_db_data" \
    -v "$(pwd)/my_local_inputs:/app/mounted_inputs" \
    testteller-agent ingest-docs /app/mounted_inputs --collection-name docs_collection

# Generate:
docker run -it --rm \
    --env-file ./my.env \
    -v "$(pwd)/my_persistent_chroma_data:/app/chroma_db_data" \
    -v "$(pwd)/my_test_outputs:/app/mounted_outputs" \
    testteller-agent generate "Query" --collection-name docs_collection --output-file /app/mounted_outputs/generated_tests.md
```
*   Use absolute paths for host-side volume mounts or `$(pwd)` (or `%cd%` on Windows CMD).
*   Container paths (e.g., `/app/mounted_inputs`) are used in `testteller` commands.

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
