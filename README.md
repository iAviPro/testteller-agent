# TestTeller RAG Agent

[![PyPI](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Tests](https://github.com/iAviPro/testteller-rag-agent/actions/workflows/test-unit.yml/badge.svg)](https://github.com/iAviPro/testteller-rag-agent/actions/workflows/test-unit.yml)
[![Downloads](https://pepy.tech/badge/testteller)](https://pepy.tech/project/testteller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

TestTeller RAG Agent is a versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases. It supports multiple LLM providers including Google Gemini, OpenAI, Anthropic Claude, and local Llama models via Ollama. The agent uses ChromaDB as a vector store and can process various input sources, including PRD documentation, API contracts, technical design documents (HLD/LLD), and code from GitHub repositories or local folders.

## Features

### Multiple LLM Provider Support
- **Google Gemini** (Default): Fast and cost-effective with support for all Gemini models and Google embeddings
- **OpenAI**: High-quality responses with support for all GPT models and OpenAI embeddings
- **Anthropic Claude**: Advanced reasoning capabilities with support for all Claude models (uses Google or OpenAI for embeddings)
- **Local Llama**: Privacy-focused local inference with support for all Llama models via Ollama
- **Flexible Model Selection**: Configure any supported model from each provider based on your needs and use case
- **Automatic Fallbacks**: Built-in retry mechanisms and error handling for robust performance

### Comprehensive Test Generation
- **End-to-End Tests**: Complete user journey validation
- **Integration Tests**: Component interaction and API contract verification
- **Technical Tests**: System limitations, edge cases, and performance validation
- **Mocked System Tests**: Isolated component testing with mocked dependencies

### Document and Code Processing
- **Multi-format Document Support**: PDF, DOCX, XLSX, MD, TXT files
- **Code Analysis**: GitHub repositories (public/private) and local codebases
- **Multiple Programming Languages**: Python, JavaScript, TypeScript, Java, Go, Rust, C++, C#, Ruby, PHP
- **Advanced RAG Pipeline**: Context-aware prompt engineering and streaming responses



## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- At least one LLM provider API key:
  - Google Gemini: [Get API key](https://aistudio.google.com/)
  - OpenAI: [Get API key](https://platform.openai.com/api-keys)
  - Anthropic Claude: [Get API key](https://console.anthropic.com/)
  - Ollama: [Install locally](https://ollama.ai/)
- GitHub Personal Access Token (optional, for private repositories)

## Installation

### Option 1: Install from PyPI

```bash
pip install testteller
```

### Option 2: Install from Source

```bash
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
pip install -e .
```

### Option 3: Docker Installation

```bash
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

## Quick Start

1. **Configure the Agent**
```bash
testteller configure
```

2. **For Llama Users: Setup Ollama**

**Local Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service and install models (example models)
ollama serve
ollama pull llama3.2  # or any preferred Llama model
ollama pull <your-preferred-model>
```

**Remote/Docker Ollama:**
TestTeller supports connecting to Ollama running on Docker or remote servers. During configuration, you can specify the Ollama server URL and port separately:
- **Local**: URL: `localhost` (default), Port: `11434` (default) → `http://localhost:11434`
- **Docker**: URL: `docker-host` or `host.docker.internal`, Port: `11434` → `http://docker-host:11434`
- **Remote**: URL: `remote-server`, Port: `11434` → `http://remote-server:11434`

The configuration wizard will ask for URL and Port separately, then automatically form the complete URL:PORT combination.

3. **Ingest Documentation or Code**
```bash
# Ingest documents
testteller ingest-docs path/to/document.pdf --collection-name my_collection

# Ingest code from GitHub
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection

# Ingest local code
testteller ingest-code ./local/code/folder --collection-name my_collection
```

4. **Generate Test Cases**
```bash
testteller generate "Create API integration tests for user authentication" --collection-name my_collection --output-file tests.md
```

## Configuration

### Environment Variables

The application uses a `.env` file for configuration. Run `testteller configure` to set up interactively, or create a `.env` file manually with the following variables:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| **LLM Provider Selection** | | | |
| `LLM_PROVIDER` | LLM provider to use (`gemini`, `openai`, `claude`, `llama`) | `gemini` | No |
| **Google Gemini Configuration** | | | |
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes (for Gemini) |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model | Configure as needed | No |
| `GEMINI_GENERATION_MODEL` | Gemini generation model | Configure as needed | No |
| **OpenAI Configuration** | | | |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes (for OpenAI/Claude) |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | Configure as needed | No |
| `OPENAI_GENERATION_MODEL` | OpenAI generation model | Configure as needed | No |
| **Anthropic Claude Configuration** | | | |
| `CLAUDE_API_KEY` | Anthropic Claude API key | - | Yes (for Claude) |
| `CLAUDE_GENERATION_MODEL` | Claude generation model | Configure as needed | No |
| `CLAUDE_EMBEDDING_PROVIDER` | Embedding provider for Claude | `google` | No |
| **Llama/Ollama Configuration** | | | |
| `LLAMA_EMBEDDING_MODEL` | Llama embedding model | Configure as needed | No |
| `LLAMA_GENERATION_MODEL` | Llama generation model | Configure as needed | No |
| `OLLAMA_BASE_URL` | Ollama server URL (local/remote/Docker) | `http://localhost:11434` | No |
| **GitHub Integration** | | | |
| `GITHUB_TOKEN` | GitHub Personal Access Token | - | No |
| **ChromaDB Configuration** | | | |
| `CHROMA_DB_HOST` | ChromaDB host | `localhost` | No |
| `CHROMA_DB_PORT` | ChromaDB port | `8000` | No |
| `CHROMA_DB_USE_REMOTE` | Use remote ChromaDB | `false` | No |
| `CHROMA_DB_PERSIST_DIRECTORY` | Local ChromaDB directory | `./chroma_data` | No |
| `DEFAULT_COLLECTION_NAME` | Default collection name | `test_collection` | No |
| **Document Processing** | | | |
| `CHUNK_SIZE` | Document chunk size | `1000` | No |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` | No |
| `CODE_EXTENSIONS` | Code file extensions | `.py,.js,.ts,.java,.go,.rs,.cpp,.c,.cs,.rb,.php` | No |
| `TEMP_CLONE_DIR_BASE` | Temporary clone directory | `./temp_cloned_repos` | No |
| **Output Configuration** | | | |
| `OUTPUT_FILE_PATH` | Default output file path | `testteller-testcases.md` | No |
| **API Retry Configuration** | | | |
| `API_RETRY_ATTEMPTS` | Number of retry attempts | `3` | No |
| `API_RETRY_WAIT_SECONDS` | Wait time between retries | `2` | No |
| **Logging Configuration** | | | |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `LOG_FORMAT` | Logging format | `json` | No |

**Provider-Specific Notes:**
- **Gemini**: Only requires `GOOGLE_API_KEY`
- **OpenAI**: Only requires `OPENAI_API_KEY`
- **Claude**: Requires `CLAUDE_API_KEY` and API key for selected embedding provider (Google or OpenAI)
  - **Important**: Claude uses other providers for embeddings since it doesn't have its own embedding API
  - If you select `google` as embedding provider, you need `GOOGLE_API_KEY`
  - If you select `openai` as embedding provider, you need `OPENAI_API_KEY`
  - Run `testteller configure` to set up Claude configuration interactively
- **Llama**: No API key required, but needs Ollama installation and model downloads
  - Supports local installation (`http://localhost:11434`) 
  - Supports remote/Docker connections (e.g., `http://remote-server:11434`)
  - Configure URL and Port separately via `testteller configure` (URL defaults to `localhost`, Port defaults to `11434`)
  - Complete URL is automatically formed as `http://URL:PORT` and saved as `OLLAMA_BASE_URL` environment variable
- **GitHub**: Only set `GITHUB_TOKEN` if accessing private repositories

## Available Commands

### Configuration
```bash
# Interactive configuration wizard
testteller configure

# Show version
testteller --version

# Show help
testteller --help
```

### Document and Code Ingestion
```bash
# Ingest documents
testteller ingest-docs path/to/document.pdf --collection-name my_collection
testteller ingest-docs path/to/docs/directory --collection-name my_collection

# Ingest code from GitHub or local folder
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection
testteller ingest-code ./local/code/folder --collection-name my_collection --no-cleanup-github
```

### Test Case Generation
```bash
# Generate test cases
testteller generate "Create API integration tests for user authentication" --collection-name my_collection

# Generate with custom output file
testteller generate "Create technical tests for login flow" --collection-name my_collection --output-file tests.md

# Generate with specific number of retrieved documents
testteller generate "Create more than 10 end-to-end tests" --collection-name my_collection --num-retrieved 10
```

### Data Management
```bash
# Check collection status
testteller status --collection-name my_collection

# Clear collection data
testteller clear-data --collection-name my_collection --force
```

## Docker Usage

### Using Docker Compose (Recommended)

1. **Setup**
```bash
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
cp .env.example .env
# Edit .env with your API keys and preferred LLM provider
```

2. **Start Services**
```bash
# For cloud providers (Gemini, OpenAI, Claude)
docker-compose up -d

# For Llama with local Docker Ollama (uncomment ollama service in docker-compose.yml first)
docker-compose up -d
docker-compose exec ollama ollama pull <your-preferred-model>
docker-compose exec ollama ollama pull <your-preferred-model>

# For Llama with remote Ollama (set OLLAMA_BASE_URL in .env)
# Example: OLLAMA_BASE_URL=http://remote-server:11434
docker-compose up -d app
```

3. **Run Commands**
```bash
# All commands use the same format with docker-compose exec
docker-compose exec app testteller configure
docker-compose exec app testteller ingest-docs /path/to/doc.pdf --collection-name my_collection
docker-compose exec app testteller generate "Create API tests" --collection-name my_collection
docker-compose exec app testteller status --collection-name my_collection
```

### Docker Management
```bash
# View logs
docker-compose logs app
docker-compose logs chromadb

# Stop services
docker-compose down

# Remove all data
docker-compose down -v
```

## Testing

TestTeller includes a comprehensive test suite for all supported LLM providers.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=testteller --cov-report=html

# Interactive test runner
python tests/test_runner.py
```

For detailed testing documentation, see [TESTING.md](TESTING.md).

## Troubleshooting

### Common Issues

**API Key Errors**
- Ensure correct API keys are set in `.env` file
- For Claude, `CLAUDE_API_KEY` is required plus API key for the embedding provider:
  - If using `google` for embeddings: `GOOGLE_API_KEY` is required
  - If using `openai` for embeddings: `OPENAI_API_KEY` is required
- Run `testteller configure` to verify configuration and set up Claude properly
- Common Claude error: "Google/OpenAI API key is required for embeddings when using Claude"

**ChromaDB Connection Issues**
```bash
# Check ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# For Docker
docker-compose logs chromadb
docker-compose restart chromadb
```

**Ollama Issues (Llama Provider)**
```bash
# Check Ollama service
ollama list

# Install missing models
ollama pull llama3.2:3b
ollama pull llama3.2:1b

# For Docker
docker-compose exec ollama ollama list
```

**Permission Issues**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./chroma_data
sudo chmod -R 755 ./temp_cloned_repos
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
