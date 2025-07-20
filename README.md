# TestTeller Agent

[![PyPI](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Tests](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml/badge.svg)](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml)
[![Downloads](https://pepy.tech/badge/testteller)](https://pepy.tech/project/testteller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**TestTeller** is a modular AI-powered test automation platform featuring:

- **🤖 Generator Agent**: Intelligent test case generation using advanced RAG technology
- **⚡ Automator Agent**: Automated code generation for multiple programming languages and frameworks  
- **🔧 Core Engine**: Shared functionality with support for multiple LLM providers

The platform analyzes your documentation and codebase to generate comprehensive test strategies, then converts those test cases into executable automation code.

## Features

### Multiple LLM Provider Support
- **Google Gemini** (Default): Fast and cost-effective with support for all Gemini models and Google embeddings
- **OpenAI**: High-quality responses with support for all GPT models and OpenAI embeddings
- **Anthropic Claude**: Advanced reasoning capabilities with support for all Claude models (uses Google or OpenAI for embeddings)
- **Local Llama**: Privacy-focused local inference with support for all Llama models via Ollama
- **Flexible Model Selection**: Configure any supported model from each provider based on your needs and use case
- **Automatic Fallbacks**: Built-in retry mechanisms and error handling for robust performance

### Comprehensive and Strategic Test Generation

TestTeller goes beyond simple test case generation. It acts as a virtual QA architect, analyzing your provided documents and code to create a strategic and comprehensive test suite. The agent intelligently categorizes your context (product requirements, technical designs, code) to generate tests with the appropriate focus and depth.

-   **End-to-End (E2E) Tests**: Generates detailed test cases designed to validate complete user journeys across the entire stack. These tests provide the steps to ensure that the integrated system functions correctly from the user's perspective, covering everything from UI interactions to backend data processing and event propagation.

-   **Integration Tests**: Creates test scenarios to verify the contracts and interactions between components. Whether it's a Frontend-Backend connection, a service-to-service API call, or an event-driven flow, these generated tests help ensure that your system's components can communicate and cooperate as expected.

-   **Technical Tests**: Goes beyond standard functional testing to generate scenarios that probe system limitations, resilience, and security. TestTeller generates test ideas for:
    -   **Performance**: Stress, load, and soak testing hypotheses.
    -   **Resilience**: Retry policies, timeouts, and graceful degradation.
    -   **Security**: Common vulnerabilities and access control issues.
    -   **Edge Cases**: Concurrency, race conditions, and unexpected inputs.

-   **Mocked System Tests**: Produces test cases that enable service and component-level testing by outlining tests for isolated services or functions. The generated tests define clear mock behaviors for external dependencies, allowing a developer or QA engineer to validate a component's logic in isolation.

#### Intelligent Test Strategy
TestTeller doesn't just create tests; it also structures them actionably. It also ensures a healthy mix of happy path, negative, and edge-case scenarios to help you build a truly resilient application.

### Advanced Document Processing & Multi-Format Support
- **Universal Document Parser**: Unified parsing engine supporting **PDF, DOCX, XLSX, MD, TXT** files
- **Enhanced RAG Ingestion**: Smart chunking with metadata extraction for improved retrieval accuracy
- **Document Intelligence**: Automatic document type detection (test cases, requirements, API docs, specifications)
- **Semantic Chunking**: Context-aware text splitting that preserves document structure and meaning
- **Batch Processing**: Concurrent processing of multiple documents with configurable performance settings
- **Code Analysis**: GitHub repositories (public/private) and local codebases across 10+ programming languages
- **Advanced RAG Pipeline**: Context-aware prompt engineering with rich metadata for better generation quality

### TestWriter: Multi-Language Test Automation
- **Four Language Support**: Generate executable automation code in **Python, JavaScript, TypeScript, Java**
- **20+ Framework Support**: 
  - **Python**: pytest, unittest, Playwright, Cucumber
  - **JavaScript/TypeScript**: Jest, Mocha, Playwright, Cypress, Cucumber
  - **Java**: JUnit5, JUnit4, TestNG, Playwright, Karate, Cucumber
- **Multi-Format Input**: Parse test cases from **any supported document format** (not just markdown)
- **AI-Enhanced Generation**: Optional LLM-powered test code enhancement with framework-specific optimizations
- **Interactive Workflows**: Choose specific test cases to automate with rich document preview
- **Production-Ready Output**: Generated tests include complete setup, configuration, and dependency management

## 🏗️ Modular Architecture

TestTeller is built with a clean, modular architecture:

```
testteller/
├── core/                    # 🔧 Shared functionality
│   ├── llm/                # Multi-LLM provider support
│   ├── config/             # Configuration system
│   ├── utils/              # Utility functions
│   ├── vector_store/       # ChromaDB integration
│   └── data_ingestion/     # Document processing
├── generator_agent/         # 🤖 Test case generation
│   ├── agent/              # RAG-powered generation
│   └── prompts.py          # Generation prompts
└── automator_agent/         # ⚡ Code generation
    ├── generators/         # Language-specific generators
    ├── parser/             # Test case parsing
    └── templates/          # Code templates
```

**Benefits:**
- **Modular Design**: Use components independently or together
- **Clean Separation**: Generator and automator agents work autonomously
- **Extensible**: Easy to add new LLM providers, languages, or frameworks
- **Production-Ready**: Follows software engineering best practices

------------------

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- At least one LLM provider API key:
  - Google Gemini: [Get API key](https://aistudio.google.com/)
  - OpenAI: [Get API key](https://platform.openai.com/api-keys)
  - Anthropic Claude: [Get API key](https://console.anthropic.com/)
  - Ollama: [Install locally](https://ollama.ai/)
- GitHub Personal Access Token (optional, for private repositories)

------------------

## Installation

### Option 1: Install from PyPI

```bash
pip install testteller
```

### Option 2: Install from Source

```bash
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
pip install -e .
```

### Option 3: Docker Installation

```bash
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

------------------

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

3. **Ingest Documentation or Code (Enhanced Multi-Format Support)**
```bash
# Enhanced document ingestion with smart chunking (NEW!)
testteller ingest-docs requirements.docx -e -s 1200  # Enhanced with chunk size 1200
testteller ingest-docs test-scenarios.xlsx -e        # Enhanced parsing enabled
testteller ingest-docs specification.pdf -s 800     # Custom chunk size

# Batch ingest directory with mixed formats (shorthand parameters)
testteller ingest-docs ./docs/ -e -s 1000 -c my_docs

# Legacy format support with shorthand
testteller ingest-docs documentation.md -c my_collection
testteller ingest-docs notes.txt -c my_collection

# Code ingestion with shorthand parameters
testteller ingest-code https://github.com/owner/repo.git -c my_collection
testteller ingest-code ./local/code/folder -c my_collection -nc  # No cleanup
```

4. **Generate Test Cases**
```bash
# Generate with shorthand parameters
testteller generate "Create API integration tests for user authentication" -c my_collection -o tests.md

# Generate with custom retrieval count using shorthand
testteller generate "Create technical tests" -c my_collection -n 10 -o technical_tests.md
```

5. **Generate Test Automation Code (Multi-Format Support)**
```bash
# NEW: Multi-format automation support with shorthand parameters
testteller automate test-cases.docx -i             # Interactive mode
testteller automate requirements.xlsx -l python -F pytest -o ./tests
testteller automate test-scenarios.pdf -l javascript -F jest -o ./js_tests

# Enhanced automation with AI optimization (shorthand)
testteller automate tests.md -l python -F playwright -E -p gemini

# Comprehensive example with all shorthand options
testteller automate tests.md -l typescript -F jest -o ./ts_tests -i -E -p openai

# Quick automation with minimal options
testteller automate tests.md -l java -F junit5
```

--------------------

## Configuration

### Enhanced Configuration Wizard

TestTeller now features a modular configuration system with multiple setup options:

```bash
# Full interactive configuration (recommended for first-time setup)
testteller configure

# Quick provider-specific setup (shorthand available)
testteller configure -p gemini    # Using shorthand
testteller configure -p openai
testteller configure -p claude
testteller configure -p llama

# TestWriter automation setup only (updated command)
testteller configure --testwriter  # or use shorthand: -tw
testteller configure -tw          # Shorthand for TestWriter-only setup
```

The configuration wizard now includes:
- **Unified Configuration Flow**: Single wizard for both test generation and automation setup
- **User Choice for TestWriter**: Ask users if they want to configure automation after LLM setup
- **Intelligent Provider Detection**: Automatically detects optimal LLM provider settings
- **TestWriter (Automation) Wizard**: Dedicated setup for test automation preferences
- **Configuration Validation**: Real-time validation with helpful error messages and suggestions
- **Enhanced UI**: Better progress tracking and user experience
- **Backward Compatibility**: All existing configurations continue to work

### CLI Shorthand Parameters

TestTeller now supports convenient shorthand parameters for faster command-line usage:

| Command | Long Form | Shorthand | Description |
|---------|-----------|-----------|-------------|
| **Global** | `--version` | `-v` | Show version |
| **Configure** | `--provider` | `-p` | Quick provider setup |
| **Configure** | `--testwriter` | `-tw` | TestWriter-only setup |
| **Ingest-Docs** | `--collection-name` | `-c` | Collection name |
| **Ingest-Docs** | `--enhanced` | `-e` | Enhanced parsing |
| **Ingest-Docs** | `--chunk-size` | `-s` | Chunk size |
| **Ingest-Code** | `--collection-name` | `-c` | Collection name |
| **Ingest-Code** | `--no-cleanup-github` | `-nc` | Keep cloned repo |
| **Generate** | `--collection-name` | `-c` | Collection name |
| **Generate** | `--num-retrieved` | `-n` | Number of docs |
| **Generate** | `--output-file` | `-o` | Output file |
| **Automate** | `--language` | `-l` | Programming language |
| **Automate** | `--framework` | `-F` | Test framework |
| **Automate** | `--output-dir` | `-o` | Output directory |
| **Automate** | `--interactive` | `-i` | Interactive mode |
| **Automate** | `--enhance` | `-E` | LLM enhancement |
| **Automate** | `--llm-provider` | `-p` | LLM provider |
| **Status** | `--collection-name` | `-c` | Collection name |
| **Clear-Data** | `--collection-name` | `-c` | Collection name |
| **Clear-Data** | `--force` | `-f` | Force without prompt |

**Examples:**
```bash
# Quick automation with all shorthand parameters
testteller automate tests.md -l python -F pytest -o ./tests -E -p gemini

# Fast document ingestion
testteller ingest-docs docs/ -c my_docs -e -s 1500

# Quick test generation
testteller generate "API tests" -c my_docs -n 10 -o api_tests.md
```

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

------------------------

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
# Ingest documents (with shorthand parameters)
testteller ingest-docs path/to/document.pdf -c my_collection
testteller ingest-docs path/to/docs/directory -c my_collection -e -s 1500

# Ingest code from GitHub or local folder (with shorthand)
testteller ingest-code https://github.com/owner/repo.git -c my_collection
testteller ingest-code ./local/code/folder -c my_collection -nc  # No cleanup

# Traditional long-form parameters (still supported)
testteller ingest-docs document.pdf --collection-name my_collection --enhanced --chunk-size 1500
testteller ingest-code repo.git --collection-name my_collection --no-cleanup-github
```

### Test Case Generation
```bash
# Generate test cases (with shorthand parameters)
testteller generate "Create API integration tests for user authentication" -c my_collection

# Generate with custom output file (shorthand)
testteller generate "Create technical tests for login flow" -c my_collection -o tests.md

# Generate with specific number of retrieved documents (shorthand)
testteller generate "Create more than 10 end-to-end tests" -c my_collection -n 10

# Traditional long-form parameters (still supported)
testteller generate "Create tests" --collection-name my_collection --output-file tests.md --num-retrieved 10
```

### TestWriter: Multi-Format Test Automation Generation
```bash
# NEW: Multi-format document support (with shorthand parameters)
testteller automate test-cases.docx -i                           # Interactive mode
testteller automate requirements.xlsx -l python -F pytest -o ./tests
testteller automate specification.pdf -l typescript -F jest -o ./ts_tests
testteller automate test-matrix.xlsx -l java -F junit5 -o ./java_tests

# AI-enhanced code generation with shorthand (NEW!)
testteller automate tests.md -l python -F playwright -E          # Enhanced with default LLM
testteller automate api-tests.docx -l javascript -F cypress -E -p openai  # Enhanced with OpenAI

# Quick automation with minimal parameters
testteller automate tests.md -l python -F pytest                 # Basic automation
testteller automate tests.md -l javascript -F jest -i            # Interactive selection

# Comprehensive example with all shorthand options
testteller automate tests.md -l typescript -F playwright -o ./e2e -i -E -p gemini

# Traditional long-form parameters (still supported)
testteller automate tests.md --language python --framework pytest --output-dir ./tests
testteller automate tests.md --interactive --enhance --llm-provider claude

# Interactive mode with document preview (recommended)
testteller automate any-format-document.* -i
```

### Data Management
```bash
# Check collection status (with shorthand)
testteller status -c my_collection

# Clear collection data (with shorthand)
testteller clear-data -c my_collection -f    # Force clear without confirmation

# Traditional long-form parameters (still supported)
testteller status --collection-name my_collection
testteller clear-data --collection-name my_collection --force
```

------------------------

## Docker Usage

### Using Docker Compose (Recommended)

1. **Setup**
```bash
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
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
# All commands use the same format with docker-compose exec (with shorthand parameters)
docker-compose exec app testteller configure
docker-compose exec app testteller ingest-docs /path/to/doc.pdf -c my_collection -e
docker-compose exec app testteller generate "Create API tests" -c my_collection -o tests.md
docker-compose exec app testteller automate tests.md -l python -F pytest -o ./tests
docker-compose exec app testteller status -c my_collection

# Traditional long-form parameters (still supported)
docker-compose exec app testteller ingest-docs /path/to/doc.pdf --collection-name my_collection
docker-compose exec app testteller generate "Create API tests" --collection-name my_collection
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

------------------------

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

-----------------------

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

------------------------

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
