# TestTeller RAG Agent

[![PyPI version](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**TestTeller RAG Agent** TestTeller RAG Agent is a versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases. It leverages Google's Gemini LLM and ChromaDB as a vector store. The agent can process various input sources, including PRD documentation, API contracts, technical design documents (HLD/LLD), and code from GitHub repositories or local folders.

## 🚀 Quick Start

1. **Install the Package**
```bash
# Install from PyPI
pip install testteller

# Or clone and install locally
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
pip install -e .
```

2. **Verify Installation**
```bash
# Check if testteller is correctly installed
testteller --version
```

**Configure the Agent**
```bash
# Run the configuration wizard
testteller configure

# Or set environment variables manually
export GOOGLE_API_KEY="your_gemini_api_key"
export GITHUB_TOKEN="your_github_token"  # Optional, for private repos
```

3. **Start ChromaDB**
```bash
# Using Docker (Optional)
docker run -d -p 8000:8000 chromadb/chroma:0.4.15
```

5. **Generate Test Cases**
```bash
# Ingest code or documentation
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection
testteller ingest-docs path/to/document.pdf --collection-name my_collection

# Generate tests
testteller generate "Create API integration tests for user authentication" --collection-name my_collection --output-file ./tests.md
```

## ✨ Features

### 🔄 Intelligent Test Generation

**TestTeller** generates a comprehensive suite of tests by analyzing your documentation and code. It covers multiple layers of your application with a focus on realism and technical depth.

- **Generates Multiple Test Types:**
  - **End-to-End (E2E) Tests:** Simulates complete user journeys, from UI interactions to backend processing, to validate entire workflows.
  - **Integration Tests:** Verifies the contracts and interactions between different components, services, and APIs, including event-driven architectures.
  - **Technical Tests:** Focuses on non-functional requirements, probing for weaknesses in performance, security, and resilience.
  - **Mocked System Tests:** Provides fast, isolated tests for individual components by mocking their dependencies.

- **Ensures Comprehensive Scenario Coverage:**
  - **Happy Paths:** Validates the primary, expected functionality.
  - **Negative & Edge Cases:** Explores system behavior with invalid inputs, at operational limits, and under stress.
  - **Failure & Recovery:** Tests resilience by simulating dependency failures and verifying recovery mechanisms.
  - **Security & Performance:** Assesses vulnerabilities and measures adherence to performance SLAs.

## 🧪 Test Case Types

### End-to-End (E2E) Tests
- Complete user journey coverage
- Focus on business workflows and user interactions
- Documentation includes:
  - User story or journey being tested
  - Prerequisites and test environment setup
  - Step-by-step test flow
  - Expected outcomes at each step
  - Test data requirements

### Integration Tests
- Component interaction verification
- API contract validation
- Documentation includes:
  - Components or services involved
  - Interface specifications
  - Data flow diagrams
  - Error handling scenarios
  - Dependencies and mocking requirements

### Technical Tests
- System limitations testing
- Edge case handling
- Documentation includes:
  - Technical constraints being tested
  - System boundaries and limits
  - Resource utilization scenarios
  - Error conditions and recovery
  - Performance thresholds

### Mocked System Tests
- Isolated component testing assuming the component is mocked
- Functional requirement verification
- Documentation includes:
  - Component specifications
  - Input/output requirements
  - State transitions
  - Configuration requirements
  - Environmental dependencies
  - Authentication and authorization
  - Data validation
  - Response times
  - Resource utilization
  - Load handling

### 📚 Document Processing
- **Multi-Format Support**
  - PDF documents (`.pdf`)
  - Word documents (`.docx`)
  - Excel spreadsheets (`.xlsx`)
  - Markdown files (`.md`)
  - Text files (`.txt`)
  - Source code files (multiple languages)

### 💻 Code Analysis
- **Repository Integration**
  - GitHub repository cloning (public and private)
  - Local codebase analysis
  - Multiple programming language support

### 🧠 Advanced RAG Pipeline
- **State-of-the-Art LLM Integration**
  - Google Gemini 2.0 Flash for fast generation
  - Optimized embeddings using text-embedding-004
  - Context-aware prompt engineering
  - Streaming response support

### 📊 Output Management
- **Flexible Output Formats**
  - Markdown documentation
  - Structured test cases

## 📋 Prerequisites

*   Python 3.11 or higher (Required)
*   Docker and Docker Compose (for containerized deployment)
*   Google Gemini API key ([Get it here](https://aistudio.google.com/))
*   (Optional) GitHub Personal Access Token for private repos

## 🛠️ Installation

### Option 1: Install from PyPI

```bash
# Install the package
pip install testteller

# Set environment variables
export GOOGLE_API_KEY="your_gemini_api_key"
export GITHUB_TOKEN="your_github_token"  # Optional, for private repos

# Start ChromaDB (in a separate terminal) (Optional)
docker run -d -p 8000:8000 chromadb/chroma:0.4.15
```

### Option 2: Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/iAviPro/testteller-rag-agent.git
cd testteller-rag-agent
```

2. Create environment file:
```bash
cat > .env << EOL
GOOGLE_API_KEY=your_gemini_api_key
# Only set GITHUB_TOKEN if you need to access private repos
# GITHUB_TOKEN=your_github_token
LOG_LEVEL=INFO
LOG_FORMAT=json
DEFAULT_COLLECTION_NAME=my_test_collection
EOL
```

3. Start services:
```bash
docker-compose up -d
```

## 📖 Available Commands

### Using pip installation (recommended)

### Configuration
```bash
# Run interactive configuration wizard
testteller configure

# Show all available commands
testteller --help

# Show help for specific command
testteller generate --help
```

### Ingest Documentation & Code
```bash
# Ingest a single document or directory
testteller ingest-docs path/to/document.pdf --collection-name my_collection

# Ingest a directory of documents
testteller ingest-docs path/to/docs/directory --collection-name my_collection

# Ingest code from GitHub repository or local folder
testteller ingest-code https://github.com/owner/repo.git --collection-name my_collection

# Ingest code with custom collection name
testteller ingest-code ./local/code/folder --collection-name my_collection
```

### Generate Test Cases
```bash
# Generate with default settings
testteller generate "Create API integration tests for user authentication" --collection-name my_collection

# Generate tests with custom output file
testteller generate "Create technical tests for login flow" --collection-name my_collection --output-file tests.md

# Generate tests with specific collection and number of retrieved docs
testteller generate "Create more than  end-to-end tests" --collection-name my_collection --num-retrieved 10 --output-file ./tests.md
```

### Manage Data
```bash
# Check collection status
testteller status --collection-name my_collection

# Clear collection data
testteller clear-data --collection-name my_collection --force
```

### Using Docker or Local Development

When using Docker or running from source, use the module format:

```bash
# Format for Docker:
docker-compose exec app python -m testteller.main [command]

# Format for local development:
python -m testteller.main [command]
```

First, ensure your environment variables are set in the `.env` file:
```bash
# Create .env file with required variables
cat > .env << EOL
GOOGLE_API_KEY=your_gemini_api_key
# Only set GITHUB_TOKEN if you need to access private repos
# GITHUB_TOKEN=your_github_token
LOG_LEVEL=INFO
LOG_FORMAT=json
DEFAULT_COLLECTION_NAME=my_test_collection
EOL
```

Note: Only set GITHUB_TOKEN if you actually need to access private repositories. If you don't need it, comment it out or remove it entirely. Setting it to an empty value will cause validation errors.

Then use the commands:
```bash
# Get help
docker-compose exec app python -m testteller.main --help

# Get command-specific help
docker-compose exec app python -m testteller.main generate --help

# Run configuration wizard
docker-compose exec app python -m testteller.main configure

# Ingest documentation
docker-compose exec app python -m testteller.main ingest-docs /path/to/doc.pdf --collection-name my_collection

# Generate tests
docker-compose exec app python -m testteller.main generate "Create API tests" --collection-name my_collection --output-file tests.md

# Check status
docker-compose exec app python -m testteller.main status --collection-name my_collection

# Clear data
docker-compose exec app python -m testteller.main clear-data --collection-name my_collection --force
```

Note: Make sure both the app and ChromaDB containers are healthy before running commands:
```bash
# Check container status
docker-compose ps

# Check container logs if needed
docker-compose logs -f
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required | Notes |
|----------|-------------|---------|----------|-------|
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes | Must be valid API key |
| `GITHUB_TOKEN` | GitHub Personal Access Token | - | No | Don't set if not using private repos |
| `LOG_LEVEL` | Logging level | INFO | No | DEBUG, INFO, WARNING, ERROR |
| `LOG_FORMAT` | Logging format | text | No | text or json |
| `GEMINI_GENERATION_MODEL` | Gemini model for generation | gemini-2.0-flash | No | |
| `GEMINI_EMBEDDING_MODEL` | Model for embeddings | text-embedding-004 | No | |
| `CHUNK_SIZE` | Document chunk size | 1000 | No | |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 | No | |
| `API_RETRY_ATTEMPTS` | Number of API retry attempts | 3 | No | |
| `API_RETRY_WAIT_SECONDS` | Wait time between retries | 2 | No | Seconds |

## 🔧 Troubleshooting

### Common Issues

1. **Container Health Check Failures**
```bash
# Check container logs
docker-compose logs -f

# Restart services
docker-compose restart
```

2. **ChromaDB Connection Issues**
```bash
# Verify ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Check ChromaDB logs
docker-compose logs chromadb
```

3. **Permission Issues**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./chroma_data
sudo chmod -R 777 ./temp_cloned_repos
```

## 📝 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
