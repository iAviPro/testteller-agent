# TestTeller RAG Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

**TestTeller RAG Agent** is a powerful CLI-based RAG (Retrieval Augmented Generation) agent designed to generate comprehensive software test cases. It leverages Google's Gemini LLM and ChromaDB as a vector store to process various input sources and generate high-quality test cases.

## ✨ Features

### 🔄 Intelligent Test Generation
- **Multiple Test Types**
  - Technical test cases for components, APIs, and system architecture
  - User journey test cases for end-to-end flows
  - Integration test scenarios
  - Edge case and error handling tests
  - Performance test considerations

### 📚 Document Processing
- **Multi-Format Support**
  - PDF documents (`.pdf`)
  - Word documents (`.docx`)
  - Excel spreadsheets (`.xlsx`)
  - Markdown files (`.md`)
  - Text files (`.txt`)
  - Source code files (multiple languages)

- **Smart Document Handling**
  - Automatic text extraction from all supported formats
  - Table and structured data processing from Excel files
  - Code context preservation
  - Metadata extraction and storage

### 💻 Code Analysis
- **Repository Integration**
  - GitHub repository cloning (public and private)
  - Local codebase analysis
  - Multiple programming language support
  - Branch-specific analysis

- **Code Understanding**
  - Function and class relationship analysis
  - API endpoint detection
  - Error handling pattern recognition
  - Configuration and environment awareness

### 🧠 Advanced RAG Pipeline
- **State-of-the-Art LLM Integration**
  - Google Gemini 2.0 Flash for fast generation
  - Optimized embeddings using text-embedding-004
  - Context-aware prompt engineering
  - Streaming response support

- **Vector Store Optimization**
  - Efficient ChromaDB integration
  - Fast similarity search
  - Persistent storage
  - Collection management
  - Metadata-based filtering

### 🔧 Processing Capabilities
- **Smart Text Processing**
  - Intelligent text chunking
  - Context preservation across chunks
  - Overlap handling for better context
  - Custom chunk size configuration

- **Performance Optimization**
  - Asynchronous processing
  - Batch operations
  - Concurrent document processing
  - Memory-efficient handling

### 🛠️ Developer Tools
- **CLI Features**
  - Interactive command-line interface
  - Progress tracking
  - Rich error messages
  - Configuration management
  - Collection statistics

- **Integration Options**
  - Docker containerization
  - REST API support (via ChromaDB)
  - Environment variable configuration
  - Custom output formatting

### 📊 Output Management
- **Flexible Output Formats**
  - Markdown documentation
  - Structured test cases
  - Custom formatting options
  - Machine-readable formats

- **Result Organization**
  - Test case categorization
  - Priority assignment
  - Traceability to source
  - Coverage reporting

### 🔐 Security Features
- **Secure Processing**
  - API key management
  - Token-based authentication
  - Secure credential handling
  - Non-root container execution

- **Data Protection**
  - Local data persistence
  - Configurable data retention
  - Secure temporary file handling
  - Private repository support

### ⚡ Performance Features
- **Resource Optimization**
  - Configurable resource limits
  - Memory-efficient processing
  - Concurrent operations
  - Background processing support

- **Scalability**
  - Horizontal scaling with Docker
  - Load balancing support
  - Distributed processing capability
  - Resource monitoring

### 🔍 Monitoring & Debugging
- **Observability**
  - Comprehensive logging
  - Performance metrics
  - Health checks
  - Status monitoring

- **Troubleshooting**
  - Detailed error reporting
  - Debug mode
  - Log level configuration
  - Health check endpoints

## 🚀 Key Features

*   **Intelligent Test Generation**:
    *   Technical test cases for components, APIs, and system architecture
    *   User journey test cases for end-to-end flows
    *   Context-aware test generation using RAG
    *   Support for multiple testing frameworks and languages

*   **Advanced Document Processing**:
    *   Multi-format support: `.docx`, `.pdf`, `.xlsx`, `.txt`, `.md`
    *   Code repository analysis (GitHub/local)
    *   Smart chunking and context preservation
    *   Efficient vector storage with ChromaDB

*   **Modern Architecture**:
    *   Google Gemini 2.0 Flash for fast generation
    *   Optimized embeddings with text-embedding-004
    *   Containerized deployment with Docker
    *   Asynchronous processing capabilities

## 📋 Prerequisites

*   Python 3.11 or higher (Required)
*   Docker and Docker Compose (for containerized deployment)
*   Google Gemini API key ([Get it here](https://aistudio.google.com/))
*   (Optional) GitHub Personal Access Token for private repos

## 🛠️ Installation

### Option 1: Local Installation (pip)

1. Ensure you have Python 3.11+ installed:
   ```bash
   python --version  # Should show 3.11 or higher
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Linux/macOS
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. Install from PyPI:
   ```bash
   pip install testteller
   ```

4. Set up environment variables:
   ```bash
   # Linux/macOS
   export GOOGLE_API_KEY="your_gemini_api_key"
   export GITHUB_TOKEN="your_github_token"  # Optional, for private repos
   
   # Windows (PowerShell)
   $env:GOOGLE_API_KEY="your_gemini_api_key"
   $env:GITHUB_TOKEN="your_github_token"
   ```

5. Install and start ChromaDB:
   ```bash
   # Install ChromaDB
   pip install chromadb

   # Start ChromaDB server (in a separate terminal)
   chroma run --path ./chroma_data_local --host 0.0.0.0 --port 8000
   ```

6. Verify installation:
   ```bash
   testteller --help
   ```

### Option 2: Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/testteller-rag-agent.git
   cd testteller-rag-agent
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   ```

4. Create a `.env` file:
   ```bash
   cat > .env << EOL
   GOOGLE_API_KEY=your_gemini_api_key
   GITHUB_TOKEN=your_github_token  # Optional
   LOG_LEVEL=INFO
   DEFAULT_COLLECTION_NAME=my_collection
   CHROMA_DB_HOST=localhost
   CHROMA_DB_PORT=8000
   CHROMA_DB_USE_REMOTE=true
   CHROMA_DB_PERSIST_DIRECTORY=./chroma_data_local
   EOL
   ```

5. Start ChromaDB server (in a separate terminal):
   ```bash
   chroma run --path ./chroma_data_local --host 0.0.0.0 --port 8000
   ```

### Option 3: Docker Installation (Recommended for Production)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/testteller-rag-agent.git
   cd testteller-rag-agent
   ```

2. Create environment file:
   ```bash
   # Create .env file
   cat > .env << EOL
   GOOGLE_API_KEY=your_gemini_api_key
   GITHUB_TOKEN=your_github_token  # Optional
   LOG_LEVEL=INFO
   DEFAULT_COLLECTION_NAME=my_test_collection
   EOL
   ```

3. Start services:
   ```bash
   # Build and start containers
   docker-compose up -d --build

   # Verify services are healthy
   docker-compose ps
   ```

## 🎯 Quick Start

### 1. Data Ingestion

```bash
# Ingest documentation
testteller ingest-docs ./docs \
    --collection-name my_project \
    --recursive

# Ingest GitHub repository
testteller ingest-code https://github.com/owner/repo.git \
    --collection-name my_project \
    --branch main

# Ingest local code
testteller ingest-code ./src \
    --collection-name my_project \
    --extensions ".py,.js,.ts"
```

### 2. Generate Test Cases

```bash
# Basic test generation
testteller generate "Create API integration tests for user authentication" \
    --collection-name my_project

# Advanced options
testteller generate "Generate end-to-end user journey tests" \
    --collection-name my_project \
    --num-retrieved 10 \
    --output-file tests.md \
    --format markdown
```

### 3. Manage Collections

```bash
# View collection status
testteller status --collection-name my_project

# Clear collection
testteller clear-data \
    --collection-name my_project \
    --force
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes |
| `GITHUB_TOKEN` | GitHub Personal Access Token | - | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `GEMINI_GENERATION_MODEL` | Gemini model for generation | gemini-2.0-flash | No |
| `GEMINI_EMBEDDING_MODEL` | Model for embeddings | text-embedding-004 | No |
| `CHUNK_SIZE` | Document chunk size | 1000 | No |
| `CHUNK_OVERLAP` | Chunk overlap size | 200 | No |

### Docker Resources

The application is configured with sensible resource limits:

- **App Container**:
  - CPU: 2 cores (min: 0.5)
  - Memory: 4GB (min: 1GB)

- **ChromaDB Container**:
  - CPU: 1 core (min: 0.25)
  - Memory: 2GB (min: 512MB)

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

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Make your changes
4. Run tests:
   ```bash
   pytest
   flake8
   black .
   ```
5. Submit a pull request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini team for the amazing LLM
- ChromaDB team for the efficient vector store
- All our contributors and users

## 📚 Additional Resources

- [API Documentation](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Security Policy](SECURITY.md)

## 📝 Local Usage Examples

### Basic Usage

1. Ingest documentation:
   ```bash
   # Load documentation from a directory
   testteller ingest-docs ./docs \
       --collection-name my_project \
       --recursive

   # Load specific file types
   testteller ingest-docs ./specs \
       --collection-name my_project \
       --file-types ".md,.txt,.pdf"
   ```

2. Process code repositories:
   ```bash
   # Load from GitHub
   testteller ingest-code https://github.com/owner/repo.git \
       --collection-name my_project \
       --branch main

   # Load local code
   testteller ingest-code ./src \
       --collection-name my_project \
       --extensions ".py,.js,.ts"
   ```

3. Generate test cases:
   ```bash
   # Basic generation
   testteller generate "Create unit tests for authentication module" \
       --collection-name my_project

   # With specific output format
   testteller generate "Generate API integration tests" \
       --collection-name my_project \
       --output-file tests.md \
       --format markdown
   ```

### Advanced Usage

1. Custom chunking:
   ```bash
   testteller ingest-docs ./large_docs \
       --collection-name my_project \
       --chunk-size 1500 \
       --chunk-overlap 300
   ```

2. Multiple source types:
   ```bash
   # Combine documentation and code
   testteller ingest-docs ./docs --collection-name my_project
   testteller ingest-code ./src --collection-name my_project
   testteller ingest-code https://github.com/owner/repo.git --collection-name my_project
   ```

3. Advanced test generation:
   ```bash
   testteller generate "Create end-to-end tests" \
       --collection-name my_project \
       --num-retrieved 15 \
       --output-file e2e_tests.md \
       --format markdown \
       --temperature 0.7
   ```

### Managing Collections

```bash
# List all collections
testteller list-collections

# View collection details
testteller status --collection-name my_project

# Clear collection data
testteller clear-data \
    --collection-name my_project \
    --force

# Export collection
testteller export-collection \
    --collection-name my_project \
    --output-dir ./backup
```

### Troubleshooting Local Setup

1. ChromaDB Connection Issues:
   ```bash
   # Verify ChromaDB is running
   curl http://localhost:8000/api/v1/heartbeat

   # Check ChromaDB logs
   tail -f chroma_data_local/chroma.log
   ```

2. Permission Issues:
   ```bash
   # Fix data directory permissions
   chmod -R 755 ./chroma_data_local
   ```

3. Environment Issues:
   ```bash
   # Verify environment variables
   testteller check-env

   # Reset ChromaDB data
   rm -rf ./chroma_data_local/*
   ```
