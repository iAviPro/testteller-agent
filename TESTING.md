# Testing Guide for TestTeller Agent

This guide provides comprehensive information about testing the TestTeller Agent with enhanced multi-format document support, unified parsing capabilities, and all supported LLM providers.

## Test Structure

```
tests/
├── conftest.py                       # Pytest configuration and fixtures
├── unit/                            # Unit tests
│   ├── test_parser.py               # TestWriter parser tests
│   ├── test_generators.py           # Code generator tests
│   ├── test_cli_automation.py       # Automation CLI tests
│   ├── test_unified_parser.py       # NEW: Unified document parser tests
│   ├── test_config_system.py        # NEW: Modular configuration tests
│   └── ...                          # Other unit tests
├── integration/                     # Integration tests
│   ├── test_automation_integration.py # End-to-end automation tests
│   ├── test_multiformat_workflow.py   # NEW: Multi-format document tests
│   ├── test_enhanced_rag.py           # NEW: Enhanced RAG ingestion tests
│   └── ...                          # Other integration tests
├── cli/                             # CLI tests
│   └── test_cli_commands.py         # Enhanced CLI command tests
├── test_automation.py               # Automation functionality tests
└── data/                            # Test data files
    ├── sample_document.md           # Markdown test data
    ├── test_cases.docx              # NEW: Word document test data
    ├── requirements.xlsx            # NEW: Excel test data
    ├── specification.pdf            # NEW: PDF test data
    └── sample_code.py               # Code test data
```

## Prerequisites

- Python 3.11 or higher
- Docker (for ChromaDB)
- LLM provider API keys (for integration tests)

## Running Tests

### Install Dependencies
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=testteller --cov=testwriter --cov-report=html
```

### Interactive Test Runner
```bash
python tests/test_runner.py
```

## Test Categories

- **Unit Tests**: Test individual components without external dependencies
- **Integration Tests**: Test complete workflows with real LLM providers  
- **CLI Tests**: Test command-line interface functionality
- **Automation Tests**: Test automation code generation and parsing functionality

### Run Specific Test Categories

```bash
# Run unit tests only
pytest -m "unit"

# Run automation tests only  
pytest -m "automation"

# Run integration tests only
pytest -m "integration"

# Run CLI tests only
pytest -m "cli"

# Run tests for specific test type
python tests/test_runner.py --type automation
```

## Environment Setup

Set appropriate environment variables based on your chosen LLM provider:

```bash
# For Gemini
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your_api_key

# For OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key

# For Claude
export LLM_PROVIDER=claude
export CLAUDE_API_KEY=your_api_key
export OPENAI_API_KEY=your_openai_key  # Required for embeddings

# For Llama
export LLM_PROVIDER=llama
export OLLAMA_BASE_URL=http://localhost:11434
```

## Docker Testing

Start required services:
```bash
docker run -d -p 8000:8000 chromadb/chroma:0.4.22
```

For Llama testing:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull <your-preferred-model>
```

## GitHub Actions

The project includes automated testing workflows:
- **Unit Tests**: Run on every push and PR
- **Integration Tests**: Provider-specific workflows for comprehensive testing

## Common Issues

**ChromaDB Connection**: Ensure ChromaDB is running on port 8000
**API Keys**: Set appropriate API keys for your chosen LLM provider
**Ollama**: Ensure Ollama service is running and models are pulled

For more detailed testing information, refer to the test files in the `tests/` directory. 