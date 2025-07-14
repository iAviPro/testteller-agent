# Testing Guide for TestTeller RAG Agent

This guide provides basic information about testing the TestTeller RAG Agent with all supported LLM providers.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── cli/                     # CLI tests
└── data/                    # Test data files
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
pytest --cov=testteller --cov-report=html
```

### Interactive Test Runner
```bash
python tests/test_runner.py
```

## Test Categories

- **Unit Tests**: Test individual components without external dependencies
- **Integration Tests**: Test complete workflows with real LLM providers
- **CLI Tests**: Test command-line interface functionality

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