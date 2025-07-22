# TestTeller Agent

[![PyPI](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Tests](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml/badge.svg)](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml)
[![Downloads](https://pepy.tech/badge/testteller)](https://pepy.tech/project/testteller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**TestTeller** is the AI-powered Test agent that transforms your documentation into comprehensive test suites and executable automation code. Powered by a **dual-feedback RAG architecture** with support for multiple GenAI/LLMs (Google Gemini, OpenAI, Anthropic Claude, Local Llama), TestTeller analyzes your requirements, designs, and existing code to generate strategic test cases and automate them across **multiple programming languages** and **supported testing frameworks**.

## Why TestTeller?

Imagine feeding your product requirements document to an AI and getting back:
- **Complete test strategies** covering E2E, integration, System Tests, Technical Tests including security, and edge cases
- **Production-ready automation code** in Python, JavaScript, TypeScript, or Java
- **Self-improving system** that learns from each generation cycle

**Real-world example**: Upload a PDF with API specifications → Get 50+ strategic test cases → Generate executable Playwright tests with proper authentication, data handling, and assertions → All in under 5 minutes.

## Key Features

- **🤖 Generator Agent**: Virtual Test architect with dual-feedback RAG enhancement - analyzes docs and generates strategic test cases with intelligent categorization (E2E, integration, security, edge cases)
- **⚡ Automator Agent**: Multi-language code generation across **Python, JavaScript, TypeScript, Java** with **20+ framework support** (pytest, Jest, JUnit, Playwright, Cypress, Cucumber, etc.)
- **🔧 Multi-Provider GenAI/LLM**: Choose your AI provider - **Google Gemini, OpenAI, Anthropic Claude**, or run completely **local with Llama/Ollama**
- **📄 Universal Document Intelligence**: Advanced RAG ingestion for **PDFs, DOCX, XLSX, MD, TXT** - understands context and generates appropriate test focus
- **🔄 Self-Learning System**: Dual-feedback architecture gets smarter with each use - stores high-quality outputs and learns from automation success patterns

→ **[View Detailed Features](FEATURES.md)** | **[Technical Architecture](ARCHITECTURE.md)**

## Quick Start

### Prerequisites
- Python 3.11+
- API key for at least one LLM provider:
  - [Google Gemini](https://aistudio.google.com/) (recommended)
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Anthropic Claude](https://console.anthropic.com/)
  - [Ollama](https://ollama.ai/) (local or within accessible environment)

### Installation

```bash
# Install from PyPI
pip install testteller

# Or install from source
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
pip install -e .
```

### Basic Usage - Get Started in 2 Minutes

```bash
# 1. Configure your LLM provider (interactive wizard)
testteller configure

# 2. Ingest your documentation (any format supported)
testteller ingest-docs requirements.pdf --collection-name my_project

# 3. Generate strategic test cases from your docs
testteller generate "Create comprehensive API integration tests" --collection-name my_project --output-file tests.pdf

# 4. Generate executable automation code
testteller automate tests.pdf --language python --framework pytest --output-dir ./tests
```

**What just happened?** TestTeller's dual-feedback RAG architecture analyzed your requirements, design, contracts and code using multiple GenAI/LLMs, generated strategic test cases covering happy paths, edge cases, and security scenarios, then created production-ready automation code across multiple languages and frameworks with proper setup, authentication, and assertions.

### Try TestTeller Now

**No API Keys?** No problem - use local Llama:
```bash
# Install Ollama (macOS/Linux)  
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2

# Configure TestTeller for local use
testteller configure --provider llama
```

## Docker Support

```bash
# Clone and setup
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
cp .env.example .env  # Add your API keys
docker-compose up -d

# Use with Docker
docker-compose exec app testteller configure
docker-compose exec app testteller ingest-docs document.pdf --collection-name project
```

## Documentation

- **[Complete Features](FEATURES.md)** - Detailed feature descriptions and capabilities
- **[Technical Architecture](ARCHITECTURE.md)** - System design and technical details  
- **[Command Reference](COMMANDS.md)** - Complete CLI command documentation
- **[Testing Guide](TESTING.md)** - Test suite and validation documentation

## Common Issues

Run `testteller configure` if you encounter API key errors. For Docker issues, check logs with `docker-compose logs app`.

## Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
