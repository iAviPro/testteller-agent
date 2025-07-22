# TestTeller Agent

[![PyPI](https://img.shields.io/pypi/v/testteller.svg)](https://pypi.org/project/testteller/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Tests](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml/badge.svg)](https://github.com/iAviPro/testteller-agent/actions/workflows/test-unit.yml)
[![Downloads](https://pepy.tech/badge/testteller)](https://pepy.tech/project/testteller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**TestTeller** is the AI-powered Test agent that transforms your documentation into comprehensive test suites and executable automation code. Powered by a **dual-feedback RAG architecture** with support for multiple GenAI/LLMs (Google Gemini, OpenAI, Anthropic Claude, Local Llama), TestTeller analyzes your requirements, designs, and existing code to generate strategic test cases and automate them across **multiple programming languages** and **supported testing frameworks**.

## Why TestTeller?

TestTeller transforms documentation and code into comprehensive test strategies and executable automation. Unlike traditional testing tools, TestTeller uses dual-feedback RAG architecture to understand your requirements and generate intelligent test scenarios.

### Test Types Generated
- **End-to-End (E2E) Tests**: Complete user journeys across frontend, middleware, and backend services
- **Integration Tests**: Component integration (FE-BE, service-to-service, event-driven) and contract validation  
- **Technical Tests**: Performance, security, resilience testing with infrastructure focus
- **Mocked System Tests**: Isolated component testing with mocked dependencies

### Critical Features
- **Dual-Feedback RAG Enhancement**: Self-improving system that learns from generation cycles and stores high-quality outputs
- **Multi-Provider LLM Support**: Works with Google Gemini, OpenAI, Anthropic Claude, and local Llama/Ollama
- **Universal Document Intelligence**: Advanced parsing for PDFs, DOCX, XLSX, MD, TXT with context understanding
- **Code Repository Analysis**: Ingests and analyzes code from GitHub repos or local folders.

### Supported Languages & Frameworks  
**Test Generation**: All test types with tabular summaries and detailed specifications  
**Automation**: Python (pytest, unittest), JavaScript/TypeScript (Jest, Mocha, Cypress, Playwright), Java (JUnit, TestNG), and more

**Real-world workflow**: Ingest project (PRDs/Contracts/Design/Schema etc.) documentation & project code → Generate strategic test cases covering authentication, error handling, and edge cases → Create executable Selenium/Playwright automation with proper setup and assertions → Commit your code.

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

# 2. Ingest your documentation (supports PDF, DOCX, XLSX, MD, TXT)
testteller ingest-docs requirements.pdf --collection-name my_project

# 3. Ingest code from repository or local folder
testteller ingest-code https://github.com/user/repo --collection-name my_project
# OR: testteller ingest-code ./src --collection-name my_project

# 4. Generate strategic test cases with RAG context
testteller generate "Create comprehensive API integration tests" --collection-name my_project --output-file tests.pdf

# 5. Generate executable automation code
testteller automate tests.pdf --language python --framework pytest --output-dir ./tests
```

**Enhanced Examples:**

```bash
# E2E Testing Workflow
testteller ingest-docs user_stories.pdf --collection-name webapp
testteller ingest-code ./frontend --collection-name webapp  
testteller generate "E2E user registration and checkout flow" --collection-name webapp
testteller automate output.pdf --language javascript --framework cypress

# API Testing with Security Focus
testteller ingest-docs api_spec.pdf --collection-name api
testteller generate "API security and integration tests" --collection-name api --output-format pdf
testteller automate tests.pdf --language python --framework pytest

# Microservices Testing
testteller ingest-code ./services --collection-name microservices
testteller generate "Inter-service communication and resilience tests" --collection-name microservices
testteller automate output.pdf --language java --framework junit
```

**What happens?** TestTeller's dual-feedback RAG analyzes your ingested docs and code, generates strategic test cases using structured templates (E2E, Integration, Technical, Mocked), then creates production-ready automation with proper setup, data management, and CI/CD integration.

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


---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
