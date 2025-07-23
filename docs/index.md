---
layout: default
title: TestTeller Documentation
---

<!-- This comment separates Jekyll front matter from content -->

# TestTeller Documentation

**AI-Powered Test Generation and Automation Agent**

[![PyPI](https://img.shields.io/pypi/v/testteller)](https://pypi.org/project/testteller/)
[![Docker Hub](https://img.shields.io/docker/v/iavipro/testteller)](https://hub.docker.com/r/iavipro/testteller)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/pypi/pyversions/testteller)](https://pypi.org/project/testteller/)

---

## Quick Navigation

| 📖 Documentation | 🚀 Resources | 🤝 Community |
|------------------|--------------|--------------|
| [Architecture](ARCHITECTURE.md) | [GitHub Repository](https://github.com/iAviPro/testteller-agent) | [Issues](https://github.com/iAviPro/testteller-agent/issues) |
| [Features](FEATURES.md) | [README](https://github.com/iAviPro/testteller-agent/blob/main/README.md) | |
| [Commands](COMMANDS.md) | [PyPI Package](https://pypi.org/project/testteller/) |  |
| [Testing Guide](TESTING.md) | [Docker Hub](https://hub.docker.com/r/iavipro/testteller) | [License](https://github.com/iAviPro/testteller-agent/blob/main/LICENSE) |

---

## What is TestTeller?

TestTeller is an intelligent test generation and automation agent that leverages:
- **RAG (Retrieval-Augmented Generation)** for context-aware test creation
- **Multiple LLM Providers** (Gemini, OpenAI, Claude, Llama)
- **Dual-feedback System** for continuous learning and improvement
- **Multi-language Support** for various programming languages and frameworks

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Test Generation** | Creates comprehensive test cases from documents, specifications, and code |
| **Automation Code** | Generates production-ready automation scripts in multiple languages |
| **Framework Support** | Works with 20+ testing frameworks out of the box |
| **Quality Assessment** | Built-in quality scoring and validation |
| **Continuous Learning** | Improves over time through feedback loops |

---

## Documentation Pages

### [Architecture](ARCHITECTURE.md)
Deep dive into TestTeller's modular architecture:
- Dual-feedback RAG System
- Generator and Automator Agents
- Vector Store Integration
- Quality Assessment Pipeline

### [Features](FEATURES.md)
Comprehensive feature overview:
- Multiple LLM Provider Support
- Test Generation Capabilities
- Automation Framework Support
- Learning and Feedback Systems

### [Commands Reference](COMMANDS.md)
Complete CLI command documentation:
- Configuration Commands
- Test Generation Commands
- Automation Commands
- Knowledge Base Management

### [Testing Guide](TESTING.md)
Testing approach and guidelines:
- Simplified Test Structure
- Running Tests
- Test Design Philosophy
- Maintenance Guidelines

---

## Quick Start

### Installation

```bash
# Using pip
pip install testteller

# Using Docker
docker pull iavipro/testteller:latest

# From source
git clone https://github.com/iAviPro/testteller-agent.git
cd testteller-agent
pip install -e .
```

### Basic Usage

```bash
# Configure TestTeller
testteller configure

# Generate test cases
testteller generate requirements.md -c collection_name

# Create automation code
testteller automate test-cases.md --language python --framework pytest -c collection_name
```

### Example Workflow

1. **Configure Provider**
   ```bash
   testteller configure --provider gemini
   ```

2. **Generate Test Cases**
   ```bash
   testteller generate docs/ --output test-suite.md
   ```

3. **Create Automation**
   ```bash
   testteller automate test-suite.md --discover-context
   ```

---

## Supported Frameworks

| Language | Frameworks |
|----------|------------|
| **Python** | pytest, unittest, behave, robot, locust |
| **JavaScript** | jest, mocha, cypress, playwright, puppeteer |
| **TypeScript** | jest, mocha, cypress, playwright |
| **Java** | junit, testng, selenide, restassured |

---

## Latest Updates

### Version 1.0.0
- Initial stable release
- Support for 4 LLM providers
- 20+ testing frameworks
- RAG-enhanced automation generation
- Quality-driven feedback loops

---

## Resources & Links

### Project Resources
- [GitHub Repository](https://github.com/iAviPro/testteller-agent) - Source code and development
- [README](https://github.com/iAviPro/testteller-agent/blob/main/README.md) - Project overview
- [PyPI Package](https://pypi.org/project/testteller/) - Python package
- [Docker Hub](https://hub.docker.com/r/iavipro/testteller) - Container images

### Community
- [Report Issues](https://github.com/iAviPro/testteller-agent/issues) - Bug reports and feature requests
- [Discussions](https://github.com/iAviPro/testteller-agent/discussions) - Questions and ideas
- [Contributing](https://github.com/iAviPro/testteller-agent/blob/main/CONTRIBUTING.md) - How to contribute

---

[![GitHub Stars](https://img.shields.io/github/stars/iAviPro/testteller-agent?style=social)](https://github.com/iAviPro/testteller-agent)
[![PyPI Downloads](https://img.shields.io/pypi/dm/testteller)](https://pypi.org/project/testteller/)

**TestTeller - Intelligent Test Agent for Modern Development**