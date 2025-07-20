# TestTeller Agent Architecture

## Overview

TestTeller Agent follows a modular architecture with three main components:

```
testteller/
├── core/                    # Shared core functionality
├── generator_agent/         # Test case generation agent
└── automator_agent/         # Test automation code generation agent
```

## Core Module (`testteller/core/`)

The core module provides shared functionality used by both agents:

### Data Ingestion (`core/data_ingestion/`)
- **`document_loader.py`**: Multi-format document loading (PDF, DOCX, XLSX, MD, TXT)
- **`code_loader.py`**: GitHub repository and local code ingestion
- **`text_splitter.py`**: Smart text chunking for RAG processing
- **`unified_document_parser.py`**: Unified parsing engine with document type detection

### LLM Integration (`core/llm/`)
- **`base_client.py`**: Abstract base class for LLM providers
- **`gemini_client.py`**: Google Gemini integration
- **`openai_client.py`**: OpenAI GPT integration
- **`claude_client.py`**: Anthropic Claude integration
- **`llama_client.py`**: Local Llama/Ollama integration
- **`llm_manager.py`**: Provider-agnostic LLM management

### Vector Store (`core/vector_store/`)
- **`chromadb_manager.py`**: ChromaDB integration for RAG functionality

### Configuration (`core/config/`)
- **`config_wizard.py`**: Interactive configuration setup
- **`providers/`**: Provider-specific configuration modules
- **`validators.py`**: Configuration validation utilities

### Utilities (`core/utils/`)
- **`exceptions.py`**: Custom exception classes
- **`helpers.py`**: Common utility functions
- **`loader.py`**: Loading and progress indicators
- **`retry_helpers.py`**: API retry decorators

## Generator Agent (`testteller/generator_agent/`)

Responsible for intelligent test case generation using RAG technology:

### Components
- **`agent/testteller_agent.py`**: Main RAG agent implementation
- **`prompts.py`**: Test case generation prompt templates

### Functionality
- Document and code ingestion into vector stores
- Context-aware test case generation
- Multi-format document processing
- RAG-powered intelligent recommendations

## Automator Agent (`testteller/automator_agent/`)

Responsible for converting test cases into executable automation code:

### Parser (`parser/`)
- **`markdown_parser.py`**: Parse test cases from various document formats

### Generators (`generators/`)
- **`base_generator.py`**: Abstract base for code generators
- **`python_generator.py`**: Python test code generation (pytest, unittest)
- **`javascript_generator.py`**: JavaScript test code generation (Jest, Mocha)
- **`typescript_generator.py`**: TypeScript test code generation
- **`java_generator.py`**: Java test code generation (JUnit, TestNG)

### Enhancement
- **`llm_enhancer.py`**: AI-powered test code enhancement
- **`prompts.py`**: Code generation prompt templates

### CLI
- **`cli.py`**: Command-line interface for automation features

## Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Generator Agent │───▶│  Test Cases     │
│   & Code        │    │      (RAG)       │    │  (Markdown)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                       ┌─────────────────┐    ┌──────────────────┐
                       │  Automation     │◀───│  Automator Agent │
                       │  Code           │    │   (Generators)   │
                       └─────────────────┘    └──────────────────┘
```

## Key Design Principles

### 1. Separation of Concerns
- **Generator Agent**: Focuses on test strategy and case generation
- **Automator Agent**: Focuses on code generation and framework integration
- **Core**: Provides shared infrastructure

### 2. Provider Agnostic
- Support for multiple LLM providers through abstract interfaces
- Easy to add new providers by implementing base classes

### 3. Multi-Format Support
- Unified document parser handles various input formats
- Extensible architecture for new document types

### 4. Framework Flexibility
- Support for multiple testing frameworks per language
- Template-based generation for easy framework addition

### 5. Modularity
- Each component can be used independently
- Clear interfaces between modules
- Easy testing and maintenance

## Configuration Management

The system uses a hierarchical configuration approach:

1. **Environment Variables**: Primary configuration method
2. **Configuration Files**: Secondary configuration
3. **CLI Parameters**: Override for specific commands
4. **Interactive Wizard**: First-time setup and updates

## Extension Points

### Adding New LLM Providers
1. Inherit from `BaseLLMClient`
2. Implement required abstract methods
3. Add provider configuration in `core/config/providers/`

### Adding New Document Formats
1. Extend `UnifiedDocumentParser`
2. Add format detection logic
3. Implement parsing method

### Adding New Code Generators
1. Inherit from `BaseTestGenerator`
2. Implement language-specific templates
3. Add framework configuration

### Adding New Testing Frameworks
1. Update framework templates in respective generator
2. Add framework validation in constants
3. Update CLI help text

## Testing Strategy

The architecture supports comprehensive testing:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Mock Framework**: Extensive mocking for external dependencies

## Security Considerations

- **API Key Management**: Environment-based configuration only
- **Input Validation**: All user inputs validated
- **Safe Imports**: No dynamic code execution
- **Error Handling**: No sensitive information in error messages