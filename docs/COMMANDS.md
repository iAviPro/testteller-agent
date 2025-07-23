# TestTeller CLI Command Reference

Complete reference for all TestTeller CLI commands, parameters, and usage examples.

## Global Options

All commands support these global options:
- `--version, -v`: Show version and exit
- `--help, -h`: Show help message and exit

## Core Commands

### `testteller configure`
**Interactive configuration wizard to set up TestTeller**

```bash
testteller configure [OPTIONS]
```

**Options:**
- `--provider, -p [gemini|openai|claude|llama]`: Quick setup for specific LLM provider
- `--automator-agent, -aa`: Configure Automator Agent settings only

**Examples:**
```bash
# Full interactive configuration
testteller configure

# Quick Gemini setup
testteller configure --provider gemini

# Configure automation settings only
testteller configure --automator-agent
```

**Features:**
- Interactive wizard with step-by-step guidance
- Provider-specific configuration flows
- API key validation and testing
- .env file generation and management

---

### `testteller ingest-docs`
**Ingest documents into knowledge base for test generation**

```bash
testteller ingest-docs PATH [OPTIONS]
```

**Arguments:**
- `PATH` (required): Path to document file or directory

**Supported Formats:** PDF, DOCX, XLSX, MD, TXT

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection name for organizing documents
- `--enhanced, -e / --no-enhanced`: Use enhanced parsing with metadata extraction (default: enabled)
- `--chunk-size, -s INTEGER`: Text chunk size for optimal retrieval (100-5000, default: 1000)

**Examples:**
```bash
# Ingest single document with enhanced parsing
testteller ingest-docs requirements.pdf --collection-name my_project --enhanced

# Ingest directory with custom chunk size
testteller ingest-docs ./documentation --chunk-size 1500 --collection-name project_docs

# Simple ingestion with defaults
testteller ingest-docs api-spec.docx --collection-name api_tests
```

**Features:**
- Multi-format document parsing with structure preservation
- Smart chunking with configurable sizes
- Metadata extraction for improved retrieval
- Progress indicators during ingestion
- Batch processing for directories

---

### `testteller ingest-code`
**Ingest code repositories or local codebases for context**

```bash
testteller ingest-code SOURCE_PATH [OPTIONS]
```

**Arguments:**
- `SOURCE_PATH` (required): GitHub repository URL or local code directory path

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection name
- `--no-cleanup-github, -nc`: Keep cloned repository after ingestion

**Examples:**
```bash
# Ingest GitHub repository
testteller ingest-code https://github.com/owner/repo.git --collection-name project_code

# Ingest local codebase
testteller ingest-code ./src --collection-name local_code

# Keep GitHub repo after ingestion
testteller ingest-code https://github.com/owner/repo.git --no-cleanup-github
```

**Supported Languages:** Python, JavaScript, TypeScript, Java, Go, Rust, C++, C, C#, Ruby, PHP

**Features:**
- Automatic GitHub repository cloning
- Multi-language code analysis
- Pattern recognition for test automation
- Temporary file management
- Code structure understanding

---

### `testteller generate`
**Generate strategic test cases using RAG-enhanced AI**

```bash
testteller generate QUERY [OPTIONS]
```

**Arguments:**
- `QUERY` (required): Description of tests to generate

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection for context retrieval
- `--num-retrieved, -n INTEGER`: Number of context documents (0-20, default: 5)
- `--output-file, -o TEXT`: Output file path (auto-generated if not provided)
- `--output-format, -f [md|pdf|docx]`: Output format (default: pdf)

**Examples:**
```bash
# Generate API integration tests
testteller generate "API integration tests for user authentication" --collection-name my_project --output-format pdf

# Generate comprehensive test suite
testteller generate "End-to-end user journey tests covering registration, login, and checkout" --collection-name ecommerce --num-retrieved 10

# Generate security tests
testteller generate "Security tests for input validation and access control" --collection-name security_docs --output-file security_tests.md
```

**Test Types Generated:**
- End-to-End (E2E) user journey tests
- Integration tests for component interactions  
- Technical tests (performance, security, edge cases)
- Unit tests with mocking strategies
- API testing scenarios

**Features:**
- Strategic test categorization and prioritization
- Context-aware generation using ingested documents
- Multiple output formats with proper formatting
- Quality assessment and scoring
- Automatic learning from high-quality outputs

---

### `testteller automate`
**Generate executable automation code from test cases**

```bash
testteller automate INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE` (required): Test cases file (PDF, DOCX, XLSX, MD, TXT)

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection for application context
- `--language, -l [python|javascript|typescript|java]`: Programming language
- `--framework, -F TEXT`: Test framework to use
- `--output-dir, -o TEXT`: Output directory (default: ./testteller_automated_tests)
- `--interactive, -i`: Interactive mode for test case selection
- `--enhance, -E`: Enable AI enhancement for code quality
- `--llm-provider, -p [gemini|openai|claude|llama]`: LLM provider for enhancement
- `--num-context, -n INTEGER`: Context documents to retrieve (1-20, default: 5)
- `--verbose, -v`: Enable verbose logging

**Supported Frameworks:**

| Language | Frameworks |
|----------|------------|
| Python | pytest, unittest, playwright, cucumber |
| JavaScript | jest, mocha, playwright, cypress, cucumber |
| TypeScript | jest, mocha, playwright, cypress, cucumber |
| Java | junit5, junit4, testng, playwright, karate, cucumber |

**Examples:**
```bash
# Generate Python pytest automation
testteller automate test-cases.docx --language python --framework pytest --collection-name my_project

# Interactive TypeScript Playwright tests
testteller automate requirements.pdf --language typescript --framework playwright --interactive --enhance

# Java JUnit5 with comprehensive context
testteller automate test-scenarios.xlsx --language java --framework junit5 --num-context 10 --output-dir ./java_tests

# Enhanced generation with specific LLM
testteller automate api_tests.md --language python --framework playwright --enhance --llm-provider openai
```

**RAG Context Discovery:**
The automator performs 9 specialized queries to discover:
1. API endpoints and schemas
2. UI patterns and selectors  
3. Authentication flows
4. Data models and validation rules
5. Existing test patterns
6. Similar test implementations
7. Configuration patterns
8. Error handling patterns
9. Integration patterns

**Generated Output:**
- Production-ready test files with proper imports and setup
- Configuration files (pytest.ini, package.json, etc.)
- Dependency files (requirements.txt, package.json)
- README with setup and execution instructions
- Framework-specific optimizations and best practices

---

### `testteller status`
**Check collection and system status**

```bash
testteller status [OPTIONS]
```

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection to check

**Examples:**
```bash
# Check specific collection
testteller status --collection-name my_project

# Check default collection
testteller status
```

**Displays:**
- Collection document count
- ChromaDB connection information
- Storage path and configuration
- Recent ingestion activity
- System health indicators

---

### `testteller clear-data`
**Clear ingested data from collections**

```bash
testteller clear-data [OPTIONS]
```

**Options:**
- `--collection-name, -c TEXT`: ChromaDB collection to clear
- `--force, -f`: Skip confirmation prompt

**Examples:**
```bash
# Clear specific collection with confirmation
testteller clear-data --collection-name old_project

# Force clear without confirmation
testteller clear-data --collection-name temp_data --force

# Clear default collection
testteller clear-data
```

**Safety Features:**
- Confirmation prompts prevent accidental deletion
- Cleanup of temporary files and cloned repositories
- Preserves other collections

---

## Configuration

### Environment Variables

TestTeller uses environment variables for configuration. Run `testteller configure` to set these up interactively.

**LLM Provider Configuration:**
```bash
LLM_PROVIDER=gemini|openai|claude|llama
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key  
CLAUDE_API_KEY=your_claude_key
OLLAMA_BASE_URL=http://localhost:11434
```

**ChromaDB Configuration:**
```bash
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000
CHROMA_DB_USE_REMOTE=false
CHROMA_DB_PERSIST_DIRECTORY=./chroma_data
DEFAULT_COLLECTION_NAME=test_collection
```

**Document Processing:**
```bash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
OUTPUT_FILE_PATH=testteller-testcases.md
```

### Provider-Specific Setup

**Google Gemini (Recommended):**
```bash
testteller configure --provider gemini
# Only requires GOOGLE_API_KEY
```

**OpenAI:**
```bash
testteller configure --provider openai
# Only requires OPENAI_API_KEY
```

**Anthropic Claude:**
```bash
testteller configure --provider claude
# Requires CLAUDE_API_KEY + embedding provider (Google or OpenAI)
```

**Local Llama:**
```bash
# Install Ollama first
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2

testteller configure --provider llama
# No API key required, uses local Ollama
```

## Common Workflows

### Complete Test Generation Workflow

```bash
# 1. Configure TestTeller
testteller configure

# 2. Ingest project documentation
testteller ingest-docs ./requirements --collection-name my_project --enhanced

# 3. Ingest codebase for context
testteller ingest-code ./src --collection-name my_project

# 4. Generate comprehensive test cases
testteller generate "End-to-end user journey and API integration tests" --collection-name my_project --output-format pdf

# 5. Generate automation code
testteller automate testteller-testcases.pdf --language python --framework playwright --interactive --enhance

# 6. Check status
testteller status --collection-name my_project
```

### Quick API Testing Setup

```bash
# Ingest API documentation
testteller ingest-docs api-spec.yaml --collection-name api_project

# Generate API tests
testteller generate "REST API testing with authentication and error handling" --collection-name api_project

# Create Python automation
testteller automate testteller-testcases.pdf --language python --framework pytest --enhance
```

### Multi-Language Automation

```bash
# Generate tests once
testteller generate "User interface tests" --collection-name ui_project --output-file ui_tests.md

# Create multiple language implementations
testteller automate ui_tests.md --language python --framework playwright --output-dir ./python_tests
testteller automate ui_tests.md --language typescript --framework cypress --output-dir ./ts_tests  
testteller automate ui_tests.md --language java --framework junit5 --output-dir ./java_tests
```

## Tips and Best Practices

### Optimal Collection Organization
- Use descriptive collection names (e.g., `user_auth_system`, `payment_api`)
- Keep related documents in the same collection
- Separate different projects or features into different collections

### Document Ingestion Best Practices
- Use `--enhanced` flag for better parsing and retrieval
- Adjust `--chunk-size` based on document complexity (larger for detailed specs)
- Ingest both documentation and relevant code for comprehensive context

### Test Generation Tips
- Be specific in your generation queries
- Use `--num-retrieved` to control context breadth
- Try different output formats based on your workflow preferences

### Automation Generation Best Practices
- Use `--interactive` mode to review and select relevant test cases
- Enable `--enhance` for production-ready code quality
- Choose frameworks based on your team's expertise and project requirements
- Use `--verbose` for debugging context discovery issues

### Performance Optimization
- Start with smaller chunk sizes (500-800) for faster retrieval
- Use fewer context documents (3-5) for faster generation
- Enable enhanced parsing only when document structure is important