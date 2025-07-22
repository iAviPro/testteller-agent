# TestTeller RAG-Enhanced Automation

Generate complete, working test automation code using RAG-enhanced approach with vector store knowledge.

## Overview

The automation command creates functional test code that runs without manual modification by leveraging:

- **Real Application Context**: Extracted from your vector store (APIs, UI patterns, auth flows, data schemas)
- **Complete Implementations**: No TODO placeholders - every function is fully implemented
- **Quality Validation**: Built-in syntax checking and application context verification
- **Framework Best Practices**: Language and framework-specific patterns applied automatically

## Quick Start

```bash
# Basic usage (uses configuration from testteller configure)
testteller automate test_cases.md

# With specific parameters
testteller automate test_cases.md -c my_collection -l python -F pytest

# Interactive mode
testteller automate test_cases.md --interactive

# Verbose output for debugging
testteller automate test_cases.md --verbose
```

## Command Parameters

Following the same pattern as `testteller generate`:

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--collection-name` | `-c` | ChromaDB collection for app context | From config |
| `--language` | `-l` | Programming language | python |
| `--framework` | `-F` | Test framework | pytest |
| `--output-dir` | `-o` | Output directory | ./testteller_automated_tests |
| `--interactive` | `-i` | Interactive test selection | False |
| `--num-context` | `-n` | Context documents to retrieve | 5 |
| `--verbose` | `-v` | Enable verbose logging | False |

## Configuration

All settings are read from your `.env` file created by `testteller configure`:

```bash
# Set up your configuration first
testteller configure

# Then automation uses the same settings
testteller automate test_cases.md
```

### Environment Variables

- `DEFAULT_COLLECTION_NAME` - Default vector store collection
- `CHROMA_DB_PERSIST_DIRECTORY` - Vector store location  
- `LLM_PROVIDER` - AI provider (openai, gemini, claude, llama)
- `AUTOMATION_LANGUAGE` - Default programming language
- `AUTOMATION_FRAMEWORK` - Default test framework

## Supported Languages & Frameworks

| Language | Frameworks |
|----------|------------|
| Python | pytest, unittest, playwright |
| JavaScript | jest, mocha, cypress, playwright |
| TypeScript | jest, playwright, cypress |
| Java | junit, testng |

## Generated Output

### Example Structure
```
testteller_automated_tests/
├── test_e2e.py          # End-to-end browser tests
├── test_integration.py  # API integration tests  
├── test_technical.py    # Performance/load tests
├── requirements.txt     # Real dependencies
├── conftest.py         # pytest fixtures with app context
└── README.md           # Usage instructions
```

### Quality Metrics

The system reports quality scores based on:
- **90%+**: Excellent - runs with minimal changes
- **70-89%**: Good - minor updates needed  
- **50-69%**: Fair - some manual work required
- **<50%**: Needs work - significant implementation needed

## Example Generated Code

Instead of templates with TODOs:

```python
def test_user_login(page, base_url, test_data):
    """Test user authentication with real application context."""
    
    # Real endpoint from your API documentation
    page.goto(f"{base_url}/auth/login")
    
    # Real selectors from your component code
    page.fill('input[data-testid="email-field"]', test_data["user"]["email"])
    page.fill('input[data-testid="password-field"]', test_data["user"]["password"])
    page.click('button[data-testid="login-submit"]')
    
    # Real success indicators from your application
    page.wait_for_url('**/dashboard')
    expect(page.locator('[data-testid="user-menu"]')).to_be_visible()
```

## Running Generated Tests

### Python
```bash
cd testteller_automated_tests
pip install -r requirements.txt
pytest --verbose
```

### JavaScript/TypeScript  
```bash
cd testteller_automated_tests
npm install
npm test
```

## Troubleshooting

### Vector Store Issues
- Ensure your vector store contains application documentation and code
- Check collection name matches your ingested data
- Verify LLM provider is configured and accessible

### Quality Issues
- Review selectors against your actual application
- Validate API endpoints match your current implementation
- Update test data to match your data schemas

## Best Practices

1. **Prepare Vector Store**: Include API docs, existing tests, and component definitions
2. **Clear Test Cases**: Write detailed objectives and expected outcomes
3. **Review Generated Code**: Verify selectors and endpoints match your app
4. **Iterative Improvement**: Add successful tests back to vector store for learning