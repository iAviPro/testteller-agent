"""
Constants and default values for the TestTeller.
This module centralizes all default values and configuration constants used throughout the application.
Includes constants from both testteller and testwriter packages.
"""

# Application Information
APP_NAME = "TestTeller"
# APP_VERSION is now imported from __init__.py to maintain single source of truth
# Import will be handled by __init__.py
FALLBACK_VERSION = "2.0.2"  # Fallback version when _version.py import fails
# Application Description - Single Source of Truth
APP_DESCRIPTION = "TestTeller: Your Next-Generation AI-Powered Test Agent for Comprehensive Test Case Generation and Test Automation leveraging RAG & GenAI"

# Short description for CLI and other contexts
APP_SHORT_DESCRIPTION = "Next-Generation AI-Powered Test Agent for Test Cases Generation and Test Automation"

# Default Environment Settings
DEFAULT_LOG_LEVEL = "ERROR"
DEFAULT_LOG_FORMAT = "json"

# ChromaDB Settings
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_CHROMA_USE_REMOTE = False
DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_data"
DEFAULT_COLLECTION_NAME = "test_collection"

# LLM Settings
# Supported LLM providers
SUPPORTED_LLM_PROVIDERS = ["gemini", "openai", "claude", "llama"]
DEFAULT_LLM_PROVIDER = "gemini"

# Gemini Settings
DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_GEMINI_GENERATION_MODEL = "gemini-2.0-flash"

# OpenAI Settings
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_GENERATION_MODEL = "gpt-4o-mini"

# Claude Settings
# Claude specific defaults
DEFAULT_CLAUDE_GENERATION_MODEL = "claude-3-5-haiku-20241022"  # Claude generation model
# Embedding provider for Claude (google, openai)
DEFAULT_CLAUDE_EMBEDDING_PROVIDER = "google"

# Llama Settings
DEFAULT_LLAMA_EMBEDDING_MODEL = "llama3.2:1b"
DEFAULT_LLAMA_GENERATION_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Document Processing Settings
DEFAULT_CHUNK_SIZE = 1000

# Feedback Loop Configuration
ENABLE_TEST_CASE_FEEDBACK = True
MIN_QUALITY_SCORE_FOR_STORAGE = 0.7
MAX_GENERATED_TESTS_TO_STORE = 1000
GENERATED_TEST_RETENTION_DAYS = 90
DEFAULT_CHUNK_OVERLAP = 200

# Code Processing Settings
DEFAULT_CODE_EXTENSIONS = [
    ".py",   # Python
    ".js",   # JavaScript
    ".ts",   # TypeScript
    ".java",  # Java
    ".go",   # Go
    ".rs",   # Rust
    ".cpp",  # C++
    ".hpp",  # C++ Headers
    ".c",    # C
    ".h",    # C Headers
    ".cs",   # C#
    ".rb",   # Ruby
    ".php"   # PHP
]
DEFAULT_TEMP_CLONE_DIR = "./temp_cloned_repos"

# Output Settings
DEFAULT_OUTPUT_FILE = "testteller-testcases.pdf"
DEFAULT_AUTOMATION_OUTPUT_DIR = "./testteller_automated_tests"
DEFAULT_TEST_GENERATION_DIR = "./testteller_generated_tests"

# Test Output Format Settings
SUPPORTED_TEST_OUTPUT_FORMATS = ["md", "pdf", "docx"]
DEFAULT_TEST_OUTPUT_FORMAT = "pdf"

# Test Automation Settings - Enhanced with new frameworks
SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "java"]
SUPPORTED_FRAMEWORKS = {
    "python": ["pytest", "unittest", "playwright", "cucumber"],
    "javascript": ["jest", "mocha", "playwright", "cypress", "cucumber"],
    "typescript": ["jest", "mocha", "playwright", "cypress", "cucumber"],
    "java": ["junit5", "junit4", "testng", "playwright", "karate", "cucumber"]
}

# Default automation configuration
DEFAULT_AUTOMATION_LANGUAGE = "python"
DEFAULT_AUTOMATION_FRAMEWORK = "pytest"

# Test Framework Dependencies and Versions
FRAMEWORK_DEPENDENCIES = {
    "python": {
        "pytest": {
            "requirements": [
                "pytest>=7.0.0",
                "pytest-html>=3.0.0",
                "pytest-xdist>=3.0.0",
                "pytest-timeout>=2.0.0",
                "requests>=2.28.0",
                "selenium>=4.0.0",
                "faker>=15.0.0"
            ]
        },
        "unittest": {
            "requirements": [
                "unittest2>=1.1.0",
                "requests>=2.28.0",
                "selenium>=4.0.0",
                "faker>=15.0.0"
            ]
        },
        "playwright": {
            "requirements": [
                "playwright>=1.40.0",
                "pytest>=7.0.0",
                "pytest-playwright>=0.4.0",
                "pytest-html>=3.0.0",
                "requests>=2.28.0",
                "faker>=15.0.0"
            ]
        },
        "cucumber": {
            "requirements": [
                "behave>=1.2.6",
                "requests>=2.28.0",
                "selenium>=4.0.0",
                "faker>=15.0.0"
            ]
        }
    },
    "javascript": {
        "jest": {
            "devDependencies": {
                "jest": "^29.0.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "mocha": {
            "devDependencies": {
                "mocha": "^10.0.0",
                "chai": "^4.3.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "playwright": {
            "devDependencies": {
                "@playwright/test": "^1.40.0",
                "axios": "^1.5.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "cypress": {
            "devDependencies": {
                "cypress": "^13.0.0",
                "axios": "^1.5.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "cucumber": {
            "devDependencies": {
                "@cucumber/cucumber": "^10.0.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        }
    },
    "typescript": {
        "jest": {
            "devDependencies": {
                "jest": "^29.0.0",
                "@types/jest": "^29.0.0",
                "ts-jest": "^29.0.0",
                "typescript": "^5.0.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "mocha": {
            "devDependencies": {
                "mocha": "^10.0.0",
                "chai": "^4.3.0",
                "@types/mocha": "^10.0.0",
                "@types/chai": "^4.3.0",
                "ts-node": "^10.0.0",
                "typescript": "^5.0.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "playwright": {
            "devDependencies": {
                "@playwright/test": "^1.40.0",
                "typescript": "^5.0.0",
                "axios": "^1.5.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "cypress": {
            "devDependencies": {
                "cypress": "^13.0.0",
                "typescript": "^5.0.0",
                "axios": "^1.5.0",
                "@faker-js/faker": "^8.0.0"
            }
        },
        "cucumber": {
            "devDependencies": {
                "@cucumber/cucumber": "^10.0.0",
                "typescript": "^5.0.0",
                "ts-node": "^10.0.0",
                "axios": "^1.5.0",
                "puppeteer": "^21.0.0",
                "@faker-js/faker": "^8.0.0"
            }
        }
    },
    "java": {
        "junit5": {
            "maven_dependencies": [
                {
                    "groupId": "org.junit.jupiter",
                    "artifactId": "junit-jupiter-engine",
                    "version": "5.9.0"
                },
                {
                    "groupId": "org.junit.jupiter",
                    "artifactId": "junit-jupiter-api",
                    "version": "5.9.0"
                },
                {
                    "groupId": "org.seleniumhq.selenium",
                    "artifactId": "selenium-java",
                    "version": "4.15.0"
                },
                {
                    "groupId": "com.github.javafaker",
                    "artifactId": "javafaker",
                    "version": "1.0.2"
                }
            ]
        },
        "junit4": {
            "maven_dependencies": [
                {
                    "groupId": "junit",
                    "artifactId": "junit",
                    "version": "4.13.2"
                },
                {
                    "groupId": "org.seleniumhq.selenium",
                    "artifactId": "selenium-java",
                    "version": "4.15.0"
                },
                {
                    "groupId": "com.github.javafaker",
                    "artifactId": "javafaker",
                    "version": "1.0.2"
                }
            ]
        },
        "testng": {
            "maven_dependencies": [
                {
                    "groupId": "org.testng",
                    "artifactId": "testng",
                    "version": "7.8.0"
                },
                {
                    "groupId": "org.seleniumhq.selenium",
                    "artifactId": "selenium-java",
                    "version": "4.15.0"
                },
                {
                    "groupId": "com.github.javafaker",
                    "artifactId": "javafaker",
                    "version": "1.0.2"
                }
            ]
        },
        "playwright": {
            "maven_dependencies": [
                {
                    "groupId": "com.microsoft.playwright",
                    "artifactId": "playwright",
                    "version": "1.40.0"
                },
                {
                    "groupId": "org.junit.jupiter",
                    "artifactId": "junit-jupiter-engine",
                    "version": "5.9.0"
                },
                {
                    "groupId": "com.github.javafaker",
                    "artifactId": "javafaker",
                    "version": "1.0.2"
                }
            ]
        },
        "karate": {
            "maven_dependencies": [
                {
                    "groupId": "com.intuit.karate",
                    "artifactId": "karate-junit5",
                    "version": "1.4.1"
                },
                {
                    "groupId": "org.junit.jupiter",
                    "artifactId": "junit-jupiter-engine",
                    "version": "5.9.0"
                }
            ]
        },
        "cucumber": {
            "maven_dependencies": [
                {
                    "groupId": "io.cucumber",
                    "artifactId": "cucumber-java",
                    "version": "7.14.0"
                },
                {
                    "groupId": "io.cucumber",
                    "artifactId": "cucumber-junit",
                    "version": "7.14.0"
                },
                {
                    "groupId": "org.seleniumhq.selenium",
                    "artifactId": "selenium-java",
                    "version": "4.15.0"
                },
                {
                    "groupId": "com.github.javafaker",
                    "artifactId": "javafaker",
                    "version": "1.0.2"
                }
            ]
        }
    }
}

# Default Base URLs for different languages
DEFAULT_BASE_URLS = {
    "python": "http://localhost:8000",
    "javascript": "http://localhost:3000",
    "typescript": "http://localhost:3000",
    "java": "http://localhost:8080"
}

# Default Timeout Values (in milliseconds)
DEFAULT_TIMEOUTS = {
    "test_timeout": 30000,  # 30 seconds
    "api_timeout": 10000,   # 10 seconds
    "browser_timeout": 60000  # 60 seconds for browser operations
}

# Default HTTP Status Codes
DEFAULT_SUCCESS_STATUS = 200
DEFAULT_CREATED_STATUS = 201
DEFAULT_NO_CONTENT_STATUS = 204

# File Extensions and Naming Patterns
FILE_EXTENSIONS = {
    "python": {
        "test": ".py",
        "feature": ".feature"
    },
    "javascript": {
        "test": [".test.js", ".spec.js"],
        "cypress": ".cy.js",
        "feature": ".feature"
    },
    "typescript": {
        "test": [".test.ts", ".spec.ts"],
        "cypress": ".cy.ts",
        "feature": ".feature"
    },
    "java": {
        "test": ".java",
        "feature": ".feature"
    }
}

# Maven Configuration Defaults
MAVEN_DEFAULTS = {
    "groupId": "com.testteller",
    "artifactId": "generated-tests",
    "version": "1.0.0",
    "javaVersion": "11",
    "mavenCompilerVersion": "3.11.0",
    "surefireVersion": "3.0.0"
}

# Package.json Default Values
PACKAGE_JSON_DEFAULTS = {
    "version": "1.0.0",
    "name": "generated-tests",
    "description": "Generated test suite by TestTeller Agent",
    "author": "TestTeller Agent"
}

# TestWriter Package Information
TESTWRITER_VERSION = "0.1.0"
TESTWRITER_DESCRIPTION = "Test automation code generator for TestTeller Agent"

# API Retry Settings
DEFAULT_API_RETRY_ATTEMPTS = 3
DEFAULT_API_RETRY_WAIT_SECONDS = 2

# Docker Settings
DOCKER_HEALTHCHECK_INTERVAL = "30s"
DOCKER_HEALTHCHECK_TIMEOUT = "10s"
DOCKER_HEALTHCHECK_RETRIES = 3
DOCKER_HEALTHCHECK_START_PERIOD = "30s"
DOCKER_DEFAULT_CPU_LIMIT = "2"
DOCKER_DEFAULT_MEMORY_LIMIT = "4G"
DOCKER_DEFAULT_CPU_RESERVATION = "0.5"
DOCKER_DEFAULT_MEMORY_RESERVATION = "1G"

# Environment Variable Names
# API Keys
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_CLAUDE_API_KEY = "CLAUDE_API_KEY"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

# LLM Configuration
ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_CHROMA_DB_HOST = "CHROMA_DB_HOST"
ENV_CHROMA_DB_PORT = "CHROMA_DB_PORT"
ENV_CHROMA_DB_USE_REMOTE = "CHROMA_DB_USE_REMOTE"
ENV_CHROMA_DB_PERSIST_DIRECTORY = "CHROMA_DB_PERSIST_DIRECTORY"
ENV_DEFAULT_COLLECTION_NAME = "DEFAULT_COLLECTION_NAME"

# Gemini Model Environment Variables
ENV_GEMINI_EMBEDDING_MODEL = "GEMINI_EMBEDDING_MODEL"
ENV_GEMINI_GENERATION_MODEL = "GEMINI_GENERATION_MODEL"

# OpenAI Model Environment Variables
ENV_OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
ENV_OPENAI_GENERATION_MODEL = "OPENAI_GENERATION_MODEL"

# Claude Model Environment Variables
ENV_CLAUDE_GENERATION_MODEL = "CLAUDE_GENERATION_MODEL"
ENV_CLAUDE_EMBEDDING_PROVIDER = "CLAUDE_EMBEDDING_PROVIDER"

# Llama/Ollama Model Environment Variables
ENV_LLAMA_EMBEDDING_MODEL = "LLAMA_EMBEDDING_MODEL"
ENV_LLAMA_GENERATION_MODEL = "LLAMA_GENERATION_MODEL"
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"

# Other Environment Variables
ENV_CHUNK_SIZE = "CHUNK_SIZE"
ENV_CHUNK_OVERLAP = "CHUNK_OVERLAP"
ENV_CODE_EXTENSIONS = "CODE_EXTENSIONS"
ENV_TEMP_CLONE_DIR_BASE = "TEMP_CLONE_DIR_BASE"
ENV_OUTPUT_FILE_PATH = "OUTPUT_FILE_PATH"
ENV_TEST_OUTPUT_FORMAT = "TEST_OUTPUT_FORMAT"
ENV_API_RETRY_ATTEMPTS = "API_RETRY_ATTEMPTS"
ENV_API_RETRY_WAIT_SECONDS = "API_RETRY_WAIT_SECONDS"

# Automation Environment Variables
ENV_AUTOMATION_LANGUAGE = "AUTOMATION_LANGUAGE"
ENV_AUTOMATION_FRAMEWORK = "AUTOMATION_FRAMEWORK"
ENV_AUTOMATION_OUTPUT_DIR = "AUTOMATION_OUTPUT_DIR"
ENV_BASE_URL = "BASE_URL"
ENV_TEST_TIMEOUT = "TEST_TIMEOUT"
