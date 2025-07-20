"""
Test automation prompts optimized for different LLM providers.
This module contains prompts specifically designed for improving test automation code generation
and enhancement across different programming languages and testing frameworks.
"""

from typing import List, Dict, Any

# Base prompt template for test automation enhancement
TEST_AUTOMATION_ENHANCEMENT_PROMPT = """
You are an expert test automation engineer with deep knowledge of {language} and {framework}.

TASK: Enhance and optimize the provided test automation code to improve:
1. Test reliability and stability
2. Code quality and maintainability  
3. Error handling and edge cases
4. Test data management
5. Assertion completeness

CONTEXT:
- Programming Language: {language}
- Testing Framework: {framework}
- Test Type: {test_type}
- Application Type: {app_type}

REQUIREMENTS:
- Follow {framework} best practices and conventions
- Include proper setup and teardown procedures
- Add comprehensive assertions and validations
- Implement robust error handling
- Use appropriate wait strategies and timeouts
- Generate realistic test data where needed
- Add meaningful comments and documentation

ORIGINAL TEST CODE:
{test_code}

ENHANCEMENT GUIDELINES:
{enhancement_guidelines}

IMPORTANT: Return ONLY the enhanced test code without any markdown formatting, explanations, or additional text. Do not wrap the code in ```python or any other markdown syntax. Return raw, executable code only.
"""

# Code generation prompt for creating new tests
TEST_CODE_GENERATION_PROMPT = """
You are an expert test automation engineer specializing in {language} and {framework}.

TASK: Generate comprehensive automated test code based on the provided test case specifications.

TEST SPECIFICATIONS:
{test_specifications}

REQUIREMENTS:
- Language: {language}
- Framework: {framework}
- Test Type: {test_type}
- Target Application: {app_type}

IMPLEMENTATION GUIDELINES:
1. Create complete, runnable test files
2. Include all necessary imports and dependencies
3. Implement proper test setup and cleanup
4. Add comprehensive assertions
5. Handle edge cases and error scenarios
6. Use realistic test data
7. Follow framework-specific patterns and conventions
8. Include configuration files if needed

FRAMEWORK-SPECIFIC REQUIREMENTS:
{framework_requirements}

IMPORTANT: Return ONLY the test code without any markdown formatting, explanations, or additional text. Do not wrap the code in ```python or any other markdown syntax. Return raw, executable code only.
"""

# Provider-specific prompt optimizations
PROVIDER_PROMPT_REFINEMENTS = {
    "gemini": {
        "style": "analytical",
        "focus": "code_structure", 
        "enhancement_guidelines": """
        - Leverage Google's best practices for test automation
        - Focus on clean, readable code structure
        - Emphasize component-based testing approaches
        - Integrate with modern CI/CD pipelines
        - Use data-driven testing patterns
        - Implement comprehensive logging and reporting
        """,
        "framework_requirements": """
        - Structure tests using modular, reusable components
        - Implement page object models for UI tests
        - Use dependency injection where applicable
        - Include performance benchmarking
        - Add accessibility testing considerations
        - Implement parallel execution capabilities
        """
    },
    "openai": {
        "style": "systematic",
        "focus": "json_structure",
        "enhancement_guidelines": """
        - Apply systematic testing methodologies
        - Implement comprehensive test coverage
        - Use structured test data and fixtures
        - Follow industry-standard patterns
        - Emphasize maintainable test architecture
        - Include detailed test documentation
        """,
        "framework_requirements": """
        - Use configuration-driven test execution
        - Implement JSON-based test data management
        - Structure tests with clear separation of concerns
        - Include API contract testing
        - Add comprehensive test reporting
        - Implement test result analytics
        """
    },
    "claude": {
        "style": "thoughtful",
        "focus": "context_awareness",
        "enhancement_guidelines": """
        - Consider the full testing context and user journey
        - Implement realistic test scenarios
        - Focus on user experience validation
        - Include accessibility and usability testing
        - Emphasize test maintainability over time
        - Consider business impact of test failures
        """,
        "framework_requirements": """
        - Design tests that reflect real user workflows
        - Implement context-aware test data generation
        - Include cross-browser and cross-platform testing
        - Add visual regression testing capabilities
        - Implement test environment management
        - Include test impact analysis
        """
    },
    "llama": {
        "style": "practical",
        "focus": "simplicity",
        "enhancement_guidelines": """
        - Keep code simple and straightforward
        - Focus on essential functionality
        - Use clear, descriptive naming
        - Implement basic but reliable patterns
        - Avoid over-engineering
        - Prioritize test execution speed
        """,
        "framework_requirements": """
        - Use simple, proven testing patterns
        - Implement basic setup and teardown
        - Focus on core functionality testing
        - Use straightforward assertion patterns
        - Keep dependencies minimal
        - Implement basic error handling
        """
    }
}

# Framework-specific enhancement patterns
FRAMEWORK_ENHANCEMENT_PATTERNS = {
    "pytest": {
        "imports": "import pytest\nfrom pytest import fixture, mark, param",
        "patterns": [
            "Use pytest fixtures for test data and setup",
            "Implement parametrized tests for data-driven testing", 
            "Use pytest markers for test categorization",
            "Add custom assertions with detailed error messages",
            "Implement test teardown with proper cleanup"
        ],
        "best_practices": [
            "Use conftest.py for shared fixtures",
            "Implement proper test isolation",
            "Add comprehensive logging",
            "Use pytest-html for reporting",
            "Implement parallel execution with pytest-xdist"
        ]
    },
    "unittest": {
        "imports": "import unittest\nfrom unittest.mock import Mock, patch, MagicMock",
        "patterns": [
            "Use setUp and tearDown methods consistently",
            "Implement test suites for organization",
            "Use mock objects for external dependencies",
            "Add descriptive test method names",
            "Implement custom assertion methods"
        ],
        "best_practices": [
            "Use TestCase inheritance properly",
            "Implement test discovery patterns",
            "Add comprehensive assertions",
            "Use subTest for related test cases",
            "Implement proper exception testing"
        ]
    },
    "playwright": {
        "imports": "from playwright.sync_api import Page, Browser, BrowserContext, expect",
        "patterns": [
            "Use page object model for element organization",
            "Implement auto-waiting for elements",
            "Use locator strategies for robust element selection",
            "Add visual testing and screenshots",
            "Implement network request interception"
        ],
        "best_practices": [
            "Use expect for assertions with auto-retry",
            "Implement browser context isolation",
            "Add mobile and tablet testing",
            "Use trace and debugging features",
            "Implement parallel test execution"
        ]
    },
    "jest": {
        "imports": "import { describe, test, expect, beforeEach, afterEach } from '@jest/globals'",
        "patterns": [
            "Use describe blocks for test organization",
            "Implement beforeEach/afterEach for setup/cleanup",
            "Use Jest matchers for comprehensive assertions",
            "Add snapshot testing for UI components",
            "Implement mock functions and modules"
        ],
        "best_practices": [
            "Use Jest configuration for project settings",
            "Implement code coverage reporting",
            "Add watch mode for development",
            "Use custom matchers for domain-specific assertions",
            "Implement test environment setup"
        ]
    },
    "cypress": {
        "imports": "/// <reference types=\"cypress\" />",
        "patterns": [
            "Use cy commands for browser interactions",
            "Implement custom commands for reusability",
            "Use cy.intercept for API mocking",
            "Add visual testing with screenshots",
            "Implement database seeding and cleanup"
        ],
        "best_practices": [
            "Use page object model with Cypress",
            "Implement proper wait strategies",
            "Add comprehensive API testing",
            "Use Cypress dashboard for CI/CD",
            "Implement test data management"
        ]
    },
    "junit5": {
        "imports": "import org.junit.jupiter.api.*;\nimport static org.junit.jupiter.api.Assertions.*;",
        "patterns": [
            "Use @Test annotation for test methods",
            "Implement @BeforeEach/@AfterEach for setup",
            "Use @ParameterizedTest for data-driven testing",
            "Add @DisplayName for descriptive test names",
            "Implement @TestMethodOrder for execution order"
        ],
        "best_practices": [
            "Use assertion messages for better debugging",
            "Implement test instance lifecycle management",
            "Add comprehensive exception testing",
            "Use @Nested for test organization",
            "Implement custom assertions"
        ]
    },
    "testng": {
        "imports": "import org.testng.annotations.*;\nimport static org.testng.Assert.*;",
        "patterns": [
            "Use @Test annotation with attributes",
            "Implement @BeforeMethod/@AfterMethod for setup",
            "Use @DataProvider for test data",
            "Add test groups for categorization",
            "Implement test dependencies"
        ],
        "best_practices": [
            "Use TestNG XML for test configuration",
            "Implement parallel test execution",
            "Add comprehensive reporting",
            "Use soft assertions for multiple validations",
            "Implement retry mechanisms"
        ]
    },
    "cucumber": {
        "imports": "from behave import given, when, then\nfrom selenium import webdriver",
        "patterns": [
            "Write clear Gherkin scenarios",
            "Implement step definitions with proper mapping",
            "Use scenario outlines for data-driven testing",
            "Add background steps for common setup",
            "Implement hooks for setup and cleanup"
        ],
        "best_practices": [
            "Keep scenarios focused and atomic",
            "Use business language in step definitions",
            "Implement proper context management",
            "Add comprehensive error handling",
            "Use tags for test organization"
        ]
    }
}

# Language-specific optimizations
LANGUAGE_OPTIMIZATIONS = {
    "python": {
        "imports": "import logging\nfrom typing import Dict, List, Optional, Any",
        "patterns": [
            "Use type hints for better code documentation",
            "Implement proper exception handling",
            "Use context managers for resource management",
            "Add comprehensive logging",
            "Use dataclasses for test data structures"
        ],
        "conventions": [
            "Follow PEP 8 style guidelines",
            "Use descriptive variable and function names",
            "Implement proper docstrings",
            "Use list comprehensions appropriately",
            "Handle None values explicitly"
        ]
    },
    "javascript": {
        "imports": "const axios = require('axios');\nconst { faker } = require('@faker-js/faker');",
        "patterns": [
            "Use async/await for asynchronous operations",
            "Implement proper error handling with try/catch",
            "Use destructuring for cleaner code",
            "Add JSDoc comments for documentation",
            "Use arrow functions appropriately"
        ],
        "conventions": [
            "Follow consistent naming conventions",
            "Use const/let instead of var",
            "Implement proper module exports",
            "Use template literals for string formatting",
            "Handle promises correctly"
        ]
    },
    "typescript": {
        "imports": "import axios from 'axios';\nimport { faker } from '@faker-js/faker';",
        "patterns": [
            "Use TypeScript interfaces and types",
            "Implement proper generic types",
            "Use enum for constants",
            "Add strict null checks",
            "Use union and intersection types"
        ],
        "conventions": [
            "Follow TypeScript style guidelines",
            "Use proper type annotations",
            "Implement interface segregation",
            "Use readonly for immutable data",
            "Handle optional properties correctly"
        ]
    },
    "java": {
        "imports": "import java.util.*;\nimport java.util.concurrent.*;",
        "patterns": [
            "Use proper access modifiers",
            "Implement builder patterns for complex objects",
            "Use streams for data processing",
            "Add proper exception handling",
            "Use annotations appropriately"
        ],
        "conventions": [
            "Follow Java naming conventions",
            "Use proper package organization",
            "Implement equals and hashCode correctly",
            "Use generics for type safety",
            "Handle resources with try-with-resources"
        ]
    }
}

def get_enhancement_prompt(provider: str, language: str, framework: str, test_code: str, 
                         test_type: str = "functional", app_type: str = "web application") -> str:
    """
    Get LLM-optimized prompt for test code enhancement.
    
    Args:
        provider: LLM provider (gemini, openai, claude, llama)
        language: Programming language (python, javascript, typescript, java)
        framework: Testing framework (pytest, jest, playwright, etc.)
        test_code: Original test code to enhance
        test_type: Type of test (functional, integration, e2e)
        app_type: Type of application being tested
        
    Returns:
        Optimized prompt string for the specific provider
    """
    base_prompt = TEST_AUTOMATION_ENHANCEMENT_PROMPT
    
    # Get provider-specific refinements
    provider_config = PROVIDER_PROMPT_REFINEMENTS.get(provider, PROVIDER_PROMPT_REFINEMENTS["llama"])
    
    # Get framework-specific patterns
    framework_patterns = FRAMEWORK_ENHANCEMENT_PATTERNS.get(framework, {})
    
    # Build enhancement guidelines
    enhancement_guidelines = provider_config["enhancement_guidelines"]
    if framework_patterns.get("best_practices"):
        enhancement_guidelines += "\n\nFramework-specific practices:\n"
        for practice in framework_patterns["best_practices"]:
            enhancement_guidelines += f"- {practice}\n"
    
    # Format the prompt
    formatted_prompt = base_prompt.format(
        language=language,
        framework=framework,
        test_type=test_type,
        app_type=app_type,
        test_code=test_code,
        enhancement_guidelines=enhancement_guidelines
    )
    
    return formatted_prompt

def get_generation_prompt(provider: str, language: str, framework: str, test_specifications: str,
                         test_type: str = "functional", app_type: str = "web application") -> str:
    """
    Get LLM-optimized prompt for test code generation.
    
    Args:
        provider: LLM provider (gemini, openai, claude, llama)
        language: Programming language (python, javascript, typescript, java)
        framework: Testing framework (pytest, jest, playwright, etc.)
        test_specifications: Test case specifications or requirements
        test_type: Type of test (functional, integration, e2e)
        app_type: Type of application being tested
        
    Returns:
        Optimized prompt string for the specific provider
    """
    base_prompt = TEST_CODE_GENERATION_PROMPT
    
    # Get provider-specific refinements
    provider_config = PROVIDER_PROMPT_REFINEMENTS.get(provider, PROVIDER_PROMPT_REFINEMENTS["llama"])
    
    # Get framework-specific requirements
    framework_requirements = provider_config["framework_requirements"]
    framework_patterns = FRAMEWORK_ENHANCEMENT_PATTERNS.get(framework, {})
    
    if framework_patterns.get("patterns"):
        framework_requirements += "\n\nFramework-specific patterns:\n"
        for pattern in framework_patterns["patterns"]:
            framework_requirements += f"- {pattern}\n"
    
    # Format the prompt
    formatted_prompt = base_prompt.format(
        language=language,
        framework=framework,
        test_type=test_type,
        app_type=app_type,
        test_specifications=test_specifications,
        framework_requirements=framework_requirements
    )
    
    return formatted_prompt

def get_test_optimization_suggestions(provider: str, language: str, framework: str) -> List[str]:
    """
    Get provider and framework-specific optimization suggestions.
    
    Args:
        provider: LLM provider
        language: Programming language
        framework: Testing framework
        
    Returns:
        List of optimization suggestions
    """
    suggestions = []
    
    # Provider-specific suggestions
    provider_config = PROVIDER_PROMPT_REFINEMENTS.get(provider, PROVIDER_PROMPT_REFINEMENTS["llama"])
    
    # Framework-specific suggestions
    framework_patterns = FRAMEWORK_ENHANCEMENT_PATTERNS.get(framework, {})
    if framework_patterns.get("best_practices"):
        suggestions.extend(framework_patterns["best_practices"])
    
    # Language-specific suggestions
    language_opts = LANGUAGE_OPTIMIZATIONS.get(language, {})
    if language_opts.get("conventions"):
        suggestions.extend(language_opts["conventions"])
    
    return suggestions

def get_framework_imports(language: str, framework: str) -> str:
    """
    Get framework-specific import statements.
    
    Args:
        language: Programming language
        framework: Testing framework
        
    Returns:
        Import statements string
    """
    framework_patterns = FRAMEWORK_ENHANCEMENT_PATTERNS.get(framework, {})
    language_opts = LANGUAGE_OPTIMIZATIONS.get(language, {})
    
    imports = []
    
    if framework_patterns.get("imports"):
        imports.append(framework_patterns["imports"])
    
    if language_opts.get("imports"):
        imports.append(language_opts["imports"])
    
    return "\n".join(imports)

# Test data generation prompts
TEST_DATA_GENERATION_PROMPT = """
Generate realistic test data for {test_type} testing in {language} using {framework}.

Requirements:
- Create diverse, realistic test data
- Include edge cases and boundary values
- Use appropriate data types for {language}
- Format data according to {framework} conventions
- Include both positive and negative test cases

Data should cover:
{data_requirements}

Generate the test data in the appropriate format for {language} and {framework}.
"""

def get_test_data_prompt(provider: str, language: str, framework: str, 
                        test_type: str, data_requirements: List[str]) -> str:
    """
    Get prompt for test data generation.
    
    Args:
        provider: LLM provider
        language: Programming language  
        framework: Testing framework
        test_type: Type of test
        data_requirements: List of data requirements
        
    Returns:
        Test data generation prompt
    """
    requirements_text = "\n".join(f"- {req}" for req in data_requirements)
    
    return TEST_DATA_GENERATION_PROMPT.format(
        test_type=test_type,
        language=language,
        framework=framework,
        data_requirements=requirements_text
    )