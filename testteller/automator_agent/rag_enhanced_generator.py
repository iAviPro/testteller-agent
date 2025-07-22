"""
RAG-Enhanced Test Generator for creating complete, working test automation code.

This module leverages vector store knowledge to generate real, functional test code
instead of template-based skeletons with TODOs.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .base_generator import BaseTestGenerator
from .application_context import ApplicationKnowledgeExtractor, ApplicationContext
from .parser.markdown_parser import TestCase
from ..core.vector_store.chromadb_manager import ChromaDBManager
from ..core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class RAGEnhancedTestGenerator(BaseTestGenerator):
    """Test generator that uses RAG to create complete, working test code."""
    
    def __init__(self, framework: str, output_dir: Path, vector_store: ChromaDBManager,
                 language: str = 'python', llm_manager: Optional[LLMManager] = None,
                 num_context_docs: int = 5):
        super().__init__(framework, output_dir)
        self.language = language
        self.vector_store = vector_store
        self.llm_manager = llm_manager or LLMManager()
        self.num_context_docs = num_context_docs
        self.knowledge_extractor = ApplicationKnowledgeExtractor(
            vector_store, self.llm_manager, num_context_docs
        )
        self.validator = TestCodeValidator(vector_store, self.llm_manager)
        
    async def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate complete test code using RAG-enhanced approach."""
        logger.info(f"Starting RAG-enhanced generation for {len(test_cases)} test cases")
        
        try:
            # 1. Extract comprehensive application context from vector store
            app_context = self.knowledge_extractor.extract_app_context(test_cases)
            logger.info("Application context extraction completed")
            
            # 2. Generate working test files
            generated_files = {}
            automation_results = {
                "success": True,
                "files": {},
                "errors": [],
                "test_case_ids": []
            }
            
            # Categorize tests for better organization
            categorized_tests = self.categorize_tests(test_cases)
            
            for category, tests in categorized_tests.items():
                if tests:
                    logger.info(f"Generating {len(tests)} {category} tests")
                    
                    # Generate complete test file for this category
                    test_file_content = self._generate_complete_test_file(
                        category, tests, app_context
                    )
                    
                    # Validate and fix the generated code
                    validated_content = self._validate_and_fix_code(
                        test_file_content, app_context
                    )
                    
                    file_name = f"test_{category}{self.get_file_extension()}"
                    generated_files[file_name] = validated_content
            
            # 3. Generate supporting files with real context
            supporting_files = self._generate_supporting_files(app_context)
            generated_files.update(supporting_files)
            
            logger.info(f"Generated {len(generated_files)} complete test files")
            
            # Track automation results for generated test cases
            automation_results["success"] = True
            automation_results["files"] = generated_files
            automation_results["test_case_ids"] = [tc.id for tc in test_cases]
            
            # Update stored test case metadata with automation success
            await self._enrich_test_case_metadata(test_cases, automation_results)
            
            return generated_files
            
        except Exception as e:
            logger.error(f"RAG-enhanced generation failed: {e}", exc_info=True)
            
            # Track automation failure
            automation_results["success"] = False
            automation_results["errors"].append(str(e))
            automation_results["test_case_ids"] = [tc.id for tc in test_cases]
            
            # Update stored test case metadata with automation failure
            await self._enrich_test_case_metadata(test_cases, automation_results)
            
            # Fallback to basic generation if RAG fails
            return self._generate_fallback_files(test_cases)
    
    def get_supported_frameworks(self) -> List[str]:
        """Return supported frameworks for the language."""
        framework_map = {
            'python': ['pytest', 'unittest', 'playwright'],
            'javascript': ['jest', 'mocha', 'cypress', 'playwright'],
            'typescript': ['jest', 'playwright', 'cypress'],
            'java': ['junit', 'testng']
        }
        return framework_map.get(self.language, ['pytest'])
    
    def get_file_extension(self) -> str:
        """Return file extension based on language."""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java'
        }
        return extensions.get(self.language, '.py')
    
    def _generate_complete_test_file(self, category: str, test_cases: List[TestCase],
                                   app_context: ApplicationContext) -> str:
        """Generate a complete test file for a category using RAG context."""
        
        # Find similar test implementations for learning
        similar_tests = self._find_similar_test_implementations(test_cases)
        
        # Build comprehensive prompt with real application context
        prompt = self._build_generation_prompt(category, test_cases, app_context, similar_tests)
        
        # Generate complete test file
        try:
            generated_code = self.llm_manager.generate_text(prompt)
            
            # Clean up any markdown formatting
            cleaned_code = self._clean_generated_code(generated_code)
            
            # Ensure proper imports and structure
            complete_code = self._ensure_proper_structure(cleaned_code, category, app_context)
            
            return complete_code
            
        except Exception as e:
            logger.error(f"Failed to generate complete test file for {category}: {e}")
            return self._generate_minimal_working_file(category, test_cases, app_context)
    
    def _build_generation_prompt(self, category: str, test_cases: List[TestCase],
                               app_context: ApplicationContext, similar_tests: List[str]) -> str:
        """Build comprehensive prompt for test generation."""
        
        # Extract test case details
        test_details = []
        for tc in test_cases:
            details = {
                'id': tc.id,
                'objective': tc.objective,
                'steps': [{'action': step.action, 'validation': step.validation} 
                         for step in tc.test_steps if step.action or step.validation],
                'feature': tc.feature,
                'type': tc.type
            }
            test_details.append(details)
        
        # Build context sections
        api_context = self._format_api_context(app_context.api_endpoints)
        ui_context = self._format_ui_context(app_context.ui_selectors)
        auth_context = self._format_auth_context(app_context.auth_patterns)
        data_context = self._format_data_context(app_context.data_schemas)
        
        prompt = f"""
Generate a COMPLETE, WORKING {self.framework} test file in {self.language} for {category} tests.

=== TEST CASES TO IMPLEMENT ===
{json.dumps(test_details, indent=2)}

=== REAL APPLICATION CONTEXT (from codebase analysis) ===

API ENDPOINTS:
{api_context}

UI SELECTORS & PATTERNS:
{ui_context}

AUTHENTICATION:
{auth_context}

DATA SCHEMAS:
{data_context}

BASE URL: {app_context.base_url or 'http://localhost:8000'}

=== EXISTING TEST PATTERNS FOR REFERENCE ===
{chr(10).join(similar_tests[:3]) if similar_tests else 'None found'}

=== STRICT REQUIREMENTS ===

1. **USE ONLY REAL APPLICATION DATA**: Use ONLY the endpoints, selectors, and schemas provided above
2. **COMPLETE IMPLEMENTATION**: NO TODO comments, NO placeholders - implement everything
3. **PROPER {self.framework.upper()} STRUCTURE**: Follow {self.framework} best practices and conventions
4. **REALISTIC TEST DATA**: Generate realistic test data using the provided schemas
5. **ERROR HANDLING**: Include proper error handling and meaningful assertions
6. **AUTHENTICATION**: Implement proper authentication using the discovered patterns
7. **SETUP/TEARDOWN**: Include proper test setup and cleanup
8. **IMPORTS**: Include all necessary imports for {self.language}

=== FRAMEWORK-SPECIFIC INSTRUCTIONS ===

{self._get_framework_specific_instructions()}

=== OUTPUT FORMAT ===

Generate ONLY the complete {self.language} test file code. Start with imports and end with the last test function.
Do not include explanations, markdown formatting, or any text outside the code.

"""
        return prompt
    
    def _format_api_context(self, endpoints: Dict[str, Any]) -> str:
        """Format API endpoints for the prompt."""
        if not endpoints:
            return "No API endpoints discovered"
        
        formatted = []
        for key, endpoint in endpoints.items():
            formatted.append(f"  {endpoint.method} {endpoint.path}")
            if hasattr(endpoint, 'description') and endpoint.description:
                formatted.append(f"    Description: {endpoint.description}")
            if hasattr(endpoint, 'auth_required') and endpoint.auth_required:
                formatted.append(f"    Requires Authentication: Yes")
        
        return '\n'.join(formatted)
    
    def _format_ui_context(self, selectors: Dict[str, Any]) -> str:
        """Format UI selectors for the prompt."""
        if not selectors:
            return "No UI selectors discovered"
        
        formatted = []
        for selector, pattern in selectors.items():
            formatted.append(f"  {selector} ({pattern.element_type})")
            if hasattr(pattern, 'description') and pattern.description:
                formatted.append(f"    Context: {pattern.description}")
        
        return '\n'.join(formatted)
    
    def _format_auth_context(self, auth_pattern) -> str:
        """Format authentication context for the prompt."""
        if not auth_pattern:
            return "No authentication pattern discovered"
        
        info = [f"  Type: {auth_pattern.auth_type}"]
        if auth_pattern.login_endpoint:
            info.append(f"  Login Endpoint: {auth_pattern.login_endpoint}")
        if auth_pattern.token_header:
            info.append(f"  Token Header: {auth_pattern.token_header}")
        if auth_pattern.login_selectors:
            info.append(f"  Login Selectors: {auth_pattern.login_selectors}")
        
        return '\n'.join(info)
    
    def _format_data_context(self, schemas: Dict[str, Any]) -> str:
        """Format data schemas for the prompt."""
        if not schemas:
            return "No data schemas discovered"
        
        formatted = []
        for name, schema in schemas.items():
            formatted.append(f"  {name}:")
            for field, field_type in schema.fields.items():
                formatted.append(f"    {field}: {field_type}")
        
        return '\n'.join(formatted)
    
    def _get_framework_specific_instructions(self) -> str:
        """Get framework-specific generation instructions."""
        instructions = {
            'pytest': """
- Use pytest fixtures for setup/teardown
- Use assert statements for validation
- Include proper parametrization for data-driven tests
- Use pytest markers for test categorization
- Include proper error handling with pytest.raises when needed
            """,
            'unittest': """
- Inherit from unittest.TestCase
- Use setUp() and tearDown() methods
- Use self.assert* methods for validation
- Include proper test method naming (test_*)
- Use self.subTest() for parameterized tests
            """,
            'playwright': """
- Use page fixture for browser automation
- Use expect() for assertions with proper locators
- Include proper wait strategies
- Use page.goto() with the discovered base URL
- Include proper error handling for timeouts
            """,
            'jest': """
- Use describe() and it()/test() structure
- Use expect() for assertions
- Include proper async/await for promises
- Use beforeEach/afterEach for setup/teardown
- Include proper mocking with jest.mock()
            """,
            'cypress': """
- Use cy.* commands for interactions
- Use cy.get() with discovered selectors
- Include proper assertions with should()
- Use cy.intercept() for API mocking
- Include proper wait strategies
            """
        }
        
        return instructions.get(self.framework, "Follow standard testing best practices")
    
    def _find_similar_test_implementations(self, test_cases: List[TestCase]) -> List[str]:
        """Find similar test implementations from both original and generated tests."""
        similar_patterns = []
        
        try:
            # Build query from all test cases
            query_parts = []
            for tc in test_cases:
                query_parts.extend([
                    tc.feature or '',
                    tc.type or '',
                    tc.category or '',
                    ' '.join([step.action for step in tc.test_steps if step.action])
                ])
            
            query = ' '.join(filter(None, query_parts))
            
            # Query for similar test implementations (including generated ones)
            results = self.vector_store.query_similar(
                query_text=query,
                n_results=8,  # Increased to get more results including generated ones
                metadata_filter={
                    "$or": [
                        {
                            "type": "code",
                            "file_type": [".py", ".js", ".ts"],
                            "document_type": ["test_cases", "code"]
                        },
                        {
                            "type": "generated_test_case"
                        }
                    ]
                }
            )
            
            if results and results.get('documents') and results.get('metadatas'):
                # Process results with quality weighting
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                    
                    # For generated test cases, check quality score
                    if metadata.get('type') == 'generated_test_case':
                        quality_score = metadata.get('quality_score', 0.5)
                        if quality_score > 0.7:  # Only use high-quality generated tests
                            similar_patterns.append(doc)
                            logger.debug(f"Including generated test case with quality score {quality_score}")
                    elif self._is_test_code(doc):
                        similar_patterns.append(doc)
                        
        except Exception as e:
            logger.warning(f"Failed to find similar test implementations: {e}")
        
        return similar_patterns
    
    def _is_test_code(self, content: str) -> bool:
        """Check if content is actual test code."""
        test_indicators = [
            'def test_', 'it(', 'test(', 'describe(', 
            'class Test', '@Test', 'function test', 'cy.', 'page.', 'assert'
        ]
        return any(indicator in content for indicator in test_indicators)
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code from markdown formatting and artifacts."""
        # Remove markdown code blocks
        code = re.sub(r'^```\w*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```$', '', code, flags=re.MULTILINE)
        
        # Remove explanatory text that might be at the beginning or end
        lines = code.strip().split('\n')
        
        # Find first line that looks like actual code
        start_idx = 0
        for i, line in enumerate(lines):
            if (line.strip().startswith(('import ', 'from ', 'def ', 'class ', '@'))
                or line.strip().startswith(('const ', 'let ', 'var ', 'function '))
                or line.strip().startswith(('package ', 'public class'))):
                start_idx = i
                break
        
        # Remove trailing explanatory text
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith(('#', '//', '/*', '*')):
                end_idx = i + 1
                break
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def _ensure_proper_structure(self, code: str, category: str, app_context: ApplicationContext) -> str:
        """Ensure the code has proper structure and imports."""
        try:
            # Add necessary imports if missing
            if self.language == 'python':
                return self._ensure_python_structure(code, category, app_context)
            elif self.language == 'javascript' or self.language == 'typescript':
                return self._ensure_js_structure(code, category, app_context)
            else:
                return code
                
        except Exception as e:
            logger.warning(f"Failed to ensure proper structure: {e}")
            return code
    
    def _ensure_python_structure(self, code: str, category: str, app_context: ApplicationContext) -> str:
        """Ensure proper Python test structure."""
        imports_needed = []
        
        # Framework-specific imports
        if self.framework == 'pytest':
            imports_needed.extend(['import pytest', 'import requests'])
        elif self.framework == 'unittest':
            imports_needed.extend(['import unittest', 'import requests'])
        elif self.framework == 'playwright':
            imports_needed.extend(['import pytest', 'from playwright.sync_api import Page, expect'])
        
        # Category-specific imports
        if category == 'e2e':
            if self.framework == 'playwright':
                pass  # Already added above
            else:
                imports_needed.extend([
                    'from selenium import webdriver',
                    'from selenium.webdriver.common.by import By',
                    'from selenium.webdriver.support.ui import WebDriverWait',
                    'from selenium.webdriver.support import expected_conditions as EC'
                ])
        
        # Add other common imports
        imports_needed.extend(['import json', 'import time', 'from typing import Dict, Any'])
        
        # Check what imports are already present
        existing_imports = set()
        for line in code.split('\n')[:20]:  # Check first 20 lines
            if line.strip().startswith(('import ', 'from ')):
                existing_imports.add(line.strip())
        
        # Add missing imports
        missing_imports = []
        for imp in imports_needed:
            if not any(imp.split()[1] in existing for existing in existing_imports):
                missing_imports.append(imp)
        
        if missing_imports:
            imports_section = '\n'.join(missing_imports) + '\n\n'
            # Insert after any existing imports or at the beginning
            code = imports_section + code
        
        return code
    
    def _ensure_js_structure(self, code: str, category: str, app_context: ApplicationContext) -> str:
        """Ensure proper JavaScript/TypeScript test structure."""
        # Similar logic for JS/TS imports
        return code
    
    def _validate_and_fix_code(self, code: str, app_context: ApplicationContext) -> str:
        """Validate generated code and fix common issues."""
        try:
            validation_result = self.validator.validate_generated_test(code, self.language)
            
            if not validation_result.is_valid:
                logger.warning(f"Generated code has {len(validation_result.issues)} issues, attempting fixes")
                fixed_code = self.validator.fix_validation_issues(code, validation_result.issues, app_context)
                return fixed_code
            
            return code
            
        except Exception as e:
            logger.warning(f"Code validation failed: {e}")
            return code
    
    def _generate_supporting_files(self, app_context: ApplicationContext) -> Dict[str, str]:
        """Generate supporting files like requirements.txt, config files."""
        supporting_files = {}
        
        try:
            if self.language == 'python':
                # Generate requirements.txt with real dependencies
                requirements = self._generate_python_requirements(app_context)
                supporting_files['requirements.txt'] = requirements
                
                # Generate pytest.ini or conftest.py if using pytest
                if self.framework == 'pytest':
                    conftest = self._generate_conftest_py(app_context)
                    supporting_files['conftest.py'] = conftest
                    
            elif self.language == 'javascript' or self.language == 'typescript':
                # Generate package.json dependencies section
                package_deps = self._generate_js_dependencies(app_context)
                supporting_files['package-dependencies.json'] = package_deps
                
        except Exception as e:
            logger.warning(f"Failed to generate supporting files: {e}")
        
        return supporting_files
    
    def _generate_python_requirements(self, app_context: ApplicationContext) -> str:
        """Generate requirements.txt based on actual usage."""
        requirements = []
        
        # Framework-specific requirements
        if self.framework == 'pytest':
            requirements.extend([
                'pytest>=7.0.0',
                'pytest-html>=3.0.0',
                'pytest-xdist>=3.0.0'
            ])
        elif self.framework == 'playwright':
            requirements.extend([
                'playwright>=1.40.0',
                'pytest-playwright>=0.4.0'
            ])
        else:
            requirements.append('unittest2>=1.1.0')
        
        # Common requirements
        requirements.extend([
            'requests>=2.28.0',
            'faker>=15.0.0'
        ])
        
        # Add requirements based on discovered context
        if app_context.api_endpoints:
            requirements.append('httpx>=0.24.0')
        
        if any('selenium' in pattern for pattern in app_context.existing_test_patterns):
            requirements.append('selenium>=4.0.0')
        
        return '\n'.join(sorted(set(requirements)))
    
    def _generate_conftest_py(self, app_context: ApplicationContext) -> str:
        """Generate conftest.py with real application context."""
        base_url = app_context.base_url or 'http://localhost:8000'
        
        conftest_content = f'''"""Test configuration and fixtures."""

import pytest
import requests
from typing import Dict, Any

@pytest.fixture(scope="session")
def base_url():
    """Base URL for the application."""
    return "{base_url}"

@pytest.fixture
def api_client(base_url):
    """HTTP client for API tests."""
    session = requests.Session()
    session.headers.update({{"Content-Type": "application/json"}})
    return session

@pytest.fixture
def test_data():
    """Test data based on discovered schemas."""
    return {{
        "user": {{
            "email": "test@example.com",
            "password": "testpassword123"
        }}
    }}
'''
        
        # Add authentication fixture if auth pattern discovered
        if app_context.auth_patterns:
            if app_context.auth_patterns.auth_type == 'jwt':
                conftest_content += '''
@pytest.fixture
def auth_token(api_client, base_url, test_data):
    """Get authentication token."""
    login_url = f"{base_url}{app_context.auth_patterns.login_endpoint or '/api/auth/login'}"
    response = api_client.post(login_url, json=test_data["user"])
    if response.status_code == 200:
        return response.json().get("token", response.json().get("access_token"))
    return None

@pytest.fixture
def authenticated_client(api_client, auth_token):
    """HTTP client with authentication."""
    if auth_token:
        api_client.headers.update({"Authorization": f"Bearer {auth_token}"})
    return api_client
'''
        
        return conftest_content
    
    def _generate_js_dependencies(self, app_context: ApplicationContext) -> str:
        """Generate JavaScript/TypeScript dependencies."""
        deps = {}
        
        if self.framework == 'jest':
            deps.update({
                'jest': '^29.0.0',
                '@types/jest': '^29.0.0'
            })
        elif self.framework == 'playwright':
            deps.update({
                '@playwright/test': '^1.40.0'
            })
        
        return json.dumps(deps, indent=2)
    
    def _generate_minimal_working_file(self, category: str, test_cases: List[TestCase], 
                                     app_context: ApplicationContext) -> str:
        """Generate minimal but working test file as fallback."""
        if self.language == 'python' and self.framework == 'pytest':
            return f'''import pytest
import requests

@pytest.fixture
def base_url():
    return "{app_context.base_url or 'http://localhost:8000'}"

def test_{category}_basic(base_url):
    """Basic {category} test."""
    # TODO: Implement specific test logic
    assert base_url is not None
'''
        
        return "# Fallback test file - manual implementation needed\n"
    
    def _generate_fallback_files(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate basic fallback files if RAG enhancement fails."""
        logger.warning("Falling back to basic generation due to RAG failure")
        
        fallback_files = {}
        categorized = self.categorize_tests(test_cases)
        
        for category, tests in categorized.items():
            if tests:
                fallback_content = self._generate_minimal_working_file(
                    category, tests, ApplicationContext()
                )
                fallback_files[f"test_{category}{self.get_file_extension()}"] = fallback_content
        
        return fallback_files
    
    async def _enrich_test_case_metadata(self, test_cases: List[TestCase], automation_results: Dict[str, Any]) -> None:
        """
        Enrich stored test case metadata with automation success information.
        
        Args:
            test_cases: List of test cases that were automated
            automation_results: Results of the automation process
        """
        try:
            from datetime import datetime
            import os
            
            # Only enrich if feedback is enabled
            if not os.getenv('ENABLE_TEST_CASE_FEEDBACK', 'true').lower() == 'true':
                return
                
            logger.info(f"Enriching metadata for {len(test_cases)} test cases with automation results")
            
            # Search for generated test cases that match these test cases
            for test_case in test_cases:
                try:
                    # Search for stored test cases that might correspond to this automated test case
                    results = self.vector_store.query_similar(
                        query_text=f"{test_case.objective} {test_case.feature} {test_case.type}",
                        n_results=5,
                        metadata_filter={
                            "type": "generated_test_case"
                        }
                    )
                    
                    if results and results.get('documents') and results.get('metadatas'):
                        # Find the best matching stored test case
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results['metadatas'][0][i]
                            
                            # Simple matching: if the stored test case contains similar keywords
                            if self._is_matching_test_case(test_case, doc, metadata):
                                # Get the document ID to update metadata
                                doc_id = results['ids'][0][i] if results.get('ids') else None
                                
                                if doc_id:
                                    # Prepare enrichment metadata
                                    enrichment_metadata = {
                                        "automation_attempted": True,
                                        "automation_success": automation_results["success"],
                                        "automation_timestamp": datetime.now().isoformat(),
                                        "automation_language": self.language,
                                        "automation_framework": self.framework,
                                        "generated_files": list(automation_results["files"].keys()) if automation_results["success"] else [],
                                        "automation_errors": automation_results.get("errors", []),
                                        "practical_validation": "passed" if automation_results["success"] else "failed"
                                    }
                                    
                                    # Update the metadata in vector store
                                    # Note: ChromaDB doesn't have direct metadata update, so we'll log this for now
                                    # In a production system, you might want to implement a metadata update mechanism
                                    logger.info(
                                        f"Test case '{test_case.id}' automation result: "
                                        f"{'SUCCESS' if automation_results['success'] else 'FAILED'} "
                                        f"(Language: {self.language}, Framework: {self.framework})"
                                    )
                                    
                                    # Future enhancement: store enriched metadata
                                    # This could be implemented as a separate metadata collection or
                                    # by re-adding the document with updated metadata
                                    
                                break
                                
                except Exception as e:
                    logger.warning(f"Failed to enrich metadata for test case {test_case.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to enrich test case metadata: {e}")
    
    def _is_matching_test_case(self, test_case: TestCase, stored_doc: str, stored_metadata: Dict[str, Any]) -> bool:
        """
        Check if a test case matches a stored document.
        
        Args:
            test_case: The test case being automated
            stored_doc: The stored document content
            stored_metadata: The stored document metadata
            
        Returns:
            True if they match, False otherwise
        """
        # Simple matching based on shared keywords
        test_case_text = f"{test_case.objective} {test_case.feature} {test_case.type}".lower()
        stored_text = stored_doc.lower()
        
        # Calculate word overlap
        test_words = set(test_case_text.split())
        stored_words = set(stored_text.split())
        
        if len(test_words) == 0 or len(stored_words) == 0:
            return False
            
        overlap = test_words.intersection(stored_words)
        similarity_score = len(overlap) / min(len(test_words), len(stored_words))
        
        # Consider it a match if >50% of words overlap
        return similarity_score > 0.5


class TestCodeValidator:
    """Validates and fixes generated test code."""
    __test__ = False  # Tell pytest this is not a test class
    
    def __init__(self, vector_store: ChromaDBManager, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager
    
    def validate_generated_test(self, test_code: str, language: str) -> 'ValidationResult':
        """Validate generated test code for common issues."""
        issues = []
        
        try:
            # 1. Basic syntax check
            if not self._has_valid_syntax(test_code, language):
                issues.append("Invalid syntax detected")
            
            # 2. Check for TODO placeholders
            if 'TODO' in test_code or 'FIXME' in test_code:
                issues.append("Contains TODO/FIXME placeholders")
            
            # 3. Check for required imports
            if not self._has_required_imports(test_code, language):
                issues.append("Missing required imports")
            
            # 4. Check for proper test structure
            if not self._has_proper_test_structure(test_code, language):
                issues.append("Invalid test structure")
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            issues.append(f"Validation error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence_score=max(0.0, 1.0 - (len(issues) * 0.2))
        )
    
    def fix_validation_issues(self, test_code: str, issues: List[str], 
                            app_context: ApplicationContext) -> str:
        """Fix validation issues in the generated code."""
        try:
            if not issues:
                return test_code
            
            fix_prompt = f"""
Fix the following issues in this test code:

ISSUES TO FIX:
{chr(10).join(f"- {issue}" for issue in issues)}

ORIGINAL CODE:
{test_code}

APPLICATION CONTEXT:
- Base URL: {app_context.base_url}
- Available endpoints: {list(app_context.api_endpoints.keys())[:5]}
- Available selectors: {list(app_context.ui_selectors.keys())[:5]}

REQUIREMENTS:
1. Fix all identified issues
2. Maintain the original test logic
3. Use only real application context provided
4. Return ONLY the fixed code, no explanations

FIXED CODE:
"""
            
            fixed_code = self.llm_manager.generate_text(fix_prompt)
            return self._clean_fixed_code(fixed_code)
            
        except Exception as e:
            logger.warning(f"Failed to fix validation issues: {e}")
            return test_code
    
    def _has_valid_syntax(self, code: str, language: str) -> bool:
        """Check if code has valid syntax."""
        try:
            if language == 'python':
                import ast
                ast.parse(code)
                return True
        except:
            pass
        return False
    
    def _has_required_imports(self, code: str, language: str) -> bool:
        """Check if code has required imports."""
        if language == 'python':
            # Check for basic imports
            return any(keyword in code for keyword in ['import ', 'from '])
        return True
    
    def _has_proper_test_structure(self, code: str, language: str) -> bool:
        """Check if code has proper test structure."""
        if language == 'python':
            return 'def test_' in code or 'class Test' in code
        elif language == 'javascript':
            return any(keyword in code for keyword in ['test(', 'it(', 'describe('])
        return True
    
    def _clean_fixed_code(self, code: str) -> str:
        """Clean fixed code from LLM response."""
        # Remove markdown formatting
        code = re.sub(r'^```\w*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code)
        return code.strip()


class ValidationResult:
    """Result of code validation."""
    
    def __init__(self, is_valid: bool, issues: List[str], confidence_score: float = 0.0):
        self.is_valid = is_valid
        self.issues = issues
        self.confidence_score = confidence_score