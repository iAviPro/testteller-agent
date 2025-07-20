"""Python test code generator supporting pytest and unittest frameworks."""

from typing import List, Dict, Optional
import json
import textwrap
from pathlib import Path

from .base_generator import BaseTestGenerator
from ..parser.markdown_parser import TestCase, TestStep
from testteller.core.constants import SUPPORTED_FRAMEWORKS


class PythonTestGenerator(BaseTestGenerator):
    """Generator for Python test code."""
    
    SUPPORTED_FRAMEWORKS = SUPPORTED_FRAMEWORKS['python']
    
    def __init__(self, framework: str, output_dir: Path):
        super().__init__(framework, output_dir)
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {framework}. Supported: {self.SUPPORTED_FRAMEWORKS}")
    
    def get_supported_frameworks(self) -> List[str]:
        return self.SUPPORTED_FRAMEWORKS
    
    def get_file_extension(self) -> str:
        return '.py'
    
    def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate Python test files from test cases."""
        generated_files = {}
        
        # Categorize tests
        categorized = self.categorize_tests(test_cases)
        
        # Generate files for each category
        for category, cases in categorized.items():
            if cases:
                file_name = f"test_{category}{self.get_file_extension()}"
                content = self._generate_test_file(category, cases)
                generated_files[file_name] = content
        
        # Generate conftest.py for pytest
        if self.framework == 'pytest':
            generated_files['conftest.py'] = self._generate_conftest()
        
        # Generate requirements.txt
        generated_files['requirements.txt'] = self._generate_requirements()
        
        return generated_files
    
    def _generate_test_file(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate a test file for a specific category."""
        imports = self._generate_imports(category)
        
        if self.framework == 'pytest':
            test_content = self._generate_pytest_tests(test_cases)
        elif self.framework == 'unittest':
            test_content = self._generate_unittest_tests(test_cases)
        elif self.framework == 'playwright':
            test_content = self._generate_playwright_tests(test_cases)
        elif self.framework == 'cucumber':
            test_content = self._generate_cucumber_tests(test_cases)
        
        return f"{imports}\n\n{test_content}"
    
    def _generate_imports(self, category: str) -> str:
        """Generate import statements based on test category."""
        imports = []
        
        if self.framework == 'pytest':
            imports.extend([
                "import pytest",
                "import requests",
                "from unittest.mock import Mock, patch",
            ])
            
            if category == 'e2e':
                imports.extend([
                    "from selenium import webdriver",
                    "from selenium.webdriver.common.by import By",
                    "from selenium.webdriver.support.ui import WebDriverWait",
                    "from selenium.webdriver.support import expected_conditions as EC",
                ])
            elif category == 'integration':
                imports.extend([
                    "import json",
                    "from typing import Dict, Any",
                ])
            elif category == 'technical':
                imports.extend([
                    "import time",
                    "import threading",
                    "from concurrent.futures import ThreadPoolExecutor",
                ])
        elif self.framework == 'playwright':
            imports.extend([
                "import pytest",
                "from playwright.sync_api import Page, Browser, BrowserContext, expect",
                "import requests",
                "from faker import Faker",
            ])
            
            if category == 'e2e':
                imports.extend([
                    "from playwright.sync_api import TimeoutError as PlaywrightTimeoutError",
                    "import json",
                ])
            elif category == 'integration':
                imports.extend([
                    "import json",
                    "from typing import Dict, Any",
                ])
        else:
            imports.extend([
                "import unittest",
                "from unittest.mock import Mock, patch",
                "import requests",
            ])
        
        return '\n'.join(imports)
    
    def _generate_pytest_tests(self, test_cases: List[TestCase]) -> str:
        """Generate pytest-style tests."""
        test_functions = []
        
        for test_case in test_cases:
            test_name = f"test_{self.sanitize_test_name(test_case.id)}"
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_test_body(test_case)
            
            function = f'''
def {test_name}():
    """
    {textwrap.indent(test_doc, '    ')}
    """
{textwrap.indent(test_body, '    ')}
'''
            test_functions.append(function)
        
        # Add fixtures if needed
        fixtures = self._generate_fixtures(test_cases)
        
        return f"{fixtures}\n\n{''.join(test_functions)}"
    
    def _generate_unittest_tests(self, test_cases: List[TestCase]) -> str:
        """Generate unittest-style tests."""
        test_methods = []
        
        for test_case in test_cases:
            test_name = f"test_{self.sanitize_test_name(test_case.id)}"
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_test_body(test_case, unittest_style=True)
            
            method = f'''
    def {test_name}(self):
        """
        {textwrap.indent(test_doc, '        ')}
        """
{textwrap.indent(test_body, '        ')}
'''
            test_methods.append(method)
        
        class_name = f"Test{self._get_category_class_name(test_cases[0].id)}"
        
        return f'''
class {class_name}(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
{''.join(test_methods)}


if __name__ == '__main__':
    unittest.main()
'''
    
    def _generate_playwright_tests(self, test_cases: List[TestCase]) -> str:
        """Generate Playwright-style tests."""
        test_functions = []
        
        for test_case in test_cases:
            test_name = f"test_{self.sanitize_test_name(test_case.id)}"
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_playwright_test_body(test_case)
            
            function = f'''
def {test_name}(page: Page):
    """
    {textwrap.indent(test_doc, '    ')}
    """
{textwrap.indent(test_body, '    ')}
'''
            test_functions.append(function)
        
        return '\n'.join(test_functions)
    
    def _generate_playwright_test_body(self, test_case: TestCase) -> str:
        """Generate the body of a Playwright test function."""
        body_lines = []
        faker = "fake = Faker()"
        body_lines.append(faker)
        body_lines.append("")
        
        # Setup section
        if test_case.prerequisites:
            body_lines.append("# Test setup")
            test_data = self.extract_test_data(test_case)
            for key, value in test_data.items():
                if isinstance(value, str):
                    body_lines.append(f'{key} = "{value}"')
                else:
                    body_lines.append(f'{key} = {value}')
            body_lines.append("")
        
        # Default base URL setup
        body_lines.append("# Navigate to application")
        body_lines.append("base_url = 'http://localhost:8000'")
        body_lines.append("page.goto(base_url)")
        body_lines.append("")
        
        # Test steps
        if test_case.test_steps:
            body_lines.append("# Test execution")
            for i, step in enumerate(test_case.test_steps, 1):
                if step.action:
                    body_lines.append(f"# Step {i}: {step.action}")
                    if step.technical_details:
                        body_lines.append(f"# Technical: {step.technical_details}")
                    body_lines.append(self._generate_playwright_step_code(step, test_case))
                    body_lines.append("")
                
                if step.validation:
                    body_lines.append(f"# Validation: {step.validation}")
                    if step.validation_details:
                        body_lines.append(f"# Details: {step.validation_details}")
                    body_lines.append(self._generate_playwright_assertion(step))
                    body_lines.append("")
        
        # Final assertions
        body_lines.append("# Final verification")
        if test_case.expected_outcome:
            if 'error' in test_case.expected_outcome.lower():
                body_lines.append("# TODO: Add error state verification")
            else:
                body_lines.append("expect(page.locator('body')).to_be_visible()")
        else:
            body_lines.append("expect(page.locator('body')).to_be_visible()")
        
        return '\n'.join(body_lines)
    
    def _generate_playwright_step_code(self, step: TestStep, test_case: TestCase) -> str:
        """Generate Playwright-specific step code."""
        action = step.action.lower()
        
        if 'navigate' in action or 'goto' in action or 'visit' in action:
            return "page.goto(base_url + '/path')"
        elif 'click' in action:
            if step.technical_details and 'button' in step.technical_details.lower():
                return "page.click('button[type=\"submit\"]')"
            else:
                return "page.click('selector')"
        elif 'fill' in action or 'enter' in action or 'input' in action:
            if step.technical_details and 'email' in step.technical_details.lower():
                return "page.fill('input[type=\"email\"]', fake.email())"
            elif step.technical_details and 'password' in step.technical_details.lower():
                return "page.fill('input[type=\"password\"]', 'TestPassword123!')"
            else:
                return "page.fill('input', fake.text(max_nb_chars=20))"
        elif 'select' in action:
            return "page.select_option('select', value='option_value')"
        elif 'wait' in action:
            return "page.wait_for_timeout(2000)"
        elif 'api' in action or 'request' in action:
            method = 'GET'
            if 'post' in action:
                method = 'POST'
            elif 'put' in action:
                method = 'PUT'
            elif 'delete' in action:
                method = 'DELETE'
            return f"response = requests.{method.lower()}(base_url + '/api/endpoint')"
        else:
            return f"# TODO: Implement {step.action}"
    
    def _generate_playwright_assertion(self, step: TestStep) -> str:
        """Generate Playwright-specific assertions."""
        validation = step.validation.lower()
        
        if 'visible' in validation or 'display' in validation:
            return "expect(page.locator('element')).to_be_visible()"
        elif 'text' in validation or 'content' in validation:
            return "expect(page.locator('element')).to_contain_text('expected text')"
        elif 'redirect' in validation or 'url' in validation:
            return "expect(page).to_have_url('expected_url')"
        elif 'error' in validation:
            return "expect(page.locator('.error')).to_be_visible()"
        elif 'success' in validation:
            return "expect(page.locator('.success')).to_be_visible()"
        elif 'status' in validation:
            if step.validation_details and '200' in step.validation_details:
                return "assert response.status_code == 200"
            else:
                return "assert response.status_code in [200, 201, 204]"
        else:
            return "expect(page.locator('element')).to_be_visible()"
    
    def _generate_test_body(self, test_case: TestCase, unittest_style: bool = False) -> str:
        """Generate the body of a test function."""
        body_lines = []
        
        # Setup section
        if test_case.prerequisites:
            body_lines.append("# Setup")
            test_data = self.extract_test_data(test_case)
            for key, value in test_data.items():
                if isinstance(value, str):
                    body_lines.append(f'{key} = "{value}"')
                else:
                    body_lines.append(f'{key} = {value}')
            
            if 'system_state' in test_case.prerequisites:
                body_lines.append(f'# System state: {test_case.prerequisites["system_state"]}')
            
            if 'mocked_services' in test_case.prerequisites:
                body_lines.append(f'# TODO: Mock services: {test_case.prerequisites["mocked_services"]}')
            
            body_lines.append("")
        
        # Test steps
        if test_case.test_steps:
            body_lines.append("# Test execution")
            for i, step in enumerate(test_case.test_steps, 1):
                if step.action:
                    body_lines.append(f"# Step {i}: {step.action}")
                    if step.technical_details:
                        body_lines.append(f"# Technical: {step.technical_details}")
                    body_lines.append(self._generate_step_code(step, test_case))
                    body_lines.append("")
                
                if step.validation:
                    body_lines.append(f"# Validation: {step.validation}")
                    if step.validation_details:
                        body_lines.append(f"# Technical: {step.validation_details}")
                    body_lines.append(self._generate_validation_code(step, unittest_style))
                    body_lines.append("")
        
        # Integration-specific handling
        if test_case.integration:
            body_lines.extend(self._generate_integration_test_body(test_case, unittest_style))
        
        # Technical test-specific handling
        if test_case.technical_area:
            body_lines.extend(self._generate_technical_test_body(test_case, unittest_style))
        
        # Expected final state assertions
        if test_case.expected_state:
            body_lines.append("# Verify final state")
            for state_type, expected in test_case.expected_state.items():
                body_lines.append(f"# {state_type}: {expected}")
                if unittest_style:
                    body_lines.append(f"# self.assert... # TODO: Implement {state_type} verification")
                else:
                    body_lines.append(f"# assert ... # TODO: Implement {state_type} verification")
            body_lines.append("")
        
        # Error scenario handling
        if test_case.error_scenario:
            body_lines.append("# Error handling")
            if 'condition' in test_case.error_scenario:
                body_lines.append(f"# Error condition: {test_case.error_scenario['condition']}")
            if 'recovery' in test_case.error_scenario:
                body_lines.append(f"# Expected recovery: {test_case.error_scenario['recovery']}")
            body_lines.append("# TODO: Implement error scenario testing")
        
        # If no specific implementation, add placeholder
        if not body_lines:
            if unittest_style:
                body_lines.append("self.fail('Test not implemented')")
            else:
                body_lines.append("pytest.fail('Test not implemented')")
        
        return '\n'.join(body_lines)
    
    def _generate_step_code(self, step: TestStep, test_case: TestCase) -> str:
        """Generate code for a test step."""
        # Analyze the step to determine the type of action
        action_lower = step.action.lower()
        
        if 'navigate' in action_lower or 'click' in action_lower or 'page' in action_lower:
            # Selenium action
            return "# TODO: Implement UI interaction"
        elif 'api' in action_lower or 'request' in action_lower or '/api/' in str(step.technical_details or ''):
            # API call
            if step.technical_details and ('PUT' in step.technical_details or 'POST' in step.technical_details):
                method = 'PUT' if 'PUT' in step.technical_details else 'POST'
                endpoint = self._extract_endpoint(step.technical_details)
                return f'''response = requests.{method.lower()}(
    f"{{base_url}}{endpoint}",
    headers={{"Authorization": f"Bearer {{token}}"}},
    json={{}}  # TODO: Add request payload
)'''
            else:
                return "# TODO: Implement API call"
        else:
            return "# TODO: Implement action"
    
    def _generate_validation_code(self, step: TestStep, unittest_style: bool) -> str:
        """Generate validation/assertion code."""
        if unittest_style:
            if 'success' in step.validation.lower():
                return "self.assertEqual(response.status_code, 200)"
            elif '200' in str(step.validation_details or ''):
                return "self.assertEqual(response.status_code, 200)"
            else:
                return "# self.assert... # TODO: Implement validation"
        else:
            if 'success' in step.validation.lower():
                return "assert response.status_code == 200"
            elif '200' in str(step.validation_details or ''):
                return "assert response.status_code == 200"
            else:
                return "# assert ... # TODO: Implement validation"
    
    def _generate_integration_test_body(self, test_case: TestCase, unittest_style: bool) -> List[str]:
        """Generate integration test specific code."""
        lines = []
        
        if test_case.technical_contract:
            lines.append("# Technical contract")
            if 'endpoint' in test_case.technical_contract:
                lines.append(f'endpoint = "{test_case.technical_contract["endpoint"]}"')
            if 'protocol' in test_case.technical_contract:
                lines.append(f'# Protocol: {test_case.technical_contract["protocol"]}')
            lines.append("")
        
        if test_case.request_payload:
            lines.append("# Request payload")
            lines.append(f"payload = {test_case.request_payload}")
            lines.append("")
        
        if test_case.expected_response:
            lines.append("# Expected response assertions")
            if 'status_code' in test_case.expected_response:
                status = test_case.expected_response['status_code']
                if unittest_style:
                    lines.append(f"# self.assertEqual(response.status_code, {self._extract_status_code(status)})")
                else:
                    lines.append(f"# assert response.status_code == {self._extract_status_code(status)}")
            lines.append("")
        
        return lines
    
    def _generate_technical_test_body(self, test_case: TestCase, unittest_style: bool) -> List[str]:
        """Generate technical test specific code."""
        lines = []
        
        if test_case.hypothesis:
            lines.append(f"# Hypothesis: {test_case.hypothesis}")
            lines.append("")
        
        if test_case.test_setup:
            lines.append("# Test setup")
            if 'targets' in test_case.test_setup:
                lines.append(f"# Target components: {test_case.test_setup['targets']}")
            if 'tooling' in test_case.test_setup:
                lines.append(f"# Tools: {test_case.test_setup['tooling']}")
            lines.append("# TODO: Implement technical test setup")
            lines.append("")
        
        return lines
    
    def _generate_fixtures(self, test_cases: List[TestCase]) -> str:
        """Generate pytest fixtures."""
        fixtures = []
        
        # Base URL fixture
        fixtures.append('''
@pytest.fixture
def base_url():
    """Base URL for API tests."""
    return "http://localhost:8000"
''')
        
        # Auth token fixture
        fixtures.append('''
@pytest.fixture
def auth_token():
    """Authentication token for API tests."""
    # TODO: Implement authentication
    return "test-token"
''')
        
        # Check if we need Selenium fixtures
        needs_selenium = any('navigate' in str(tc.test_steps).lower() for tc in test_cases)
        if needs_selenium:
            fixtures.append('''
@pytest.fixture
def driver():
    """Selenium WebDriver fixture."""
    driver = webdriver.Chrome()  # Or use other browsers
    driver.implicitly_wait(10)
    yield driver
    driver.quit()
''')
        
        return '\n'.join(fixtures)
    
    def _generate_conftest(self) -> str:
        """Generate conftest.py for pytest."""
        return '''"""Pytest configuration and shared fixtures."""

import pytest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "base_url": "http://localhost:8000",
        "timeout": 30,
        "retry_count": 3
    }
'''
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt file."""
        requirements = []
        
        if self.framework == 'pytest':
            requirements.extend([
                "pytest>=7.0.0",
                "pytest-html>=3.0.0",
                "pytest-xdist>=3.0.0",  # For parallel execution
                "pytest-timeout>=2.0.0",
            ])
        elif self.framework == 'playwright':
            requirements.extend([
                "playwright>=1.40.0",
                "pytest>=7.0.0",
                "pytest-playwright>=0.4.0",
                "pytest-html>=3.0.0",
            ])
        else:
            requirements.append("unittest2>=1.1.0")
        
        # Common requirements
        requirements.extend([
            "requests>=2.28.0",
            "selenium>=4.0.0",  # For E2E tests
            "faker>=15.0.0",  # For test data generation
        ])
        
        return '\n'.join(requirements)
    
    def _get_category_class_name(self, test_id: str) -> str:
        """Get class name based on test category."""
        if test_id.startswith('E2E_'):
            return "EndToEnd"
        elif test_id.startswith('INT_'):
            return "Integration"
        elif test_id.startswith('TECH_'):
            return "Technical"
        elif test_id.startswith('MOCK_'):
            return "Mocked"
        return "Tests"
    
    def _extract_endpoint(self, technical_details: str) -> str:
        """Extract endpoint from technical details."""
        # Simple extraction - look for /api/... pattern
        import re
        match = re.search(r'(/api/[^\s]+)', technical_details)
        if match:
            return match.group(1)
        return "/api/endpoint"
    
    def _extract_status_code(self, status_text: str) -> str:
        """Extract numeric status code from text."""
        import re
        match = re.search(r'(\d{3})', status_text)
        if match:
            return match.group(1)
        return "200"  # Default