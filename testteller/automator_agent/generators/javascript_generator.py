"""JavaScript test code generator supporting Jest and Mocha frameworks."""

from typing import List, Dict
import json
import textwrap
from pathlib import Path

from .base_generator import BaseTestGenerator
from ..parser.markdown_parser import TestCase, TestStep
from testteller.core.constants import SUPPORTED_FRAMEWORKS


class JavaScriptTestGenerator(BaseTestGenerator):
    """Generator for JavaScript test code."""
    
    SUPPORTED_FRAMEWORKS = SUPPORTED_FRAMEWORKS['javascript']
    
    def __init__(self, framework: str, output_dir: Path):
        super().__init__(framework, output_dir)
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {framework}. Supported: {self.SUPPORTED_FRAMEWORKS}")
    
    def get_supported_frameworks(self) -> List[str]:
        return self.SUPPORTED_FRAMEWORKS
    
    def get_file_extension(self) -> str:
        if self.framework == 'playwright':
            return '.spec.js'
        elif self.framework == 'cypress':
            return '.cy.js'
        return '.test.js'
    
    def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate JavaScript test files from test cases."""
        generated_files = {}
        
        # Categorize tests
        categorized = self.categorize_tests(test_cases)
        
        # Generate files for each category
        for category, cases in categorized.items():
            if cases:
                file_name = f"{category}{self.get_file_extension()}"
                content = self._generate_test_file(category, cases)
                generated_files[file_name] = content
        
        # Generate package.json
        generated_files['package.json'] = self._generate_package_json()
        
        # Generate config files
        if self.framework == 'jest':
            generated_files['jest.config.js'] = self._generate_jest_config()
        elif self.framework == 'playwright':
            generated_files['playwright.config.js'] = self._generate_playwright_config()
        elif self.framework == 'cypress':
            generated_files['cypress.config.js'] = self._generate_cypress_config()
        
        return generated_files
    
    def _generate_test_file(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate a test file for a specific category."""
        imports = self._generate_imports(category)
        test_content = self._generate_test_content(category, test_cases)
        
        return f"{imports}\n\n{test_content}"
    
    def _generate_imports(self, category: str) -> str:
        """Generate import statements based on test category and framework."""
        imports = []
        
        if self.framework == 'jest':
            imports.extend([
                "const axios = require('axios');",
                "const { expect } = require('@jest/globals');"
            ])
            if category == 'e2e':
                imports.append("const puppeteer = require('puppeteer');")
        
        elif self.framework == 'mocha':
            imports.extend([
                "const { expect } = require('chai');",
                "const axios = require('axios');",
                "const sinon = require('sinon');"
            ])
        
        elif self.framework == 'playwright':
            imports.extend([
                "const { test, expect } = require('@playwright/test');",
                "const axios = require('axios');"
            ])
        
        elif self.framework == 'cypress':
            imports.append("// Cypress imports are global")
        
        return '\n'.join(imports)
    
    def _generate_test_content(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate test content based on framework."""
        if self.framework == 'jest':
            return self._generate_jest_tests(category, test_cases)
        elif self.framework == 'mocha':
            return self._generate_mocha_tests(category, test_cases)
        elif self.framework == 'playwright':
            return self._generate_playwright_tests(category, test_cases)
        elif self.framework == 'cypress':
            return self._generate_cypress_tests(category, test_cases)
    
    def _generate_jest_tests(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate Jest test suite."""
        suite_name = self._get_suite_name(category)
        test_blocks = []
        
        for test_case in test_cases:
            test_name = self._generate_test_name(test_case)
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_test_body(test_case)
            
            test_block = f'''
  test('{test_name}', async () => {{
    /*
     * {test_doc.replace(chr(10), chr(10) + '     * ')}
     */
{textwrap.indent(test_body, '    ')}
  }});
'''
            test_blocks.append(test_block)
        
        return f'''describe('{suite_name}', () => {{
  beforeAll(async () => {{
    // Global setup
  }});

  afterAll(async () => {{
    // Global cleanup
  }});

  beforeEach(async () => {{
    // Test setup
  }});

  afterEach(async () => {{
    // Test cleanup
  }});

{''.join(test_blocks)}}});'''
    
    def _generate_mocha_tests(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate Mocha test suite."""
        suite_name = self._get_suite_name(category)
        test_blocks = []
        
        for test_case in test_cases:
            test_name = self._generate_test_name(test_case)
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_test_body(test_case)
            
            test_block = f'''
  it('{test_name}', async function() {{
    /*
     * {test_doc.replace(chr(10), chr(10) + '     * ')}
     */
{textwrap.indent(test_body, '    ')}
  }});
'''
            test_blocks.append(test_block)
        
        return f'''describe('{suite_name}', function() {{
  this.timeout(30000); // 30 second timeout

  before(async function() {{
    // Global setup
  }});

  after(async function() {{
    // Global cleanup
  }});

  beforeEach(async function() {{
    // Test setup
  }});

  afterEach(async function() {{
    // Test cleanup
  }});

{''.join(test_blocks)}}});'''
    
    def _generate_playwright_tests(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate Playwright test suite."""
        test_blocks = []
        
        for test_case in test_cases:
            test_name = self._generate_test_name(test_case)
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_playwright_test_body(test_case)
            
            test_block = f'''
test('{test_name}', async ({{ page }}) => {{
  /*
   * {test_doc.replace(chr(10), chr(10) + '   * ')}
   */
{textwrap.indent(test_body, '  ')}
}});
'''
            test_blocks.append(test_block)
        
        return '\n'.join(test_blocks)
    
    def _generate_cypress_tests(self, category: str, test_cases: List[TestCase]) -> str:
        """Generate Cypress test suite."""
        suite_name = self._get_suite_name(category)
        test_blocks = []
        
        for test_case in test_cases:
            test_name = self._generate_test_name(test_case)
            test_doc = self.generate_test_description(test_case)
            test_body = self._generate_cypress_test_body(test_case)
            
            test_block = f'''
  it('{test_name}', () => {{
    /*
     * {test_doc.replace(chr(10), chr(10) + '     * ')}
     */
{textwrap.indent(test_body, '    ')}
  }});
'''
            test_blocks.append(test_block)
        
        return f'''describe('{suite_name}', () => {{
  beforeEach(() => {{
    // Visit base URL or login
    cy.visit('/');
  }});

{''.join(test_blocks)}}});'''
    
    def _generate_test_body(self, test_case: TestCase) -> str:
        """Generate generic test body for Jest/Mocha."""
        body_lines = []
        
        # Setup
        if test_case.prerequisites:
            body_lines.append("// Setup")
            test_data = self.extract_test_data(test_case)
            for key, value in test_data.items():
                if isinstance(value, str):
                    body_lines.append(f'const {key} = "{value}";')
                else:
                    body_lines.append(f'const {key} = {json.dumps(value)};')
            body_lines.append("")
        
        # Test steps
        if test_case.test_steps:
            body_lines.append("// Test execution")
            for i, step in enumerate(test_case.test_steps, 1):
                if step.action:
                    body_lines.append(f"// Step {i}: {step.action}")
                    if step.technical_details:
                        body_lines.append(f"// Technical: {step.technical_details}")
                    body_lines.append(self._generate_step_code(step, test_case))
                    body_lines.append("")
                
                if step.validation:
                    body_lines.append(f"// Validation: {step.validation}")
                    body_lines.append(self._generate_validation_code(step))
                    body_lines.append("")
        
        # Integration specific
        if test_case.integration:
            body_lines.extend(self._generate_integration_test_body(test_case))
        
        # Expected state
        if test_case.expected_state:
            body_lines.append("// Verify final state")
            for state_type, expected in test_case.expected_state.items():
                body_lines.append(f"// {state_type}: {expected}")
                body_lines.append("// TODO: Implement state verification")
            body_lines.append("")
        
        # Default if empty
        if not body_lines:
            body_lines.append("throw new Error('Test not implemented');")
        
        return '\n'.join(body_lines)
    
    def _generate_playwright_test_body(self, test_case: TestCase) -> str:
        """Generate Playwright-specific test body."""
        body_lines = []
        
        # Navigate to page
        body_lines.append("// Navigate to application")
        body_lines.append("await page.goto('http://localhost:3000');")
        body_lines.append("")
        
        # Test steps
        if test_case.test_steps:
            for i, step in enumerate(test_case.test_steps, 1):
                if step.action:
                    body_lines.append(f"// Step {i}: {step.action}")
                    if 'click' in step.action.lower():
                        body_lines.append("// await page.click('selector');")
                    elif 'navigate' in step.action.lower():
                        body_lines.append("// await page.goto('url');")
                    else:
                        body_lines.append("// TODO: Implement action")
                    body_lines.append("")
                
                if step.validation:
                    body_lines.append(f"// Validation: {step.validation}")
                    body_lines.append("// await expect(page.locator('selector')).toBeVisible();")
                    body_lines.append("")
        
        return '\n'.join(body_lines)
    
    def _generate_cypress_test_body(self, test_case: TestCase) -> str:
        """Generate Cypress-specific test body."""
        body_lines = []
        
        # Test steps
        if test_case.test_steps:
            for i, step in enumerate(test_case.test_steps, 1):
                if step.action:
                    body_lines.append(f"// Step {i}: {step.action}")
                    if 'click' in step.action.lower():
                        body_lines.append("// cy.get('selector').click();")
                    elif 'navigate' in step.action.lower():
                        body_lines.append("// cy.visit('/path');")
                    else:
                        body_lines.append("// TODO: Implement action")
                    body_lines.append("")
                
                if step.validation:
                    body_lines.append(f"// Validation: {step.validation}")
                    body_lines.append("// cy.get('selector').should('be.visible');")
                    body_lines.append("")
        
        return '\n'.join(body_lines)
    
    def _generate_step_code(self, step: TestStep, test_case: TestCase) -> str:
        """Generate code for a test step."""
        action_lower = step.action.lower()
        
        if 'api' in action_lower or 'request' in action_lower:
            if step.technical_details and 'PUT' in step.technical_details:
                return '''const response = await axios.put(
  `${baseUrl}/api/endpoint`,
  {}, // TODO: Add request body
  { headers: { Authorization: `Bearer ${token}` } }
);'''
            else:
                return "// TODO: Implement API call"
        else:
            return "// TODO: Implement action"
    
    def _generate_validation_code(self, step: TestStep) -> str:
        """Generate validation code."""
        if self.framework == 'jest':
            return "expect(response.status).toBe(200);"
        elif self.framework == 'mocha':
            return "expect(response.status).to.equal(200);"
        else:
            return "// TODO: Implement validation"
    
    def _generate_integration_test_body(self, test_case: TestCase) -> List[str]:
        """Generate integration test specific code."""
        lines = []
        
        if test_case.request_payload:
            lines.append("// Request payload")
            lines.append(f"const payload = {test_case.request_payload};")
            lines.append("")
        
        if test_case.technical_contract and 'endpoint' in test_case.technical_contract:
            lines.append(f'const endpoint = "{test_case.technical_contract["endpoint"]}";')
            lines.append("")
        
        return lines
    
    def _generate_package_json(self) -> str:
        """Generate package.json file."""
        dependencies = {
            "axios": "^1.5.0"
        }
        
        dev_dependencies = {}
        scripts = {}
        
        if self.framework == 'jest':
            dev_dependencies["jest"] = "^29.0.0"
            dev_dependencies["@jest/globals"] = "^29.0.0"
            dev_dependencies["puppeteer"] = "^21.0.0"
            scripts["test"] = "jest"
            scripts["test:watch"] = "jest --watch"
        
        elif self.framework == 'mocha':
            dev_dependencies["mocha"] = "^10.0.0"
            dev_dependencies["chai"] = "^4.3.0"
            dev_dependencies["sinon"] = "^16.0.0"
            scripts["test"] = "mocha '**/*.test.js'"
        
        elif self.framework == 'playwright':
            dev_dependencies["@playwright/test"] = "^1.40.0"
            scripts["test"] = "playwright test"
            scripts["test:ui"] = "playwright test --ui"
        
        elif self.framework == 'cypress':
            dev_dependencies["cypress"] = "^13.0.0"
            scripts["test"] = "cypress run"
            scripts["test:open"] = "cypress open"
        
        package = {
            "name": "testteller-generated-tests",
            "version": "1.0.0",
            "description": "Automated tests generated by TestTeller",
            "scripts": scripts,
            "dependencies": dependencies,
            "devDependencies": dev_dependencies
        }
        
        return json.dumps(package, indent=2)
    
    def _generate_jest_config(self) -> str:
        """Generate Jest configuration."""
        return '''module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/*.test.js'],
  testTimeout: 30000,
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js'
  ]
};'''
    
    def _generate_playwright_config(self) -> str:
        """Generate Playwright configuration."""
        return '''module.exports = {
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'on-first-retry'
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' }
    }
  ],
  reporter: [
    ['html', { open: 'never' }]
  ]
};'''
    
    def _generate_cypress_config(self) -> str:
        """Generate Cypress configuration."""
        return '''module.exports = {
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: false,
    video: false,
    screenshotOnRunFailure: true
  }
};'''
    
    def _get_suite_name(self, category: str) -> str:
        """Get test suite name based on category."""
        return {
            'e2e': 'End-to-End Tests',
            'integration': 'Integration Tests',
            'technical': 'Technical Tests',
            'mocked': 'Mocked Tests'
        }.get(category, 'Tests')
    
    def _generate_test_name(self, test_case: TestCase) -> str:
        """Generate a descriptive test name."""
        if test_case.objective:
            # Clean and shorten the objective for test name
            name = test_case.objective[:80]
            if len(test_case.objective) > 80:
                name += '...'
            return name
        return f"Test {test_case.id}"