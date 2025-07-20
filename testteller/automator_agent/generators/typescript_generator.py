"""TypeScript test code generator supporting multiple frameworks."""

from typing import List, Dict, Optional
import json
import textwrap
from pathlib import Path

from .base_generator import BaseTestGenerator
from ..parser.markdown_parser import TestCase, TestStep
from testteller.core.constants import SUPPORTED_FRAMEWORKS


class TypeScriptTestGenerator(BaseTestGenerator):
    """Generator for TypeScript test code."""
    
    SUPPORTED_FRAMEWORKS = SUPPORTED_FRAMEWORKS['typescript']
    
    def __init__(self, framework: str, output_dir: Path):
        super().__init__(framework, output_dir)
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {framework}. Supported: {self.SUPPORTED_FRAMEWORKS}")
    
    def get_supported_frameworks(self) -> List[str]:
        return self.SUPPORTED_FRAMEWORKS
    
    def get_file_extension(self) -> str:
        if self.framework == 'playwright':
            return '.spec.ts'
        elif self.framework == 'cypress':
            return '.cy.ts'
        else:
            return '.test.ts'
    
    def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate TypeScript test files from test cases."""
        categorized_tests = self.categorize_tests(test_cases)
        generated_files = {}
        
        for category, cases in categorized_tests.items():
            if not cases:
                continue
                
            if self.framework == 'jest':
                content = self._generate_jest_tests(cases, category)
            elif self.framework == 'mocha':
                content = self._generate_mocha_tests(cases, category)
            elif self.framework == 'playwright':
                content = self._generate_playwright_tests(cases, category)
            elif self.framework == 'cypress':
                content = self._generate_cypress_tests(cases, category)
            elif self.framework == 'cucumber':
                content = self._generate_cucumber_tests(cases, category)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
            
            filename = f"{category.lower()}_tests{self.get_file_extension()}"
            generated_files[filename] = content
        
        # Generate configuration files
        generated_files.update(self._generate_config_files())
        
        return generated_files
    
    def _generate_jest_tests(self, test_cases: List[TestCase], category: str) -> str:
        """Generate Jest TypeScript test file."""
        imports = """import axios from 'axios';
import { expect, describe, test, beforeAll, afterAll } from '@jest/globals';
import { faker } from '@faker-js/faker';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';
const apiClient = axios.create({ baseURL, timeout: 10000 });
"""
        
        test_methods = []
        for test_case in test_cases:
            test_data = self.extract_test_data(test_case)
            method_name = self.sanitize_test_name(test_case.id)
            
            steps_code = []
            for step in test_case.test_steps:
                if step.action.lower() in ['get', 'post', 'put', 'delete']:
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        const response = await apiClient.{step.action.lower()}('{step.action}');")
                    steps_code.append(f"        expect(response.status).toBe(200);")
                else:
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        // TODO: Implement {step.action}")
            
            test_method = f"""
    test('{test_case.objective}', async () => {{
        // Test setup
        const testData = {json.dumps(test_data, indent=8)};
        
{chr(10).join(steps_code)}
        
        // Test cleanup
        // TODO: Add cleanup logic if needed
    }});"""
            
            test_methods.append(test_method)
        
        return f"""{imports}

describe('{category} Tests', () => {{
    beforeAll(async () => {{
        // Global test setup
    }});
    
    afterAll(async () => {{
        // Global test cleanup
    }});
{chr(10).join(test_methods)}
}});"""

    def _generate_mocha_tests(self, test_cases: List[TestCase], category: str) -> str:
        """Generate Mocha TypeScript test file."""
        imports = """import axios from 'axios';
import { expect } from 'chai';
import { faker } from '@faker-js/faker';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';
const apiClient = axios.create({ baseURL, timeout: 10000 });
"""
        
        test_methods = []
        for test_case in test_cases:
            test_data = self.extract_test_data(test_case)
            method_name = self.sanitize_test_name(test_case.id)
            
            steps_code = []
            for step in test_case.test_steps:
                if step.action.lower() in ['get', 'post', 'put', 'delete']:
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        const response = await apiClient.{step.action.lower()}('{step.action}');")
                    steps_code.append(f"        expect(response.status).to.equal(200);")
                else:
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        // TODO: Implement {step.action}")
            
            test_method = f"""
    it('{test_case.objective}', async function() {{
        this.timeout(30000);
        
        // Test setup
        const testData = {json.dumps(test_data, indent=8)};
        
{chr(10).join(steps_code)}
        
        // Test cleanup
        // TODO: Add cleanup logic if needed
    }});"""
            
            test_methods.append(test_method)
        
        return f"""{imports}

describe('{category} Tests', function() {{
    before(async function() {{
        // Global test setup
    }});
    
    after(async function() {{
        // Global test cleanup
    }});
{chr(10).join(test_methods)}
}});"""

    def _generate_playwright_tests(self, test_cases: List[TestCase], category: str) -> str:
        """Generate Playwright TypeScript test file."""
        imports = """import { test, expect, Page, BrowserContext } from '@playwright/test';
import { faker } from '@faker-js/faker';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';
"""
        
        test_methods = []
        for test_case in test_cases:
            test_data = self.extract_test_data(test_case)
            method_name = self.sanitize_test_name(test_case.id)
            
            steps_code = []
            for step in test_case.test_steps:
                if 'navigate' in step.action.lower() or 'goto' in step.action.lower():
                    steps_code.append(f"    // {step.action}")
                    steps_code.append(f"    await page.goto(baseURL);")
                elif 'click' in step.action.lower():
                    steps_code.append(f"    // {step.action}")
                    steps_code.append(f"    await page.click('selector');")
                elif 'fill' in step.action.lower() or 'type' in step.action.lower():
                    steps_code.append(f"    // {step.action}")
                    steps_code.append(f"    await page.fill('input', 'value');")
                else:
                    steps_code.append(f"    // {step.action}")
                    steps_code.append(f"    // TODO: Implement {step.action}")
            
            test_method = f"""
test('{test_case.objective}', async ({{ page }}) => {{
    // Test setup
    const testData = {json.dumps(test_data, indent=4)};
    
{chr(10).join(steps_code)}
    
    // Test assertions
    await expect(page).toHaveTitle(/.*title.*/);
}});"""
            
            test_methods.append(test_method)
        
        return f"""{imports}
{chr(10).join(test_methods)}"""

    def _generate_cypress_tests(self, test_cases: List[TestCase], category: str) -> str:
        """Generate Cypress TypeScript test file."""
        imports = """/// <reference types="cypress" />
import { faker } from '@faker-js/faker';

const baseURL = Cypress.env('BASE_URL') || 'http://localhost:3000';
"""
        
        test_methods = []
        for test_case in test_cases:
            test_data = self.extract_test_data(test_case)
            method_name = self.sanitize_test_name(test_case.id)
            
            steps_code = []
            for step in test_case.test_steps:
                if 'visit' in step.action.lower() or 'navigate' in step.action.lower():
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        cy.visit(baseURL);")
                elif 'click' in step.action.lower():
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        cy.get('selector').click();")
                elif 'type' in step.action.lower():
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        cy.get('input').type('value');")
                else:
                    steps_code.append(f"        // {step.description}")
                    steps_code.append(f"        // TODO: Implement {step.action}")
            
            test_method = f"""
    it('{test_case.objective}', () => {{
        // Test setup
        const testData = {json.dumps(test_data, indent=8)};
        
{chr(10).join(steps_code)}
        
        // Test assertions
        cy.get('body').should('be.visible');
    }});"""
            
            test_methods.append(test_method)
        
        return f"""{imports}

describe('{category} Tests', () => {{
    beforeEach(() => {{
        // Setup before each test
    }});
    
    afterEach(() => {{
        // Cleanup after each test
    }});
{chr(10).join(test_methods)}
}});"""

    def _generate_cucumber_tests(self, test_cases: List[TestCase], category: str) -> str:
        """Generate Cucumber TypeScript feature file."""
        feature_content = f"""Feature: {category} Tests
  As a user
  I want to test {category.lower()} functionality
  So that I can ensure the system works correctly

"""
        
        for test_case in test_cases:
            scenario = f"""  Scenario: {test_case.objective}
"""
            
            for step in test_case.test_steps:
                if step.action.lower().startswith('given'):
                    scenario += f"    Given {step.description}\n"
                elif step.action.lower().startswith('when'):
                    scenario += f"    When {step.description}\n"
                elif step.action.lower().startswith('then'):
                    scenario += f"    Then {step.description}\n"
                else:
                    scenario += f"    And {step.description}\n"
            
            feature_content += scenario + "\n"
        
        return feature_content
    
    def _generate_config_files(self) -> Dict[str, str]:
        """Generate framework-specific configuration files."""
        config_files = {}
        
        if self.framework == 'jest':
            config_files['jest.config.ts'] = """import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>'],
  testMatch: ['**/*.test.ts'],
  collectCoverageFrom: [
    '**/*.ts',
    '!**/*.d.ts'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html']
};

export default config;"""
            
            config_files['tsconfig.json'] = """{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "types": ["jest", "node"]
  },
  "include": ["**/*.ts"],
  "exclude": ["node_modules", "dist"]
}"""
        
        elif self.framework == 'playwright':
            config_files['playwright.config.ts'] = """import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
});"""
        
        elif self.framework == 'cypress':
            config_files['cypress.config.ts'] = """import { defineConfig } from 'cypress';

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: false,
    video: false,
    screenshotOnRunFailure: false,
    viewportWidth: 1280,
    viewportHeight: 720,
  },
});"""
        
        # Generate package.json
        from testteller.core.constants import FRAMEWORK_DEPENDENCIES, PACKAGE_JSON_DEFAULTS
        dependencies = FRAMEWORK_DEPENDENCIES.get('typescript', {}).get(self.framework, {})
        
        package_json = {
            "name": PACKAGE_JSON_DEFAULTS["name"],
            "version": PACKAGE_JSON_DEFAULTS["version"],
            "description": f"TypeScript {self.framework} tests generated by TestTeller Agent",
            "scripts": {
                "test": self._get_test_script(),
                "test:watch": f"{self._get_test_script()} --watch",
                "test:coverage": f"{self._get_test_script()} --coverage"
            },
            "devDependencies": dependencies.get('devDependencies', {})
        }
        
        config_files['package.json'] = json.dumps(package_json, indent=2)
        
        return config_files
    
    def _get_test_script(self) -> str:
        """Get the test script command for the framework."""
        if self.framework == 'jest':
            return "jest"
        elif self.framework == 'mocha':
            return "mocha --require ts-node/register **/*.test.ts"
        elif self.framework == 'playwright':
            return "playwright test"
        elif self.framework == 'cypress':
            return "cypress run"
        elif self.framework == 'cucumber':
            return "cucumber-js --require-module ts-node/register --require '**/*.steps.ts' **/*.feature"
        else:
            return "npm test"