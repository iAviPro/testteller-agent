"""Base class for RAG-enhanced test code generators."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import logging

from .parser.markdown_parser import TestCase

logger = logging.getLogger(__name__)


class BaseTestGenerator(ABC):
    """Abstract base class for RAG-enhanced test code generators."""
    
    def __init__(self, framework: str, output_dir: Path):
        self.framework = framework
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """
        Generate test code from parsed test cases.
        
        Args:
            test_cases: List of parsed test cases
            
        Returns:
            Dictionary mapping file paths to generated code
        """
        pass
    
    @abstractmethod
    def get_supported_frameworks(self) -> List[str]:
        """Return list of supported test frameworks."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Return file extension for generated tests."""
        pass
    
    def categorize_tests(self, test_cases: List[TestCase]) -> Dict[str, List[TestCase]]:
        """Categorize test cases by type."""
        categories = {
            'e2e': [],
            'integration': [],
            'technical': [],
            'mocked': []
        }
        
        for test_case in test_cases:
            if test_case.id.startswith('E2E_'):
                categories['e2e'].append(test_case)
            elif test_case.id.startswith('INT_'):
                categories['integration'].append(test_case)
            elif test_case.id.startswith('TECH_'):
                categories['technical'].append(test_case)
            elif test_case.id.startswith('MOCK_'):
                categories['mocked'].append(test_case)
        
        return categories
    
    def sanitize_test_name(self, name: str) -> str:
        """Convert test ID or feature name to valid function/method name."""
        # Replace special characters with underscores
        sanitized = name.lower()
        sanitized = sanitized.replace(' ', '_')
        sanitized = sanitized.replace('-', '_')
        sanitized = sanitized.replace('[', '_')
        sanitized = sanitized.replace(']', '')
        sanitized = sanitized.replace('/', '_')
        sanitized = sanitized.replace('\\', '_')
        
        # Remove any remaining non-alphanumeric characters except underscore
        import re
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'test_' + sanitized
        
        return sanitized
    
    def generate_test_description(self, test_case: TestCase) -> str:
        """Generate a descriptive comment/docstring for the test."""
        description_parts = [
            f"Test ID: {test_case.id}",
            f"Feature: {test_case.feature}" if test_case.feature else None,
            f"Type: {test_case.type}" if test_case.type else None,
            f"Category: {test_case.category}" if test_case.category else None,
            f"Objective: {test_case.objective}" if test_case.objective else None,
        ]
        
        # Filter out None values
        description_parts = [part for part in description_parts if part]
        
        return '\n'.join(description_parts)
    
    def extract_test_data(self, test_case: TestCase) -> Dict[str, Any]:
        """Extract test data from prerequisites."""
        test_data = {}
        
        if 'test_data' in test_case.prerequisites:
            # Parse test data string (e.g., "user_id: 123, product_id: 456")
            data_str = test_case.prerequisites['test_data']
            if data_str:
                pairs = data_str.split(',')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().strip('`')
                        value = value.strip().strip('`')
                        
                        # Try to infer type
                        if value.isdigit():
                            value = int(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        else:
                            # Try to parse as float
                            try:
                                if '.' in value:
                                    value = float(value)
                            except ValueError:
                                pass  # Keep as string
                        
                        test_data[key] = value
        
        return test_data
    
    def write_files(self, generated_files: Dict[str, str]) -> None:
        """Write generated files to disk."""
        for file_path, content in generated_files.items():
            full_path = self.output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated test file: {full_path}")