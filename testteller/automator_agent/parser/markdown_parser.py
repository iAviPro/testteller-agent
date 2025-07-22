"""
Markdown parser for TestTeller generated test cases.

This module parses the structured markdown output from TestTeller
and converts it into structured TestCase objects.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestStep:
    """Represents a single test step."""
    __test__ = False  # Tell pytest this is not a test class
    action: str
    technical_details: Optional[str] = None
    validation: Optional[str] = None
    validation_details: Optional[str] = None


@dataclass
class TestCase:
    """Represents a parsed test case."""
    __test__ = False  # Tell pytest this is not a test class
    id: str
    feature: str
    type: str
    category: str
    objective: str
    references: Dict[str, str] = field(default_factory=dict)
    prerequisites: Dict[str, Any] = field(default_factory=dict)
    test_steps: List[TestStep] = field(default_factory=list)
    expected_state: Dict[str, str] = field(default_factory=dict)
    error_scenario: Optional[Dict[str, str]] = None
    
    # Integration-specific fields
    integration: Optional[str] = None
    technical_contract: Optional[Dict[str, str]] = None
    request_payload: Optional[str] = None
    expected_response: Optional[Dict[str, Any]] = None
    
    # Technical test-specific fields
    technical_area: Optional[str] = None
    focus: Optional[str] = None
    hypothesis: Optional[str] = None
    test_setup: Optional[Dict[str, str]] = None


class MarkdownTestCaseParser:
    """Parser for TestTeller markdown test case files."""
    
    def __init__(self):
        # Enhanced pattern to match both markdown headers and plain text headers
        self.test_case_pattern = re.compile(
            r'(?:###\s+)?Test Case (E2E_|INT_|TECH_|MOCK_)(.+?)(?=\n|$)'
        )
        self.table_pattern = re.compile(
            r'\|\s*S\.No\s*\|\s*Test ID\s*\|'
        )
        
    def parse_file(self, file_path: Path) -> List[TestCase]:
        """Parse a markdown file and extract all test cases."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[TestCase]:
        """Parse markdown content and extract test cases."""
        test_cases = []
        
        # Check if content contains tabular format
        if self._has_tabular_format(content):
            logger.info("Detected tabular format, parsing tables...")
            tabular_test_cases = self._parse_tabular_format(content)
            test_cases.extend(tabular_test_cases)
        
        # Parse traditional detailed format (detailed specifications section)
        traditional_test_cases = self._parse_traditional_format(content)
        test_cases.extend(traditional_test_cases)
        
        # Remove duplicates based on test ID
        unique_test_cases = []
        seen_ids = set()
        for tc in test_cases:
            if tc.id not in seen_ids:
                unique_test_cases.append(tc)
                seen_ids.add(tc.id)
        
        logger.info(f"Parsed {len(unique_test_cases)} unique test cases")
        return unique_test_cases
    
    def _has_tabular_format(self, content: str) -> bool:
        """Check if content contains tabular format."""
        return bool(self.table_pattern.search(content))
    
    def _parse_tabular_format(self, content: str) -> List[TestCase]:
        """Parse test cases from tabular summary format."""
        test_cases = []
        lines = content.split('\n')
        
        current_table_type = None
        table_headers = []
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect table type from section headers (both markdown and plain text)
            if ('Test Cases' in line and 
                (line.startswith('### ') or not line.startswith('|'))):
                if 'End-to-End' in line or 'E2E' in line:
                    current_table_type = 'E2E'
                elif 'Integration' in line:
                    current_table_type = 'INT'
                elif 'Technical' in line:
                    current_table_type = 'TECH'
                elif 'Mocked System' in line:
                    current_table_type = 'MOCK'
                continue
            
            # Parse table headers
            if line.startswith('|') and 'S.No' in line and 'Test ID' in line:
                table_headers = [header.strip() for header in line.split('|')[1:-1]]
                in_table = True
                continue
            
            # Skip table separator line
            if in_table and line.startswith('|') and '---' in line:
                continue
            
            # Parse table rows
            if in_table and line.startswith('|') and current_table_type:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) >= 5:  # Minimum expected columns
                    test_case = self._create_test_case_from_table_row(
                        current_table_type, cells, table_headers
                    )
                    if test_case:
                        test_cases.append(test_case)
                continue
            
            # End of table detection
            if in_table and (not line.startswith('|') or not line):
                in_table = False
                current_table_type = None
                table_headers = []
        
        return test_cases
    
    def _create_test_case_from_table_row(self, test_type: str, cells: List[str], headers: List[str]) -> Optional[TestCase]:
        """Create a TestCase object from a table row."""
        try:
            # Map headers to values
            row_data = dict(zip(headers, cells))
            
            # Extract common fields
            serial_no = row_data.get('S.No', '')
            test_id = row_data.get('Test ID', '')
            objective = row_data.get('Objective', '')
            priority = row_data.get('Priority', 'Medium')
            category = row_data.get('Category', '')
            
            if not test_id or not objective:
                return None
            
            test_case = TestCase(
                id=test_id,
                feature="",
                type="",
                category=category,
                objective=objective
            )
            
            # Set type-specific fields
            if test_type == 'E2E':
                test_case.feature = row_data.get('Feature', '')
                test_case.type = 'Journey/Flow'
            elif test_type == 'INT':
                test_case.integration = row_data.get('Integration', '')
                test_case.type = row_data.get('Type', 'API')
            elif test_type == 'TECH':
                test_case.technical_area = row_data.get('Technical Area', '')
                test_case.focus = row_data.get('Focus', '')
                test_case.type = 'Technical'
            elif test_type == 'MOCK':
                test_case.feature = row_data.get('Component Under Test', '')
                test_case.type = row_data.get('Type', 'Functional')
            
            return test_case
            
        except Exception as e:
            logger.error(f"Failed to create test case from table row: {e}")
            return None
    
    def _parse_traditional_format(self, content: str) -> List[TestCase]:
        """Parse test cases using traditional detailed format."""
        test_cases = []
        
        # Split content by test case headers
        test_case_sections = self.test_case_pattern.split(content)
        
        # Process each test case section
        for i in range(1, len(test_case_sections), 3):
            if i + 2 < len(test_case_sections):
                prefix = test_case_sections[i]
                number = test_case_sections[i + 1]
                section_content = test_case_sections[i + 2]
                
                test_case = self._parse_test_case(
                    f"{prefix}{number}", 
                    section_content
                )
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _parse_test_case(self, test_id: str, content: str) -> Optional[TestCase]:
        """Parse a single test case section."""
        try:
            test_case = TestCase(id=test_id, feature="", type="", category="", objective="")
            
            # Determine test type
            if test_id.startswith("E2E_"):
                test_case = self._parse_e2e_test(test_case, content)
            elif test_id.startswith("INT_"):
                test_case = self._parse_integration_test(test_case, content)
            elif test_id.startswith("TECH_"):
                test_case = self._parse_technical_test(test_case, content)
            elif test_id.startswith("MOCK_"):
                test_case = self._parse_mocked_test(test_case, content)
            
            return test_case
            
        except Exception as e:
            logger.error(f"Failed to parse test case {test_id}: {e}")
            return None
    
    def _parse_e2e_test(self, test_case: TestCase, content: str) -> TestCase:
        """Parse E2E test case specific fields."""
        lines = content.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Parse metadata
            if line.startswith('**Feature:**'):
                test_case.feature = self._extract_value(line, '**Feature:**')
            elif line.startswith('**Type:**'):
                test_case.type = self._extract_value(line, '**Type:**')
            elif line.startswith('**Category:**'):
                test_case.category = self._extract_value(line, '**Category:**')
            
            # Parse sections (both markdown headers and plain text)
            elif line.startswith('#### Objective') or line == 'Objective':
                current_section = 'objective'
            elif line.startswith('#### References') or line == 'References':
                current_section = 'references'
            elif line.startswith('#### Prerequisites & Setup') or line == 'Prerequisites & Setup':
                current_section = 'prerequisites'
            elif line.startswith('#### Test Steps') or line == 'Test Steps':
                current_section = 'steps'
            elif line.startswith('#### Expected Final State') or line.startswith('Expected Final'):
                current_section = 'expected_state'
            elif line.startswith('#### Error Scenario Details') or line == 'Error Scenario Details':
                current_section = 'error_scenario'
            
            # Parse section content
            elif current_section and line:
                if current_section == 'objective' and not line.startswith('#'):
                    test_case.objective = line
                elif current_section == 'references':
                    self._parse_reference_line(test_case, line)
                elif current_section == 'prerequisites':
                    self._parse_prerequisite_line(test_case, line)
                elif current_section == 'steps':
                    self._parse_step_line(test_case, line)
                elif current_section == 'expected_state':
                    self._parse_expected_state_line(test_case, line)
                elif current_section == 'error_scenario':
                    self._parse_error_scenario_line(test_case, line)
        
        return test_case
    
    def _parse_integration_test(self, test_case: TestCase, content: str) -> TestCase:
        """Parse Integration test case specific fields."""
        lines = content.strip().split('\n')
        current_section = None
        json_buffer = []
        in_json_block = False
        
        for line in lines:
            # Handle JSON blocks
            if line.strip() == '```json':
                in_json_block = True
                json_buffer = []
                continue
            elif line.strip() == '```' and in_json_block:
                in_json_block = False
                if current_section == 'payload':
                    test_case.request_payload = '\n'.join(json_buffer)
                continue
            elif in_json_block:
                json_buffer.append(line)
                continue
            
            line = line.strip()
            
            # Parse metadata
            if line.startswith('**Integration:**'):
                test_case.integration = self._extract_value(line, '**Integration:**')
            elif line.startswith('**Type:**'):
                test_case.type = self._extract_value(line, '**Type:**')
            elif line.startswith('**Category:**'):
                test_case.category = self._extract_value(line, '**Category:**')
            
            # Parse sections (both markdown headers and plain text)
            elif line.startswith('#### Objective') or line == 'Objective':
                current_section = 'objective'
            elif line.startswith('#### Technical Contract') or line == 'Technical Contract':
                current_section = 'contract'
                test_case.technical_contract = {}
            elif line.startswith('#### Test Scenario') or line == 'Test Scenario':
                current_section = 'scenario'
            elif line.startswith('#### Request/Message Payload') or line == 'Request/Message Payload':
                current_section = 'payload'
            elif line.startswith('#### Expected Response/Assertions') or line == 'Expected Response/Assertions':
                current_section = 'response'
                test_case.expected_response = {}
            elif line.startswith('#### Error Scenario Details') or line == 'Error Scenario Details':
                current_section = 'error_scenario'
            
            # Parse section content
            elif current_section and line:
                if current_section == 'objective' and not line.startswith('#'):
                    test_case.objective = line
                elif current_section == 'contract':
                    self._parse_contract_line(test_case, line)
                elif current_section == 'scenario':
                    self._parse_scenario_line(test_case, line)
                elif current_section == 'response':
                    self._parse_response_line(test_case, line)
                elif current_section == 'error_scenario':
                    self._parse_error_scenario_line(test_case, line)
        
        return test_case
    
    def _parse_technical_test(self, test_case: TestCase, content: str) -> TestCase:
        """Parse Technical test case specific fields."""
        lines = content.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Parse metadata
            if line.startswith('**Technical Area:**'):
                test_case.technical_area = self._extract_value(line, '**Technical Area:**')
            elif line.startswith('**Focus:**'):
                test_case.focus = self._extract_value(line, '**Focus:**')
            
            # Parse sections (both markdown headers and plain text)
            elif line.startswith('#### Objective') or line == 'Objective':
                current_section = 'objective'
            elif line.startswith('#### Test Hypothesis') or line == 'Test Hypothesis':
                current_section = 'hypothesis'
            elif line.startswith('#### Test Setup') or line == 'Test Setup':
                current_section = 'setup'
                test_case.test_setup = {}
            elif line.startswith('#### Execution Steps') or line == 'Execution Steps':
                current_section = 'steps'
            
            # Parse section content
            elif current_section and line:
                if current_section == 'objective' and not line.startswith('#'):
                    test_case.objective = line
                elif current_section == 'hypothesis' and not line.startswith('#'):
                    test_case.hypothesis = line
                elif current_section == 'setup':
                    self._parse_setup_line(test_case, line)
                elif current_section == 'steps':
                    self._parse_step_line(test_case, line)
        
        return test_case
    
    def _parse_mocked_test(self, test_case: TestCase, content: str) -> TestCase:
        """Parse Mocked test case specific fields."""
        # Similar structure to E2E tests with focus on mocked components
        return self._parse_e2e_test(test_case, content)
    
    # Helper methods
    def _extract_value(self, line: str, prefix: str) -> str:
        """Extract value after a prefix."""
        return line.replace(prefix, '').strip().strip('[').strip(']')
    
    def _parse_reference_line(self, test_case: TestCase, line: str):
        """Parse a reference line."""
        if '**Product:**' in line:
            test_case.references['product'] = self._extract_value(line, '**Product:**')
        elif '**Technical:**' in line:
            test_case.references['technical'] = self._extract_value(line, '**Technical:**')
    
    def _parse_prerequisite_line(self, test_case: TestCase, line: str):
        """Parse a prerequisite line."""
        if '**System State:**' in line:
            test_case.prerequisites['system_state'] = self._extract_value(line, '**System State:**')
        elif '**Test Data:**' in line:
            test_case.prerequisites['test_data'] = self._extract_value(line, '**Test Data:**')
        elif '**Mocked Services:**' in line:
            test_case.prerequisites['mocked_services'] = self._extract_value(line, '**Mocked Services:**')
    
    def _parse_step_line(self, test_case: TestCase, line: str):
        """Parse a test step line."""
        if line.strip() and line[0].isdigit():
            # Parse numbered steps
            step = TestStep(action="")
            
            if '**Action:**' in line:
                step.action = self._extract_value(line, '**Action:**')
            elif '**Validation:**' in line:
                step.validation = self._extract_value(line, '**Validation:**')
            elif '**Technical Details:**' in line:
                # This is a sub-item of the previous step
                if test_case.test_steps:
                    last_step = test_case.test_steps[-1]
                    if last_step.action and not last_step.technical_details:
                        last_step.technical_details = self._extract_value(line, '**Technical Details:**')
                    elif last_step.validation and not last_step.validation_details:
                        last_step.validation_details = self._extract_value(line, '**Technical Details:**')
            
            if step.action or step.validation:
                test_case.test_steps.append(step)
    
    def _parse_expected_state_line(self, test_case: TestCase, line: str):
        """Parse expected state line."""
        for key in ['**UI/Frontend:**', '**Backend/API:**', '**Database:**', '**Events/Messages:**']:
            if key in line:
                state_key = key.replace('**', '').replace(':', '').lower().replace('/', '_')
                test_case.expected_state[state_key] = self._extract_value(line, key)
    
    def _parse_error_scenario_line(self, test_case: TestCase, line: str):
        """Parse error scenario line."""
        if not test_case.error_scenario:
            test_case.error_scenario = {}
        
        if '**Error Condition:**' in line:
            test_case.error_scenario['condition'] = self._extract_value(line, '**Error Condition:**')
        elif '**Recovery/Expected Behavior:**' in line:
            test_case.error_scenario['recovery'] = self._extract_value(line, '**Recovery/Expected Behavior:**')
        elif '**Fault:**' in line:
            test_case.error_scenario['fault'] = self._extract_value(line, '**Fault:**')
        elif '**Expected Handling:**' in line:
            test_case.error_scenario['handling'] = self._extract_value(line, '**Expected Handling:**')
    
    def _parse_contract_line(self, test_case: TestCase, line: str):
        """Parse technical contract line."""
        if '**Endpoint/Topic:**' in line:
            test_case.technical_contract['endpoint'] = self._extract_value(line, '**Endpoint/Topic:**')
        elif '**Protocol/Pattern:**' in line:
            test_case.technical_contract['protocol'] = self._extract_value(line, '**Protocol/Pattern:**')
        elif '**Schema/Contract:**' in line:
            test_case.technical_contract['schema'] = self._extract_value(line, '**Schema/Contract:**')
    
    def _parse_scenario_line(self, test_case: TestCase, line: str):
        """Parse test scenario line."""
        if '**Given:**' in line:
            if 'given' not in test_case.prerequisites:
                test_case.prerequisites['given'] = self._extract_value(line, '**Given:**')
        elif '**When:**' in line:
            if 'when' not in test_case.prerequisites:
                test_case.prerequisites['when'] = self._extract_value(line, '**When:**')
        elif '**Then:**' in line:
            if 'then' not in test_case.prerequisites:
                test_case.prerequisites['then'] = self._extract_value(line, '**Then:**')
    
    def _parse_response_line(self, test_case: TestCase, line: str):
        """Parse expected response line."""
        if '**Status Code:**' in line:
            test_case.expected_response['status_code'] = self._extract_value(line, '**Status Code:**')
        elif '**Response Body/Schema:**' in line:
            test_case.expected_response['body_schema'] = self._extract_value(line, '**Response Body/Schema:**')
        elif '**Target State Change:**' in line:
            test_case.expected_response['state_change'] = self._extract_value(line, '**Target State Change:**')
        elif '**Headers/Metadata:**' in line:
            test_case.expected_response['headers'] = self._extract_value(line, '**Headers/Metadata:**')
    
    def _parse_setup_line(self, test_case: TestCase, line: str):
        """Parse test setup line."""
        if '**Target Component(s):**' in line:
            test_case.test_setup['targets'] = self._extract_value(line, '**Target Component(s):**')
        elif '**Tooling:**' in line:
            test_case.test_setup['tooling'] = self._extract_value(line, '**Tooling:**')
        elif '**Monitoring:**' in line:
            test_case.test_setup['monitoring'] = self._extract_value(line, '**Monitoring:**')
        elif '**Load Profile/Attack Vector:**' in line:
            test_case.test_setup['load_profile'] = self._extract_value(line, '**Load Profile/Attack Vector:**')
    
    def extract_test_data(self, test_case: TestCase) -> Dict[str, Any]:
        """Extract test data from a test case's prerequisites."""
        test_data = {}
        
        if 'test_data' in test_case.prerequisites:
            data_str = test_case.prerequisites['test_data']
            # Remove leading bullet points and whitespace
            data_str = data_str.lstrip('- ').strip()
            
            # Parse key-value pairs from the test data string
            # Format: "key1: value1, key2: value2, ..."
            pairs = data_str.split(',')
            for pair in pairs:
                pair = pair.strip()
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse the value to appropriate type
                    if value.lower() == 'true':
                        test_data[key] = True
                    elif value.lower() == 'false':
                        test_data[key] = False
                    elif value.replace('.', '').replace('-', '').isdigit():
                        # Check if it's a number
                        if '.' in value:
                            test_data[key] = float(value)
                        else:
                            test_data[key] = int(value)
                    else:
                        test_data[key] = value
        
        return test_data