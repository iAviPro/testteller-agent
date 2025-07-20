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
    action: str
    technical_details: Optional[str] = None
    validation: Optional[str] = None
    validation_details: Optional[str] = None


@dataclass
class TestCase:
    """Represents a parsed test case."""
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
        self.test_case_pattern = re.compile(
            r'### Test Case (E2E_|INT_|TECH_|MOCK_)\[(\d+)\]'
        )
        
    def parse_file(self, file_path: Path) -> List[TestCase]:
        """Parse a markdown file and extract all test cases."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> List[TestCase]:
        """Parse markdown content and extract test cases."""
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
                    f"{prefix}[{number}]", 
                    section_content
                )
                if test_case:
                    test_cases.append(test_case)
        
        logger.info(f"Parsed {len(test_cases)} test cases")
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
            
            # Parse sections
            elif line.startswith('#### Objective'):
                current_section = 'objective'
            elif line.startswith('#### References'):
                current_section = 'references'
            elif line.startswith('#### Prerequisites & Setup'):
                current_section = 'prerequisites'
            elif line.startswith('#### Test Steps'):
                current_section = 'steps'
            elif line.startswith('#### Expected Final State'):
                current_section = 'expected_state'
            elif line.startswith('#### Error Scenario Details'):
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
            
            # Parse sections
            elif line.startswith('#### Objective'):
                current_section = 'objective'
            elif line.startswith('#### Technical Contract'):
                current_section = 'contract'
                test_case.technical_contract = {}
            elif line.startswith('#### Test Scenario'):
                current_section = 'scenario'
            elif line.startswith('#### Request/Message Payload'):
                current_section = 'payload'
            elif line.startswith('#### Expected Response/Assertions'):
                current_section = 'response'
                test_case.expected_response = {}
            elif line.startswith('#### Error Scenario Details'):
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
            
            # Parse sections
            elif line.startswith('#### Objective'):
                current_section = 'objective'
            elif line.startswith('#### Test Hypothesis'):
                current_section = 'hypothesis'
            elif line.startswith('#### Test Setup'):
                current_section = 'setup'
                test_case.test_setup = {}
            elif line.startswith('#### Execution Steps'):
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