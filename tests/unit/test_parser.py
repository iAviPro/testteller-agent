"""Unit tests for testteller automator_agent markdown parser."""

import pytest

pytestmark = pytest.mark.automation
from pathlib import Path
import tempfile

from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser, TestCase, TestStep


class TestMarkdownTestCaseParser:
    """Test the markdown test case parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MarkdownTestCaseParser()
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        result = self.parser.parse_content("")
        assert result == []
    
    def test_parse_content_without_test_cases(self):
        """Test parsing content without test cases."""
        content = """
        # Some documentation
        This is just regular markdown content without test cases.
        """
        result = self.parser.parse_content(content)
        assert result == []
    
    def test_parse_single_e2e_test_case(self):
        """Test parsing a single E2E test case."""
        content = """
### Test Case E2E_[1]
**Feature:** User Authentication
**Type:** Login Flow
**Category:** Happy Path

#### Objective
Verify that users can successfully log in with valid credentials.

#### References
- **Product:** User Login PRD v1.2
- **Technical:** Authentication API Specification

#### Prerequisites & Setup
- **System State:** Clean database with test user
- **Test Data:** user_id: 123, email: test@example.com, password: testpass123
- **Mocked Services:** Payment service, Email service

#### Test Steps
1. **Action:** Navigate to the login page
   - **Technical Details:** Send GET request to /login
2. **Validation:** Verify login form is displayed
   - **Technical Details:** Check for username and password input fields
3. **Action:** Enter valid credentials and submit
   - **Technical Details:** Send POST request to /api/auth/login
4. **Validation:** Verify successful login
   - **Technical Details:** Expect 200 OK response with auth token

#### Expected Final State
- **UI/Frontend:** User dashboard page is displayed
- **Backend/API:** User session is created in Redis
- **Database:** Last login timestamp is updated
- **Events/Messages:** UserLoggedIn event published to auth-events topic

#### Error Scenario Details
- **Error Condition:** Invalid credentials provided
- **Recovery/Expected Behavior:** Error message displayed, user remains on login page
"""
        
        result = self.parser.parse_content(content)
        
        assert len(result) == 1
        test_case = result[0]
        
        # Basic metadata
        assert test_case.id == "E2E_[1]"
        assert test_case.feature == "User Authentication"
        assert test_case.type == "Login Flow"
        assert test_case.category == "Happy Path"
        assert "successfully log in" in test_case.objective
        
        # References
        assert "product" in test_case.references
        assert "technical" in test_case.references
        
        # Prerequisites
        assert "system_state" in test_case.prerequisites
        assert "test_data" in test_case.prerequisites
        assert "mocked_services" in test_case.prerequisites
        
        # Test steps
        assert len(test_case.test_steps) >= 1
        assert "Navigate" in test_case.test_steps[0].action
        assert "login page" in test_case.test_steps[0].action
        # Note: technical_details parsing may vary based on implementation
        
        # Expected state
        assert "ui_frontend" in test_case.expected_state
        assert "backend_api" in test_case.expected_state
        assert "database" in test_case.expected_state
        assert "events_messages" in test_case.expected_state
        
        # Error scenario
        assert test_case.error_scenario is not None
        assert "condition" in test_case.error_scenario
        assert "recovery" in test_case.error_scenario
    
    def test_parse_integration_test_case(self):
        """Test parsing an integration test case."""
        content = """
### Test Case INT_[1]
**Integration:** User Service -> Order Service
**Type:** API
**Category:** Contract

#### Objective
Verify that User Service correctly sends user data to Order Service when creating orders.

#### Technical Contract
- **Endpoint/Topic:** /api/v1/orders
- **Protocol/Pattern:** REST/HTTP
- **Schema/Contract:** OpenAPI 3.0 specification

#### Test Scenario
- **Given:** User service has valid user data
- **When:** Order creation request is sent
- **Then:** Order service receives and processes the request

#### Request/Message Payload
```json
{
  "userId": "user-123",
  "items": [
    {
      "productId": "prod-456",
      "quantity": 2,
      "price": 29.99
    }
  ],
  "shippingAddress": {
    "street": "123 Test St",
    "city": "Test City",
    "zipCode": "12345"
  }
}
```

#### Expected Response/Assertions
- **Status Code:** 201 Created
- **Response Body/Schema:** Matches order creation response schema
- **Target State Change:** New order record exists in orders database
- **Headers/Metadata:** Content-Type is application/json

#### Error Scenario Details
- **Fault:** Malformed JSON payload sent
- **Expected Handling:** 400 Bad Request returned, error logged
"""
        
        result = self.parser.parse_content(content)
        
        assert len(result) == 1
        test_case = result[0]
        
        assert test_case.id == "INT_[1]"
        assert test_case.integration == "User Service -> Order Service"
        assert test_case.type == "API"
        assert test_case.category == "Contract"
        
        # Technical contract
        assert test_case.technical_contract is not None
        assert "/api/v1/orders" in str(test_case.technical_contract["endpoint"])
        assert "REST/HTTP" in test_case.technical_contract["protocol"]
        
        # Request payload
        assert test_case.request_payload is not None
        assert "userId" in test_case.request_payload
        assert "user-123" in test_case.request_payload
        
        # Expected response
        assert test_case.expected_response is not None
        assert "201 Created" in test_case.expected_response["status_code"]
        
        # Error scenario
        assert test_case.error_scenario is not None
        assert "fault" in test_case.error_scenario
        assert "handling" in test_case.error_scenario
    
    def test_parse_technical_test_case(self):
        """Test parsing a technical test case."""
        content = """
### Test Case TECH_[1]
**Technical Area:** Performance
**Focus:** Load Testing

#### Objective
Determine the maximum throughput of the authentication service under load.

#### Test Hypothesis
The authentication service will handle 1000 concurrent users without exceeding 500ms response time.

#### Test Setup
- **Target Component(s):** Authentication Service, Redis Cache
- **Tooling:** k6 load testing framework
- **Monitoring:** Prometheus metrics, Grafana dashboards
- **Load Profile/Attack Vector:** Ramp up to 1000 concurrent users over 2 minutes

#### Execution Steps
1. Start monitoring systems
2. Configure k6 script with authentication endpoints
3. Execute load test with gradual ramp-up
4. Monitor response times and error rates
5. Analyze results and identify bottlenecks
"""
        
        result = self.parser.parse_content(content)
        
        assert len(result) == 1
        test_case = result[0]
        
        assert test_case.id == "TECH_[1]"
        assert test_case.technical_area == "Performance"
        assert test_case.focus == "Load Testing"
        assert "maximum throughput" in test_case.objective
        assert "1000 concurrent users" in test_case.hypothesis
        
        # Test setup
        assert test_case.test_setup is not None
        assert "targets" in test_case.test_setup
        assert "tooling" in test_case.test_setup
        assert "monitoring" in test_case.test_setup
        assert "load_profile" in test_case.test_setup
    
    def test_parse_multiple_test_cases(self):
        """Test parsing multiple test cases in one document."""
        content = """
### Test Case E2E_[1]
**Feature:** Login
**Type:** Authentication
**Category:** Happy Path

#### Objective
Test user login functionality.

### Test Case INT_[1]
**Integration:** Service A -> Service B
**Type:** API
**Category:** Contract

#### Objective
Test service integration.

### Test Case TECH_[1]
**Technical Area:** Security
**Focus:** Authentication

#### Objective
Test security vulnerabilities.
"""
        
        result = self.parser.parse_content(content)
        
        assert len(result) == 3
        assert result[0].id == "E2E_[1]"
        assert result[1].id == "INT_[1]"
        assert result[2].id == "TECH_[1]"
    
    def test_parse_file(self):
        """Test parsing from a file."""
        content = """
### Test Case E2E_[1]
**Feature:** Test Feature
**Type:** Test Type
**Category:** Test Category

#### Objective
Test objective.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            result = self.parser.parse_file(temp_path)
            assert len(result) == 1
            assert result[0].id == "E2E_[1]"
        finally:
            temp_path.unlink()
    
    def test_extract_test_data(self):
        """Test extracting test data from prerequisites."""
        content = """
### Test Case E2E_[1]
**Feature:** Test Feature
**Type:** Test Type
**Category:** Test Category

#### Objective
Test objective.

#### Prerequisites & Setup
- **Test Data:** user_id: 123, email: test@example.com, active: true, balance: 100.50
"""
        
        result = self.parser.parse_content(content)
        test_case = result[0]
        
        parser = MarkdownTestCaseParser()
        test_data = parser.extract_test_data(test_case)
        
        assert test_data["user_id"] == 123
        assert test_data["email"] == "test@example.com"
        assert test_data["active"] == True
        assert test_data["balance"] == 100.50
    
    def test_malformed_test_case(self):
        """Test parsing malformed test case doesn't crash."""
        content = """
### Test Case INVALID_FORMAT
This is not a properly formatted test case.
"""
        
        result = self.parser.parse_content(content)
        # Should not crash and return empty list or skip malformed cases
        assert isinstance(result, list)
    
    def test_helper_methods(self):
        """Test helper methods in the parser."""
        parser = self.parser
        
        # Test extract_value
        assert parser._extract_value("**Feature:** Test Feature", "**Feature:**") == "Test Feature"
        assert parser._extract_value("**Type:** [API Call]", "**Type:**") == "API Call"
        
        # Test with empty values
        assert parser._extract_value("**Feature:**", "**Feature:**") == ""


class TestTestCaseDataClass:
    """Test the TestCase data class."""
    
    def test_test_case_creation(self):
        """Test creating a TestCase instance."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="Login",
            type="Authentication",
            category="Happy Path",
            objective="Test login functionality"
        )
        
        assert test_case.id == "E2E_[1]"
        assert test_case.feature == "Login"
        assert test_case.type == "Authentication"
        assert test_case.category == "Happy Path"
        assert test_case.objective == "Test login functionality"
        
        # Default values
        assert test_case.references == {}
        assert test_case.prerequisites == {}
        assert test_case.test_steps == []
        assert test_case.expected_state == {}
        assert test_case.error_scenario is None
    
    def test_test_step_creation(self):
        """Test creating a TestStep instance."""
        step = TestStep(
            action="Click login button",
            technical_details="Send POST to /login",
            validation="Check response",
            validation_details="Expect 200 OK"
        )
        
        assert step.action == "Click login button"
        assert step.technical_details == "Send POST to /login"
        assert step.validation == "Check response"
        assert step.validation_details == "Expect 200 OK"