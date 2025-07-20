"""Java test code generator supporting JUnit and TestNG frameworks."""

from typing import List, Dict
import textwrap
from pathlib import Path

from .base_generator import BaseTestGenerator
from ..parser.markdown_parser import TestCase, TestStep
from testteller.core.constants import SUPPORTED_FRAMEWORKS


class JavaTestGenerator(BaseTestGenerator):
    """Generator for Java test code."""
    
    SUPPORTED_FRAMEWORKS = SUPPORTED_FRAMEWORKS['java']
    
    def __init__(self, framework: str, output_dir: Path):
        super().__init__(framework, output_dir)
        if framework not in self.SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {framework}. Supported: {self.SUPPORTED_FRAMEWORKS}")
    
    def get_supported_frameworks(self) -> List[str]:
        return self.SUPPORTED_FRAMEWORKS
    
    def get_file_extension(self) -> str:
        return '.java'
    
    def generate(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate Java test files from test cases."""
        generated_files = {}
        
        # Categorize tests
        categorized = self.categorize_tests(test_cases)
        
        # Generate files for each category
        for category, cases in categorized.items():
            if cases:
                class_name = f"{self._get_category_class_name(category)}Test"
                file_name = f"{class_name}{self.get_file_extension()}"
                content = self._generate_test_class(class_name, category, cases)
                generated_files[file_name] = content
        
        # Generate pom.xml for Maven
        generated_files['pom.xml'] = self._generate_pom_xml()
        
        # Generate test base class
        generated_files['TestBase.java'] = self._generate_test_base()
        
        return generated_files
    
    def _generate_test_class(self, class_name: str, category: str, test_cases: List[TestCase]) -> str:
        """Generate a Java test class."""
        package = "com.testteller.generated"
        imports = self._generate_imports(category)
        class_body = self._generate_class_body(class_name, category, test_cases)
        
        return f'''package {package};

{imports}

{class_body}'''
    
    def _generate_imports(self, category: str) -> str:
        """Generate import statements."""
        imports = []
        
        # Framework-specific imports
        if self.framework == 'junit5':
            imports.extend([
                "import org.junit.jupiter.api.*",
                "import static org.junit.jupiter.api.Assertions.*",
            ])
        elif self.framework == 'junit4':
            imports.extend([
                "import org.junit.*",
                "import static org.junit.Assert.*",
            ])
        elif self.framework == 'testng':
            imports.extend([
                "import org.testng.annotations.*",
                "import static org.testng.Assert.*",
            ])
        elif self.framework == 'playwright':
            imports.extend([
                "import com.microsoft.playwright.*",
                "import org.junit.jupiter.api.*",
                "import static org.junit.jupiter.api.Assertions.*",
            ])
        elif self.framework == 'karate':
            imports.extend([
                "import com.intuit.karate.junit5.Karate",
                "import org.junit.jupiter.api.*",
            ])
        elif self.framework == 'cucumber':
            imports.extend([
                "import io.cucumber.java.en.*",
                "import static org.junit.Assert.*",
            ])
        
        # Common imports
        imports.extend([
            "import java.util.*",
            "import java.net.http.*",
            "import java.net.URI",
            "import java.time.Duration",
            "import com.fasterxml.jackson.databind.ObjectMapper",
        ])
        
        # Category-specific imports
        if category == 'e2e':
            imports.extend([
                "import org.openqa.selenium.*",
                "import org.openqa.selenium.chrome.ChromeDriver",
                "import org.openqa.selenium.support.ui.*",
            ])
        
        return '\n'.join(imports) + ';'
    
    def _generate_class_body(self, class_name: str, category: str, test_cases: List[TestCase]) -> str:
        """Generate the test class body."""
        setup_method = self._generate_setup_method()
        teardown_method = self._generate_teardown_method()
        test_methods = []
        
        for test_case in test_cases:
            test_method = self._generate_test_method(test_case)
            test_methods.append(test_method)
        
        class_content = f'''
public class {class_name} extends TestBase {{
    
    private HttpClient httpClient;
    private ObjectMapper objectMapper;
    {"private WebDriver driver;" if category == "e2e" else ""}
    
{textwrap.indent(setup_method, '    ')}

{textwrap.indent(teardown_method, '    ')}

{''.join(test_methods)}
}}'''
        
        return class_content.strip()
    
    def _generate_setup_method(self) -> str:
        """Generate setup method based on framework."""
        if self.framework == 'junit5':
            annotation = "@BeforeEach"
        elif self.framework == 'junit4':
            annotation = "@Before"
        else:  # testng
            annotation = "@BeforeMethod"
        
        return f'''{annotation}
public void setUp() {{
    httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build();
    objectMapper = new ObjectMapper();
}}'''
    
    def _generate_teardown_method(self) -> str:
        """Generate teardown method based on framework."""
        if self.framework == 'junit5':
            annotation = "@AfterEach"
        elif self.framework == 'junit4':
            annotation = "@After"
        else:  # testng
            annotation = "@AfterMethod"
        
        return f'''{annotation}
public void tearDown() {{
    // Cleanup resources
    if (driver != null) {{
        driver.quit();
    }}
}}'''
    
    def _generate_test_method(self, test_case: TestCase) -> str:
        """Generate a test method."""
        method_name = f"test{self._to_camel_case(test_case.id)}"
        test_doc = self.generate_test_description(test_case)
        test_body = self._generate_test_body(test_case)
        
        if self.framework in ['junit5', 'junit4']:
            annotation = "@Test"
        else:  # testng
            annotation = '@Test(description = "' + test_case.objective + '")'
        
        return f'''
    {annotation}
    public void {method_name}() throws Exception {{
        /*
         * {test_doc.replace(chr(10), chr(10) + '         * ')}
         */
{textwrap.indent(test_body, '        ')}
    }}
'''
    
    def _generate_test_body(self, test_case: TestCase) -> str:
        """Generate test method body."""
        body_lines = []
        
        # Setup
        if test_case.prerequisites:
            body_lines.append("// Setup")
            test_data = self.extract_test_data(test_case)
            for key, value in test_data.items():
                java_type = self._infer_java_type(value)
                if isinstance(value, str):
                    body_lines.append(f'{java_type} {self._to_camel_case(key)} = "{value}";')
                else:
                    body_lines.append(f'{java_type} {self._to_camel_case(key)} = {value};')
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
        
        # Default if empty
        if not body_lines:
            body_lines.append('fail("Test not implemented");')
        
        return '\n'.join(body_lines)
    
    def _generate_step_code(self, step: TestStep, test_case: TestCase) -> str:
        """Generate code for a test step."""
        action_lower = step.action.lower()
        
        if 'api' in action_lower or 'request' in action_lower:
            return '''HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create(baseUrl + "/api/endpoint"))
    .header("Authorization", "Bearer " + token)
    .PUT(HttpRequest.BodyPublishers.ofString("{}")) // TODO: Add request body
    .build();
    
HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());'''
        elif 'navigate' in action_lower or 'click' in action_lower:
            return '''// TODO: Implement UI interaction
// driver.get(baseUrl);
// driver.findElement(By.id("element-id")).click();'''
        else:
            return "// TODO: Implement action"
    
    def _generate_validation_code(self, step: TestStep) -> str:
        """Generate validation code."""
        if self.framework == 'testng':
            return "assertEquals(response.statusCode(), 200);"
        else:
            return "assertEquals(200, response.statusCode());"
    
    def _generate_integration_test_body(self, test_case: TestCase) -> List[str]:
        """Generate integration test specific code."""
        lines = []
        
        if test_case.request_payload:
            lines.append("// Request payload")
            lines.append(f'String payload = """')
            lines.append(test_case.request_payload)
            lines.append('""";')
            lines.append("")
        
        if test_case.technical_contract and 'endpoint' in test_case.technical_contract:
            lines.append(f'String endpoint = "{test_case.technical_contract["endpoint"]}";')
            lines.append("")
        
        return lines
    
    def _generate_test_base(self) -> str:
        """Generate base test class."""
        return '''package com.testteller.generated;

import java.util.Properties;
import java.io.InputStream;

public abstract class TestBase {
    protected static String baseUrl;
    protected static String token;
    protected static Properties config;
    
    static {
        try {
            config = new Properties();
            InputStream is = TestBase.class.getClassLoader().getResourceAsStream("test.properties");
            if (is != null) {
                config.load(is);
            }
            baseUrl = config.getProperty("base.url", "http://localhost:8080");
            token = config.getProperty("auth.token", "test-token");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}'''
    
    def _generate_pom_xml(self) -> str:
        """Generate Maven pom.xml file."""
        junit_version = "5.9.0" if self.framework == 'junit5' else "4.13.2"
        
        dependencies = []
        
        if self.framework == 'junit5':
            dependencies.append(f'''
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>{junit_version}</version>
            <scope>test</scope>
        </dependency>''')
        elif self.framework == 'junit4':
            dependencies.append(f'''
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>{junit_version}</version>
            <scope>test</scope>
        </dependency>''')
        else:  # testng
            dependencies.append('''
        <dependency>
            <groupId>org.testng</groupId>
            <artifactId>testng</artifactId>
            <version>7.8.0</version>
            <scope>test</scope>
        </dependency>''')
        
        # Common dependencies
        dependencies.extend(['''
        <dependency>
            <groupId>org.seleniumhq.selenium</groupId>
            <artifactId>selenium-java</artifactId>
            <version>4.15.0</version>
            <scope>test</scope>
        </dependency>''', '''
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>2.15.2</version>
        </dependency>''', '''
        <dependency>
            <groupId>io.rest-assured</groupId>
            <artifactId>rest-assured</artifactId>
            <version>5.3.2</version>
            <scope>test</scope>
        </dependency>'''])
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.testteller</groupId>
    <artifactId>generated-tests</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
{''.join(dependencies)}
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.0.0-M7</version>
            </plugin>
        </plugins>
    </build>
</project>'''
    
    def _get_category_class_name(self, category: str) -> str:
        """Get Java class name for category."""
        return {
            'e2e': 'EndToEnd',
            'integration': 'Integration',
            'technical': 'Technical',
            'mocked': 'Mocked'
        }.get(category, 'Test')
    
    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        parts = text.replace('[', '_').replace(']', '').split('_')
        return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
    
    def _infer_java_type(self, value) -> str:
        """Infer Java type from Python value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "double"
        else:
            return "String"