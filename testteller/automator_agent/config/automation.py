"""
TestAutomator (Automation) Wizard configuration.

Handles configuration of test automation settings including language,
framework, and related automation parameters.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ...core.config.ui import UIHelper, validate_url, validate_directory_path
from ...core.constants import (
    SUPPORTED_LANGUAGES, 
    SUPPORTED_FRAMEWORKS,
    DEFAULT_AUTOMATION_LANGUAGE,
    DEFAULT_AUTOMATION_FRAMEWORK,
    DEFAULT_AUTOMATION_OUTPUT_DIR
)

logger = logging.getLogger(__name__)


class TestAutomatorWizard:
    """TestAutomator (Automation) configuration wizard."""
    
    def __init__(self):
        """Initialize TestAutomator wizard."""
        self.config_data = {}
    
    def configure(self, ui_helper: UIHelper) -> Dict[str, str]:
        """
        Run TestAutomator (Automation) configuration wizard.
        
        Args:
            ui_helper: UI helper for user interaction
            
        Returns:
            Dictionary containing automation configuration
        """
        config = {}
        
        # Show section header
        ui_helper.show_section_header(
            "🧪 TestAutomator (Automation) Wizard",
            "Configure automated test code generation for multiple languages and frameworks",
            "🧪"
        )
        
        # Show capabilities overview
        self._show_automation_overview(ui_helper)
        
        # Ask if user wants to configure automation
        configure_automation = ui_helper.confirm(
            "Would you like to configure test automation?",
            default=True
        )
        
        if not configure_automation:
            ui_helper.show_info("TestAutomator automation configuration skipped.")
            return self._get_default_config()
        
        # Configure automation settings
        config.update(self._configure_language_and_framework(ui_helper))
        config.update(self._configure_testing_environment(ui_helper))
        config.update(self._configure_output_settings(ui_helper))
        
        # Show LLM enhancement option
        config.update(self._configure_llm_enhancement(ui_helper))
        
        # Validate configuration
        is_valid, errors = self._validate_configuration(config)
        if not is_valid:
            ui_helper.show_error("Configuration validation failed:", errors)
            raise Exception("TestAutomator configuration validation failed")
        
        # Show configuration summary
        self._show_configuration_summary(ui_helper, config)
        
        # Store configuration
        self.config_data = config
        
        return config
    
    def _show_automation_overview(self, ui_helper: UIHelper):
        """Show automation capabilities overview."""
        ui_helper.show_info(
            "TestAutomator can generate automated test code in multiple languages:",
            [
                f"🐍 Python: {', '.join(SUPPORTED_FRAMEWORKS['python'])}",
                f"🟨 JavaScript: {', '.join(SUPPORTED_FRAMEWORKS['javascript'])}",
                f"🔷 TypeScript: {', '.join(SUPPORTED_FRAMEWORKS['typescript'])}",
                f"☕ Java: {', '.join(SUPPORTED_FRAMEWORKS['java'])}"
            ]
        )
        
        ui_helper.show_info(
            "Features include:",
            [
                "✅ Multi-framework test generation",
                "✅ AI-powered test enhancement (optional)",
                "✅ Realistic test data generation", 
                "✅ Framework-specific best practices",
                "✅ Configuration files and dependencies"
            ]
        )
    
    def _configure_language_and_framework(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure programming language and testing framework."""
        config = {}
        
        # Language selection
        language_descriptions = {
            "python": "Popular for data science, web apps, and automation",
            "javascript": "Ubiquitous for web development and Node.js apps",
            "typescript": "Type-safe JavaScript for larger applications",
            "java": "Enterprise applications and Android development"
        }
        
        language = ui_helper.get_choice(
            "Select your primary programming language",
            SUPPORTED_LANGUAGES,
            default=DEFAULT_AUTOMATION_LANGUAGE,
            descriptions=language_descriptions
        )
        config["AUTOMATION_LANGUAGE"] = language
        
        # Framework selection based on language
        available_frameworks = SUPPORTED_FRAMEWORKS[language]
        framework_descriptions = self._get_framework_descriptions(language)
        
        ui_helper.show_info(f"Available frameworks for {language}:", available_frameworks)
        
        framework = ui_helper.get_choice(
            f"Select testing framework for {language}",
            available_frameworks,
            default=available_frameworks[0],
            descriptions=framework_descriptions
        )
        config["AUTOMATION_FRAMEWORK"] = framework
        
        # Show framework information
        self._show_framework_info(ui_helper, language, framework)
        
        return config
    
    def _configure_testing_environment(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure testing environment settings."""
        config = {}
        
        ui_helper.show_info(
            "Configure your testing environment:",
            ["Base URL for your application", "Timeout settings", "Environment variables"]
        )
        
        # Base URL configuration
        base_url = ui_helper.get_input(
            "Base URL for testing (where your app runs)",
            default="http://localhost:8000",
            validator=validate_url,
            error_message="Please enter a valid URL (http://... or https://...)"
        )
        config["BASE_URL"] = base_url
        
        
        return config
    
    def _configure_output_settings(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure output and file settings."""
        config = {}
        
        ui_helper.show_info(
            "Configure output settings:",
            ["Where to save generated tests", "File naming conventions"]
        )
        
        # Output directory
        output_dir = ui_helper.get_input(
            "Output directory for generated tests",
            default=DEFAULT_AUTOMATION_OUTPUT_DIR,
            validator=validate_directory_path,
            error_message="Please enter a valid directory path"
        )
        config["AUTOMATION_OUTPUT_DIR"] = output_dir
        
        return config
    
    def _configure_llm_enhancement(self, ui_helper: UIHelper) -> Dict[str, str]:
        """Configure LLM enhancement options."""
        config = {}
        
        try:
            # Check if LLM enhancement is available by trying to import LLM manager
            from ...core.llm.llm_manager import LLMManager
            
            # Test if LLM is configured
            try:
                llm_manager = LLMManager()
                llm_available = llm_manager is not None
            except:
                llm_available = False
            
            if llm_available:
                ui_helper.show_section_header(
                    "🤖 AI Enhancement Options",
                    "TestAutomator can use AI to enhance generated test code"
                )
                
                ui_helper.show_info(
                    "AI enhancement provides:",
                    [
                        "🧠 Improved error handling and edge cases",
                        "✅ Better assertions and validations",
                        "🎯 Framework-specific optimizations",
                        "📝 Enhanced code documentation"
                    ]
                )
                
                enable_enhancement = ui_helper.confirm(
                    "Enable AI enhancement by default for generated tests?",
                    default=False
                )
                
                config["AUTOMATION_ENHANCE_BY_DEFAULT"] = "true" if enable_enhancement else "false"
                
                if enable_enhancement:
                    ui_helper.show_success(
                        "AI enhancement enabled!",
                        ["Generated tests will be automatically enhanced", "You can still disable it per generation"]
                    )
            else:
                ui_helper.show_info(
                    "💡 AI Enhancement",
                    [
                        "Configure an LLM provider to enable AI-powered test enhancement",
                        "This will improve generated test quality and add best practices"
                    ]
                )
                
        except ImportError:
            # LLM enhancement not available - skip this section
            pass
        except Exception as e:
            logger.warning(f"Failed to check LLM enhancement availability: {e}")
        
        return config
    
    def _get_framework_descriptions(self, language: str) -> Dict[str, str]:
        """Get descriptions for frameworks in the given language."""
        descriptions = {
            "python": {
                "pytest": "Modern, feature-rich testing framework (recommended)",
                "unittest": "Built-in Python testing framework",
                "playwright": "Cross-browser E2E testing with Python",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "javascript": {
                "jest": "Popular testing framework by Facebook",
                "mocha": "Flexible testing framework with rich ecosystem", 
                "playwright": "Cross-browser E2E testing",
                "cypress": "Developer-friendly E2E testing",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "typescript": {
                "jest": "Popular testing framework with TypeScript support",
                "mocha": "Flexible testing framework with TypeScript support",
                "playwright": "Cross-browser E2E testing with TypeScript",
                "cypress": "Developer-friendly E2E testing with TypeScript",
                "cucumber": "Behavior-driven development (BDD) testing"
            },
            "java": {
                "junit5": "Modern Java testing framework (recommended)",
                "junit4": "Legacy Java testing framework",
                "testng": "Feature-rich testing framework for Java",
                "playwright": "Cross-browser E2E testing for Java",
                "karate": "API testing framework for Java",
                "cucumber": "Behavior-driven development (BDD) testing"
            }
        }
        
        return descriptions.get(language, {})
    
    def _show_framework_info(self, ui_helper: UIHelper, language: str, framework: str):
        """Show information about the selected framework."""
        framework_info = {
            ("python", "pytest"): [
                "✅ Rich plugin ecosystem",
                "✅ Fixtures and parametrization",
                "✅ Parallel test execution",
                "✅ Excellent reporting"
            ],
            ("python", "playwright"): [
                "✅ Cross-browser testing",
                "✅ Auto-waiting for elements",
                "✅ Network interception",
                "✅ Visual testing support"
            ],
            ("javascript", "jest"): [
                "✅ Snapshot testing",
                "✅ Built-in mocking",
                "✅ Code coverage reports",
                "✅ Watch mode for development"
            ],
            ("javascript", "playwright"): [
                "✅ Multi-browser support",
                "✅ Mobile testing",
                "✅ Network mocking",
                "✅ Trace viewer for debugging"
            ],
            ("java", "junit5"): [
                "✅ Modern annotations",
                "✅ Parameterized tests",
                "✅ Dynamic tests",
                "✅ Extension model"
            ],
            ("java", "karate"): [
                "✅ API testing without code",
                "✅ Built-in JSON/XML handling",
                "✅ Parallel execution",
                "✅ Comprehensive reporting"
            ]
        }
        
        info = framework_info.get((language, framework))
        if info:
            ui_helper.show_info(f"{framework.title()} features:", info)
    
    def _validate_configuration(self, config: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate TestAutomator configuration."""
        errors = []
        
        # Validate language
        language = config.get("AUTOMATION_LANGUAGE", "")
        if not language:
            errors.append("Automation language is required")
        elif language not in SUPPORTED_LANGUAGES:
            errors.append(f"Unsupported language: {language}")
        
        # Validate framework
        framework = config.get("AUTOMATION_FRAMEWORK", "")
        if not framework:
            errors.append("Automation framework is required")
        elif language and framework not in SUPPORTED_FRAMEWORKS.get(language, []):
            errors.append(f"Framework '{framework}' not supported for language '{language}'")
        
        # Validate base URL
        base_url = config.get("BASE_URL", "")
        if base_url and not validate_url(base_url):
            errors.append("Invalid base URL format")
        
        # Validate output directory
        output_dir = config.get("AUTOMATION_OUTPUT_DIR", "")
        if output_dir and not validate_directory_path(output_dir):
            errors.append("Invalid output directory path")
        
        return len(errors) == 0, errors
    
    def _show_configuration_summary(self, ui_helper: UIHelper, config: Dict[str, str]):
        """Show configuration summary."""
        language = config.get("AUTOMATION_LANGUAGE", "")
        framework = config.get("AUTOMATION_FRAMEWORK", "")
        base_url = config.get("BASE_URL", "")
        output_dir = config.get("AUTOMATION_OUTPUT_DIR", "")
        
        ui_helper.show_success(
            "TestAutomator (Automation) configuration completed!",
            [
                f"Language: {language}",
                f"Framework: {framework}",
                f"Base URL: {base_url}",
                f"Output directory: {output_dir}"
            ]
        )
        
        # Show usage examples
        ui_helper.show_info(
            "Usage examples:",
            [
                "testteller generate 'Create user registration tests'",
                "testteller automate test-cases.md",
                f"testteller automate test-cases.md --language {language} --framework {framework}"
            ]
        )
    
    def _get_default_config(self) -> Dict[str, str]:
        """Get default automation configuration."""
        return {
            "AUTOMATION_LANGUAGE": DEFAULT_AUTOMATION_LANGUAGE,
            "AUTOMATION_FRAMEWORK": DEFAULT_AUTOMATION_FRAMEWORK,
            "BASE_URL": "http://localhost:8000",
            "AUTOMATION_OUTPUT_DIR": DEFAULT_AUTOMATION_OUTPUT_DIR,
            "TEST_TIMEOUT": "30000",
            "AUTOMATION_ENHANCE_BY_DEFAULT": "false"
        }
    
    def get_configuration_summary(self) -> Dict[str, str]:
        """Get sanitized configuration summary for display."""
        return self.config_data.copy()


def validate_automation_config(config: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Standalone function to validate automation configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    wizard = TestAutomatorWizard()
    return wizard._validate_configuration(config)