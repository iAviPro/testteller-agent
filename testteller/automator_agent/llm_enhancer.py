"""
LLM integration for TestWriter to enhance generated test automation code.
This module provides optional LLM-powered enhancement of generated test code.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .prompts import (
    get_enhancement_prompt, 
    get_generation_prompt,
    get_test_optimization_suggestions,
    get_framework_imports
)

logger = logging.getLogger(__name__)


class TestEnhancer:
    """Enhances test automation code using LLM providers."""
    
    def __init__(self, provider: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the test enhancer.
        
        Args:
            provider: LLM provider name (gemini, openai, claude, llama)
            config: Configuration dictionary for LLM settings
        """
        self.provider = provider
        self.config = config or {}
        self.llm_manager = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM manager if available."""
        try:
            # Try to import and initialize LLM manager from testteller
            from testteller.core.llm.llm_manager import LLMManager
            from testteller.config import settings
            
            if self.provider:
                self.llm_manager = LLMManager(self.provider)
            else:
                # Use configured provider from settings
                self.llm_manager = LLMManager()
                self.provider = self.llm_manager.provider
            
            logger.info(f"LLM enhancer initialized with provider: {self.provider}")
            
        except ImportError as e:
            logger.warning(f"LLM integration not available: {e}")
            self.llm_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM manager: {e}")
            self.llm_manager = None
    
    def is_available(self) -> bool:
        """Check if LLM enhancement is available."""
        return self.llm_manager is not None
    
    def enhance_test_code(self, test_code: str, language: str, framework: str, 
                         test_type: str = "functional", app_type: str = "web application") -> Tuple[str, bool]:
        """
        Enhance existing test code using LLM.
        
        Args:
            test_code: Original test code to enhance
            language: Programming language (python, javascript, typescript, java)
            framework: Testing framework (pytest, jest, playwright, etc.)
            test_type: Type of test (functional, integration, e2e)
            app_type: Type of application being tested
            
        Returns:
            Tuple of (enhanced_code, success_flag)
        """
        if not self.is_available():
            logger.warning("LLM enhancement not available, returning original code")
            return test_code, False
        
        try:
            # Get optimized prompt for the provider
            prompt = get_enhancement_prompt(
                provider=self.provider,
                language=language,
                framework=framework,
                test_code=test_code,
                test_type=test_type,
                app_type=app_type
            )
            
            # Generate enhanced code
            enhanced_code = self.llm_manager.generate_text(prompt)
            
            if enhanced_code and len(enhanced_code.strip()) > len(test_code.strip()) * 0.5:
                logger.info("Test code successfully enhanced")
                return enhanced_code, True
            else:
                logger.warning("Enhancement produced insufficient content, using original")
                return test_code, False
                
        except Exception as e:
            logger.error(f"Failed to enhance test code: {e}")
            return test_code, False
    
    def generate_test_from_specs(self, test_specifications: str, language: str, framework: str,
                               test_type: str = "functional", app_type: str = "web application") -> Tuple[str, bool]:
        """
        Generate test code from specifications using LLM.
        
        Args:
            test_specifications: Test case specifications or requirements
            language: Programming language
            framework: Testing framework
            test_type: Type of test
            app_type: Type of application being tested
            
        Returns:
            Tuple of (generated_code, success_flag)
        """
        if not self.is_available():
            logger.warning("LLM generation not available")
            return "", False
        
        try:
            # Get optimized prompt for the provider
            prompt = get_generation_prompt(
                provider=self.provider,
                language=language,
                framework=framework,
                test_specifications=test_specifications,
                test_type=test_type,
                app_type=app_type
            )
            
            # Generate test code
            generated_code = self.llm_manager.generate_text(prompt)
            
            if generated_code and len(generated_code.strip()) > 100:  # Minimum viable code length
                logger.info("Test code successfully generated")
                return generated_code, True
            else:
                logger.warning("Generation produced insufficient content")
                return "", False
                
        except Exception as e:
            logger.error(f"Failed to generate test code: {e}")
            return "", False
    
    def get_optimization_suggestions(self, language: str, framework: str) -> List[str]:
        """
        Get optimization suggestions for the given language and framework.
        
        Args:
            language: Programming language
            framework: Testing framework
            
        Returns:
            List of optimization suggestions
        """
        try:
            suggestions = get_test_optimization_suggestions(self.provider, language, framework)
            return suggestions
        except Exception as e:
            logger.error(f"Failed to get optimization suggestions: {e}")
            return []
    
    def enhance_test_file(self, file_path: Path, language: str, framework: str) -> bool:
        """
        Enhance a complete test file.
        
        Args:
            file_path: Path to the test file
            language: Programming language
            framework: Testing framework
            
        Returns:
            True if enhancement was successful
        """
        if not file_path.exists():
            logger.error(f"Test file not found: {file_path}")
            return False
        
        try:
            # Read original content
            original_content = file_path.read_text(encoding='utf-8')
            
            # Enhance the content
            enhanced_content, success = self.enhance_test_code(
                test_code=original_content,
                language=language,
                framework=framework
            )
            
            if success and enhanced_content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                backup_path.write_text(original_content, encoding='utf-8')
                
                # Write enhanced content
                file_path.write_text(enhanced_content, encoding='utf-8')
                
                logger.info(f"Enhanced test file: {file_path}")
                logger.info(f"Backup created: {backup_path}")
                return True
            else:
                logger.warning(f"No enhancement applied to: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to enhance test file {file_path}: {e}")
            return False
    
    def enhance_generated_tests(self, generated_files: Dict[str, str], language: str, 
                              framework: str) -> Dict[str, str]:
        """
        Enhance all generated test files.
        
        Args:
            generated_files: Dictionary of filename -> content
            language: Programming language
            framework: Testing framework
            
        Returns:
            Dictionary of filename -> enhanced_content
        """
        if not self.is_available():
            logger.info("LLM enhancement not available, returning original files")
            return generated_files
        
        enhanced_files = {}
        
        for filename, content in generated_files.items():
            if filename.endswith(('.py', '.js', '.ts', '.java')):  # Test files
                logger.info(f"Enhancing test file: {filename}")
                
                enhanced_content, success = self.enhance_test_code(
                    test_code=content,
                    language=language,
                    framework=framework
                )
                
                if success:
                    enhanced_files[filename] = enhanced_content
                    logger.info(f"Successfully enhanced: {filename}")
                else:
                    enhanced_files[filename] = content
                    logger.warning(f"Enhancement failed for: {filename}, using original")
            else:
                # Non-test files (config, requirements, etc.) - keep original
                enhanced_files[filename] = content
        
        return enhanced_files
    
    async def enhance_test_code_async(self, test_code: str, language: str, framework: str,
                                    test_type: str = "functional", app_type: str = "web application") -> Tuple[str, bool]:
        """
        Async version of enhance_test_code.
        
        Args:
            test_code: Original test code to enhance
            language: Programming language
            framework: Testing framework
            test_type: Type of test
            app_type: Type of application being tested
            
        Returns:
            Tuple of (enhanced_code, success_flag)
        """
        if not self.is_available():
            return test_code, False
        
        try:
            # Get optimized prompt for the provider
            prompt = get_enhancement_prompt(
                provider=self.provider,
                language=language,
                framework=framework,
                test_code=test_code,
                test_type=test_type,
                app_type=app_type
            )
            
            # Generate enhanced code asynchronously
            enhanced_code = await self.llm_manager.generate_text_async(prompt)
            
            if enhanced_code and len(enhanced_code.strip()) > len(test_code.strip()) * 0.5:
                return enhanced_code, True
            else:
                return test_code, False
                
        except Exception as e:
            logger.error(f"Failed to enhance test code async: {e}")
            return test_code, False
    
    def add_imports_and_setup(self, test_code: str, language: str, framework: str) -> str:
        """
        Add proper imports and setup code to test files.
        
        Args:
            test_code: Original test code
            language: Programming language
            framework: Testing framework
            
        Returns:
            Test code with proper imports and setup
        """
        try:
            imports = get_framework_imports(language, framework)
            
            # Check if imports are already present
            if imports and not any(imp.strip() in test_code for imp in imports.split('\n') if imp.strip()):
                # Add imports at the beginning
                enhanced_code = f"{imports}\n\n{test_code}"
                return enhanced_code
            
            return test_code
            
        except Exception as e:
            logger.error(f"Failed to add imports and setup: {e}")
            return test_code


# Convenience functions for easy integration
def create_enhancer(provider: Optional[str] = None) -> TestEnhancer:
    """
    Create a test enhancer instance.
    
    Args:
        provider: LLM provider name (optional)
        
    Returns:
        TestEnhancer instance
    """
    return TestEnhancer(provider=provider)


def enhance_test_with_llm(test_code: str, language: str, framework: str, 
                         provider: Optional[str] = None) -> Tuple[str, bool]:
    """
    Convenience function to enhance test code with LLM.
    
    Args:
        test_code: Original test code
        language: Programming language
        framework: Testing framework
        provider: LLM provider (optional)
        
    Returns:
        Tuple of (enhanced_code, success_flag)
    """
    enhancer = create_enhancer(provider)
    return enhancer.enhance_test_code(test_code, language, framework)


def is_llm_enhancement_available() -> bool:
    """
    Check if LLM enhancement is available.
    
    Returns:
        True if LLM enhancement can be used
    """
    try:
        enhancer = create_enhancer()
        return enhancer.is_available()
    except Exception:
        return False