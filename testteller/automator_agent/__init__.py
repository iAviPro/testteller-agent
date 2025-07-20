"""
TestTeller Automator Agent Package.

This package contains all automation-specific functionality including:
- Test automation generators for different languages and frameworks
- Automation configuration and wizards
- CLI commands for test automation
- LLM enhancement for generated test code
"""

from .cli import automate_command

__all__ = ["automate_command"]