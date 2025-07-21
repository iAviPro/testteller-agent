"""
TestTeller RAG Agent
A versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases.
"""

# Import version from the single source of truth
from ._version import __version__

__author__ = "Aviral Nigam"
__license__ = "Apache License 2.0"
__url__ = "https://github.com/iAviPro/testteller-agent"
__description__ = "TestTeller: Modular AI-powered test automation platform with generator_agent for intelligent test case generation and automator_agent for automated code generation. Supports multiple LLM providers: Gemini, OpenAI, Claude, and Llama."

# Make version easily accessible
try:
    from testteller.core.constants import APP_NAME, APP_DESCRIPTION
except ImportError:
    APP_NAME = "TestTeller"
    APP_DESCRIPTION = "TestTeller : A versatile RAG AI agent for generating test cases"

# Update APP_VERSION in constants to use the version from here
APP_VERSION = __version__

# Import core modules for easy access
try:
    from . import core
    from . import generator_agent as agent
    from . import automator_agent
except ImportError:
    pass  # Modules may not be available in all environments
