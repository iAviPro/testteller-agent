"""
TestTeller RAG Agent
A versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases.
"""

# Import version from the single source of truth
from ._version import __version__

# Import constants from single source of truth
try:
    from .core.constants import APP_NAME, APP_DESCRIPTION, APP_SHORT_DESCRIPTION
except ImportError:
    # Fallback constants if import fails
    APP_NAME = "TestTeller"
    APP_DESCRIPTION = "TestTeller: Your Next-Generation AI-Powered Test Agent for Comprehensive Test Case Generation and Test Automation leveraging RAG & GenAI"
    APP_SHORT_DESCRIPTION = "Next-Generation AI-Powered Test Agent for Test Cases Generation and Test Automation"

__author__ = "Aviral Nigam"
__license__ = "Apache License 2.0"
__url__ = "https://github.com/iAviPro/testteller-agent"
# Use description from constants
__description__ = APP_DESCRIPTION

# Update APP_VERSION in constants to use the version from here
APP_VERSION = __version__

# Import core modules for easy access
try:
    from . import core
    from . import generator_agent as agent
    from . import automator_agent
except ImportError:
    pass  # Modules may not be available in all environments
