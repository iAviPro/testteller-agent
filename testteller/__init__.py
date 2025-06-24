"""
TestTeller RAG Agent
A versatile CLI-based RAG (Retrieval Augmented Generation) agent designed to generate software test cases.
"""

# Import version from the single source of truth
from ._version import __version__

__author__ = "Aviral Nigam"
__license__ = "Apache License 2.0"
__url__ = "https://github.com/iAviPro/testteller-rag-agent"
__description__ = "TestTeller : A versatile RAG AI agent for generating test cases from project documentation (PRDs, Contracts, Design Docs, etc.) and project code, leveraging LLMs."

# Make version easily accessible
from .constants import APP_NAME, APP_DESCRIPTION

# Update APP_VERSION in constants to use the version from here
APP_VERSION = __version__
