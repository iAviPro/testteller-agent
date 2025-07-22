"""
TestTeller RAG-Enhanced Automator Agent Package.

This package provides RAG-enhanced test automation generation using vector store knowledge:
- Complete working test code generation (no templates or TODOs)
- Real application context discovery from vector stores
- Multi-language and framework support
- Quality validation and assessment
"""

from .cli import automate_command

__all__ = ["automate_command"]