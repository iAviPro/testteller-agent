"""
Package initialization for testteller.agent.
"""
from .testteller_agent import TestTellerAgent

# Re-export for backward compatibility
TestTellerRagAgent = TestTellerAgent
__all__ = ['TestTellerAgent', 'TestTellerRagAgent']
