import pytest
from prompts import TEST_CASE_GENERATION_PROMPT_TEMPLATE

def test_template_exists_and_is_string():
    assert TEST_CASE_GENERATION_PROMPT_TEMPLATE is not None
    assert isinstance(TEST_CASE_GENERATION_PROMPT_TEMPLATE, str)

def test_template_formatting():
    try:
        formatted_prompt = TEST_CASE_GENERATION_PROMPT_TEMPLATE.format(
            context="Test context",
            query="Test query"
        )
        assert "Test context" in formatted_prompt
        assert "Test query" in formatted_prompt
    except Exception as e:
        pytest.fail(f"Prompt formatting failed: {e}")

def test_template_placeholders():
    assert "{context}" in TEST_CASE_GENERATION_PROMPT_TEMPLATE
    assert "{query}" in TEST_CASE_GENERATION_PROMPT_TEMPLATE
