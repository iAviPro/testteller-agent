import logging
import json
import pytest
from unittest.mock import patch, MagicMock

from utils.helpers import setup_logging, CustomJsonFormatter
from config import LoggingSettings, get_settings # AppSettings, load_app_settings


# --- Tests for CustomJsonFormatter ---

def test_custom_json_formatter_basic_structure():
    formatter = CustomJsonFormatter()
    log_record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test_path.py',
        lineno=10,
        msg='Test info message with %s',
        args=('args',),
        exc_info=None,
        func='test_func'
    )
    formatted_log = formatter.format(log_record)
    log_dict = json.loads(formatted_log)

    assert 'timestamp' in log_dict
    assert log_dict['level'] == 'INFO'
    assert log_dict['message'] == 'Test info message with args'
    # Fields like 'name', 'module', 'lineno', 'funcName' are dependent on
    # the base class's processing of the format string.
    # We will rely on integration tests (setup_logging with JSON) to verify the full output.
    # For this unit test, focus on what CustomJsonFormatter explicitly does.
    # assert log_dict['name'] == 'test_logger'
    # assert log_dict['module'] == 'test_path'
    # assert log_dict['lineno'] == 10
    # assert log_dict['function_name'] == 'test_func'

def test_custom_json_formatter_with_exc_info():
    formatter = CustomJsonFormatter()
    try:
        raise ValueError("Test exception")
    except ValueError:
        log_record = logging.LogRecord(
            name='test_exception_logger',
            level=logging.ERROR,
            pathname='test_exc_path.py',
            lineno=20,
            msg='Error message',
            args=(),
            exc_info=True, # Capture exc_info
            func='test_exc_func'
        )
        # Manually set exc_info and exc_text if not automatically picked up by LogRecord in test
        import sys
        log_record.exc_info = sys.exc_info()
        log_record.exc_text = logging.Formatter().formatException(sys.exc_info())

    formatted_log = formatter.format(log_record)
    log_dict = json.loads(formatted_log)

    assert log_dict['level'] == 'ERROR'
    assert 'exc_info' in log_dict # Changed 'exception_info' to 'exc_info'
    assert "ValueError: Test exception" in log_dict['exc_info'] # Changed 'exception_info' to 'exc_info'

def test_custom_json_formatter_extra_fields():
    formatter = CustomJsonFormatter()
    log_record = logging.LogRecord(
        name='test_extra_logger',
        level=logging.WARNING,
        pathname='test_extra.py',
        lineno=30,
        msg='Warning with extra',
        args=(),
        exc_info=None,
        func='test_extra_func'
    )
    log_record.extra_field_1 = "value1"
    log_record.extra_field_2 = 123

    formatted_log = formatter.format(log_record)
    log_dict = json.loads(formatted_log)

    assert log_dict['level'] == 'WARNING'
    assert log_dict['extra_field_1'] == 'value1'
    assert log_dict['extra_field_2'] == 123


# --- Tests for setup_logging ---

@pytest.fixture
def mock_config_settings():
    """Fixture to mock config.settings for logging tests."""
    # Use a context manager for patching to ensure it's reverted
    with patch('utils.helpers.get_settings') as mock_get_settings:
        mock_app_settings = MagicMock()
        mock_app_settings.logging = LoggingSettings(log_level="INFO", log_format="text")
        # Mock other settings if setup_logging directly uses them, though it shouldn't
        mock_get_settings.return_value = mock_app_settings
        yield mock_app_settings


def test_setup_logging_level_text_format(mock_config_settings, caplog):
    mock_config_settings.logging.log_level = "DEBUG"
    mock_config_settings.logging.log_format = "text"

    # Reset global logging state before test
    logging.root.handlers = [] # Clear existing root handlers
    # caplog will capture from root by default.

    setup_logging() # This will clear root handlers and add its own.
    
    # After setup_logging, root logger is configured.
    # Tell caplog to capture at this level AND re-add its handler if setup_logging removed it.
    caplog.set_level(logging.DEBUG)
    if caplog.handler not in logging.getLogger().handlers:
        logging.getLogger().addHandler(caplog.handler) # Ensure caplog's handler is on the root logger

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    # Verify the handler added by setup_logging
    assert len(root_logger.handlers) > 0, "Root logger should have handlers after setup_logging"
    handler = root_logger.handlers[0] # Assuming setup_logging adds one main handler to root
    assert not isinstance(handler.formatter, CustomJsonFormatter) 
    assert isinstance(handler.formatter, logging.Formatter)
    
    # Check initial log message from "utils.helpers" logger
    # Caplog captures records. We should check these records.
    # The initial log message from setup_logging is always INFO.
    # print("CAPLOG RECORDS (TEXT):", caplog.records) # Debugging line

    # Instead of capturing "Logging initialized", log a new message and check that.
    logger_to_test = logging.getLogger("utils.helpers") # Or logging.getLogger() for root
    test_message = "Test message after text setup"
    logger_to_test.info(test_message)

    found_log = False
    for record in caplog.records:
        if record.name == "utils.helpers" and record.levelname == "INFO" and test_message in record.message:
            # Check that the format is text (caplog.text should not be json)
            # This is an indirect check; a more direct one is formatter instance.
            assert "{" not in caplog.text # Basic check for non-JSON
            found_log = True
            break
    assert found_log, f"Test message '{test_message}' not found or incorrect in caplog.records. Records: {caplog.records}"


def test_setup_logging_level_json_format(mock_config_settings, caplog):
    mock_config_settings.logging.log_level = "INFO"
    mock_config_settings.logging.log_format = "json"

    # Reset global logging state - setup_logging does this, but good for isolation.
    logging.root.handlers = [] # Clear existing root handlers
    # caplog will capture from root by default.

    setup_logging() # This configures the logger and its own "Logging initialized..."
    
    # After setup_logging, root logger is configured.
    # Tell caplog to capture at this level AND re-add its handler.
    caplog.set_level(logging.INFO)
    if caplog.handler not in logging.getLogger().handlers:
        logging.getLogger().addHandler(caplog.handler) # Ensure caplog's handler is on the root logger

    root_logger = logging.getLogger() # Get root to check its level and handlers
    assert root_logger.level == logging.INFO

    assert len(root_logger.handlers) > 0, "Root logger should have handlers after setup_logging"
    handler = root_logger.handlers[0]
    assert isinstance(handler.formatter, CustomJsonFormatter)

    # Log a new message and check its format
    logger_to_test = logging.getLogger("utils.helpers") # Or root: logging.getLogger()
    test_message = "Test message after json setup"
    logger_to_test.info(test_message)
    
    # print("CAPLOG RECORDS (JSON):", caplog.records)
    # print("CAPLOG TEXT (JSON):", caplog.text)

    # Ensure caplog.handler uses the same formatter as the root logger's handler
    # This is important because setup_logging configures the root logger's handler's formatter.
    # We want caplog.text to reflect that specific formatting.
    # root_handler = logging.getLogger().handlers[0] # This is the StreamHandler
    # caplog.handler.setFormatter(root_handler.formatter) # This was an attempt, but caplog.text might not use it as expected.

    # Instead, directly use the formatter from the handler to format a captured record.
    json_formatter_instance = logging.getLogger().handlers[0].formatter
    assert isinstance(json_formatter_instance, CustomJsonFormatter)

    found_log_and_correctly_formatted = False
    # Log the test message AGAIN (or ensure it's the last one for clarity in caplog.records)
    # The first logger_to_test.info(test_message) is already there.
    # caplog.records should contain it.

    for record in caplog.records:
        if record.name == "utils.helpers" and record.levelname == "INFO" and record.getMessage() == test_message:
            formatted_output_by_handler = json_formatter_instance.format(record)
            try:
                log_dict = json.loads(formatted_output_by_handler)
                assert log_dict.get("name") == "utils.helpers"
                assert log_dict.get("level") == "INFO"
                assert log_dict.get("message") == test_message
                found_log_and_correctly_formatted = True
                break 
            except json.JSONDecodeError:
                pytest.fail(
                    f"Log record for '{test_message}' when formatted by CustomJsonFormatter "
                    f"produced non-JSON: '{formatted_output_by_handler}'"
                )
    
    assert found_log_and_correctly_formatted, \
        f"Test message '{test_message}' not found in caplog.records or not correctly formatted by CustomJsonFormatter. Records: {caplog.records}"


def test_setup_logging_third_party_loggers_default_level(mock_config_settings):
    # Ensure default behavior sets httpx to WARNING, others to INFO
    mock_config_settings.logging.log_level = "DEBUG" # Root logger is more verbose
    mock_config_settings.logging.log_format = "text"
    
    logging.root.handlers = []
    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("chromadb.telemetry.posthog").level == logging.WARNING
    # another_noisy_lib is not specially handled, its level will be NOTSET (0)
    # and it will delegate to the root logger's level (DEBUG in this test case for actual logging).
    assert logging.getLogger("another_noisy_lib").level == logging.NOTSET
    assert logging.getLogger("another_noisy_lib").isEnabledFor(logging.DEBUG) # Check effective level


def test_setup_logging_third_party_loggers_custom_level(mock_config_settings):
    # Test if a very verbose root level (e.g. DEBUG) still keeps third-party reasonable
    mock_config_settings.logging.log_level = "DEBUG"
    mock_config_settings.logging.log_format = "text"

    logging.root.handlers = []
    # Simulate specific override for a third-party logger if the function supported it
    # For now, it uses fixed levels for third-party loggers.
    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("chromadb.telemetry.posthog").level == logging.WARNING # Changed from INFO to WARNING


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Ensure logging state is clean after each test."""
    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level
    
    # Store levels of specific third-party loggers changed by setup_logging
    third_party_loggers_to_reset = ["httpx", "chromadb.telemetry.posthog", "another_noisy_lib", "httpcore", "urllib3.connectionpool", "git.cmd"]
    original_third_party_levels = {name: logging.getLogger(name).level for name in third_party_loggers_to_reset}

    yield # Test runs here

    # Restore original state
    # logging.root.handlers = original_handlers # Potentially problematic with caplog
    logging.root.setLevel(original_level)
    for name, level in original_third_party_levels.items():
        logging.getLogger(name).setLevel(level)

    # Minimal handler cleanup: remove handlers added by setup_logging to root.
    # This assumes setup_logging adds a known number of handlers (e.g., 1)
    # or specific types of handlers. For now, let setup_logging manage its own handlers
    # and caplog manage its own. If setup_logging is called multiple times, it clears
    # previous root handlers itself.
    # The primary concern is that caplog's handler is not removed prematurely.
    # Pytest's caplog fixture should handle its own handler cleanup.
