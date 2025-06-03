import pytest
import logging
from unittest import mock # For older Python versions, else use unittest.mock
import sys # Added import for sys

from utils.helpers import CustomJsonFormatter, setup_logging
# Assuming config.py defines a base Settings class or similar for type hinting
# and that we can monkeypatch specific settings values for tests.

# Fixture to mock settings for logging tests
@pytest.fixture
def mock_log_settings(monkeypatch):
    class MockLoggingSettings:
        def __init__(self, level="INFO", format="text"):
            self.log_level = level
            self.log_format = format

    class MockSettings:
        def __init__(self, log_level="INFO", log_format="text"):
            self.logging = MockLoggingSettings(level=log_level, format=log_format)

    # Use a function to allow parametrization of settings
    def _mock_settings(log_level="INFO", log_format="text"):
        settings_instance = MockSettings(log_level, log_format)
        monkeypatch.setattr("utils.helpers.settings", settings_instance)
        return settings_instance
    return _mock_settings

# --- Tests for CustomJsonFormatter ---

def test_custom_json_formatter_adds_timestamp():
    formatter = CustomJsonFormatter()
    log_record_dict = {}
    record = logging.LogRecord(
        name='test', level=logging.INFO, pathname='test.py', lineno=10,
        msg='Test message', args=(), exc_info=None, func='test_func'
    )
    # Simulate how add_fields is called by the parent class
    # The actual 'timestamp' field might be populated by the parent or here.
    # We ensure that if it's not there, our method adds it.
    # If the parent class JsonFormatter already adds 'timestamp', this test might need adjustment
    # to check our specific logic if it differs or enhances.

    # Let's assume the base class doesn't add 'timestamp', and we want to test our addition
    if 'timestamp' in log_record_dict: # Should not be the case if testing our addition
        del log_record_dict['timestamp']

    formatter.add_fields(log_record_dict, record, {}) # message_dict is empty for this test

    assert 'timestamp' in log_record_dict
    assert log_record_dict['timestamp'] == record.created

def test_custom_json_formatter_uppercases_level():
    formatter = CustomJsonFormatter()
    log_record_dict = {'level': 'info'} # Start with a level
    record = logging.LogRecord( # Dummy record, not directly used for level here
        name='test', level=logging.INFO, pathname='test.py', lineno=10,
        msg='Test message', args=(), exc_info=None, func='test_func'
    )
    formatter.add_fields(log_record_dict, record, {})
    assert log_record_dict['level'] == 'INFO'

def test_custom_json_formatter_adds_level_if_missing():
    formatter = CustomJsonFormatter()
    log_record_dict = {} # No level initially
    record = logging.LogRecord(
        name='test', level=logging.WARNING, pathname='test.py', lineno=10,
        msg='Test message', args=(), exc_info=None, func='test_func'
    )
    formatter.add_fields(log_record_dict, record, {})
    assert 'level' in log_record_dict
    assert log_record_dict['level'] == record.levelname # e.g., "WARNING"

# --- Tests for setup_logging ---

@pytest.mark.parametrize("log_level_str, expected_level_val", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
    ("CRITICAL", logging.CRITICAL),
    ("INVALID_LEVEL", logging.INFO), # Default for invalid
])
def test_setup_logging_level(mock_log_settings, log_level_str, expected_level_val):
    mock_log_settings(log_level=log_level_str)
    with mock.patch("logging.root.handlers", []): # Ensure no pre-existing handlers
        setup_logging()
    assert logging.getLogger().getEffectiveLevel() == expected_level_val

@pytest.mark.parametrize("log_format_str, expected_formatter_class_name", [
    ("json", "CustomJsonFormatter"),
    ("text", "Formatter"),
    ("INVALID_FORMAT", "Formatter"), # Default for invalid
])
def test_setup_logging_formatter(mock_log_settings, log_format_str, expected_formatter_class_name):
    mock_log_settings(log_format=log_format_str)
    # Patch specific loggers that setup_logging tries to modify
    with mock.patch("logging.getLogger") as mock_get_logger, \
         mock.patch("logging.root.handlers", []): # Clear handlers for clean test

        # Mock the behavior of getLogger for the root logger and others
        root_logger_mock = mock.MagicMock()
        # Make getLogger return the root_logger_mock when called without args or with specific names
        # For simplicity, we assume the first handler added to root is the one we care about
        mock_get_logger.return_value = root_logger_mock

        setup_logging()

        assert root_logger_mock.addHandler.called
        added_handler = root_logger_mock.addHandler.call_args[0][0]
        assert added_handler.formatter.__class__.__name__ == expected_formatter_class_name


def test_setup_logging_removes_existing_handlers(mock_log_settings):
    mock_log_settings() # Default settings
    # Add a dummy handler to the root logger
    dummy_handler = logging.StreamHandler()
    logging.root.addHandler(dummy_handler)
    assert dummy_handler in logging.root.handlers

    with mock.patch("logging.root.handlers", [dummy_handler]): # Ensure patch sees this handler
        setup_logging()

    # This assertion is tricky because setup_logging itself manipulates logging.root.handlers.
    # The key is that setup_logging should *start* by removing handlers.
    # A better way is to mock removeHandler and check it was called.
    with mock.patch.object(logging.root, 'removeHandler') as mock_remove_handler, \
         mock.patch.object(logging.root, 'addHandler') as mock_add_handler: # also mock add
        logging.root.handlers = [dummy_handler] # Reset for this test context
        setup_logging()
        mock_remove_handler.assert_called_with(dummy_handler)


def test_setup_logging_silences_third_party_loggers(mock_log_settings):
    mock_log_settings() # Default settings
    loggers_to_silence = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "chromadb.telemetry.posthog": logging.WARNING,
        "urllib3.connectionpool": logging.INFO,
        "git.cmd": logging.INFO
    }

    with mock.patch("logging.getLogger") as mock_get_logger:
        # Create mocks for each third-party logger
        mocked_loggers = {name: mock.MagicMock() for name in loggers_to_silence.keys()}

        # getLogger should return the appropriate mock, or a default mock if name is not in our list
        def mock_get_logger_side_effect(name=None):
            if name in mocked_loggers:
                return mocked_loggers[name]
            # For other loggers (like root or __name__ in setup_logging), return a new MagicMock
            return mock.MagicMock()

        mock_get_logger.side_effect = mock_get_logger_side_effect

        setup_logging()

        for name, level in loggers_to_silence.items():
            assert mocked_loggers[name].setLevel.called
            # Ensure it was called with the correct level
            # This part can be tricky if setLevel is called multiple times or if the mock
            # isn't specific enough. For this example, we assume one call after setup.
            # A more robust check might involve inspecting call_args_list.
            # For simplicity, checking it was called is a good start.
            # To check the specific level, you'd do:
            mocked_loggers[name].setLevel.assert_any_call(level)


import io # For capturing stdout

def test_setup_logging_initial_log_message(mock_log_settings, monkeypatch): # Removed caplog, added monkeypatch
    mock_log_settings(log_level="DEBUG", log_format="text")

    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    setup_logging() # This will now write its initial log to fake_stdout

    log_output = fake_stdout.getvalue()

    expected_message_part_level = "Level: DEBUG"
    expected_message_part_format = "Format: text"
    expected_name = "utils.helpers" # The logger is logging.getLogger(__name__) from utils.helpers

    # Check for the initial log message from utils.helpers in the captured stdout
    assert "Logging initialized" in log_output
    assert expected_message_part_level in log_output
    assert expected_message_part_format in log_output
    # The name 'utils.helpers' and level 'INFO' would be part of the formatted text message
    assert expected_name in log_output
    assert "INFO" in log_output # The log level name
