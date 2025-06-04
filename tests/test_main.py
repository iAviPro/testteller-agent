import pytest
import logging # Added import
from unittest import mock # To be used by mocker or directly
from pathlib import Path # For mocking Path objects

from typer.testing import CliRunner
from main import app # Assuming your Typer app instance is named 'app' in main.py
from config import settings as global_app_settings # To get default collection name

# We need to mock TestTellerRagAgent before it's used by the commands in main.py
# The agent is imported in main.py. So, we need to patch it in main's namespace.

@pytest.fixture(scope="function") # Changed scope to function
def mock_agent_class_for_main(mocker): # Renamed to avoid conflict if other test files use similar name
    """Mocks the TestTellerRagAgent class in the main module."""
    mock_agent_cls = mocker.patch("main.TestTellerRagAgent")
    mock_agent_instance = mocker.MagicMock(name="MockTestTellerRagAgentInstanceInMain")
    mock_agent_cls.return_value = mock_agent_instance

    # Define async methods on the MOCK INSTANCE
    mock_agent_instance.ingest_documents_from_path = mocker.AsyncMock(name="ingest_docs_mock")
    mock_agent_instance.ingest_code_from_source = mocker.AsyncMock(name="ingest_code_mock")
    mock_agent_instance.generate_test_cases = mocker.AsyncMock(name="generate_tests_mock", return_value="Generated test cases from mock agent.")
    mock_agent_instance.get_ingested_data_count = mocker.AsyncMock(name="get_count_mock", return_value=0)
    mock_agent_instance.clear_ingested_data = mocker.AsyncMock(name="clear_data_mock")
    # Add vector_store.db_path attribute for the status command test
    mock_agent_instance.vector_store = mocker.MagicMock()
    mock_agent_instance.vector_store.db_path = "mock/db/path_from_fixture"
    return mock_agent_cls

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def reset_main_agent_instance_mocks(mock_agent_class_for_main, mocker): # Added mocker here for vector_store re-mock
    """Resets mocks on the agent INSTANCE before each test."""
    mock_instance = mock_agent_class_for_main.return_value
    methods_to_reset = [
        'ingest_documents_from_path', 'ingest_code_from_source',
        'generate_test_cases', 'get_ingested_data_count', 'clear_ingested_data'
    ]
    for method_name in methods_to_reset:
        getattr(mock_instance, method_name).reset_mock()

    mock_instance.get_ingested_data_count.return_value = 0
    mock_instance.generate_test_cases.return_value = "Generated test cases from mock agent."

    if hasattr(mock_instance, 'vector_store') and isinstance(mock_instance.vector_store, mock.MagicMock):
        mock_instance.vector_store.db_path = "mock/db/path_from_fixture"
    else:
        mock_instance.vector_store = mocker.MagicMock()
        mock_instance.vector_store.db_path = "mock/db/path_from_fixture"


# --- Tests for 'ingest-docs' command ---
def test_ingest_docs_success(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "dummy_docs_path/"
    mocker.patch("os.path.exists", return_value=True)

    result = runner.invoke(app, ["ingest-docs", test_path, "--collection-name", "custom_docs"])

    assert result.exit_code == 0, result.stdout
    # Check new success message
    assert f"Successfully ingested documents. Collection 'custom_docs' now contains 0 items." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="custom_docs")
    mock_instance.ingest_documents_from_path.assert_called_once_with(test_path)

def test_ingest_docs_non_existent_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "non_existent_path/"
    mocker.patch("os.path.exists", return_value=False) # os.path.exists is used in main.py for this check
    result = runner.invoke(app, ["ingest-docs", test_path])
    assert result.exit_code == 1
    assert f"Error: Path does not exist: {test_path}" in result.stdout # Updated error message
    mock_instance.ingest_documents_from_path.assert_not_called()

# --- Tests for 'ingest-code' command ---
def test_ingest_code_success_url(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_url = "https://github.com/example/repo.git"
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("main.Path", lambda p: mocker.MagicMock(exists=lambda: False, __str__=lambda: p))

    result = runner.invoke(app, ["ingest-code", test_url, "--collection-name", "custom_code", "--no-cleanup-github"])
    assert result.exit_code == 0, result.stdout
    # Check new success message
    assert f"Successfully ingested code from '{test_url}'. Collection 'custom_code' now contains 0 items." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="custom_code")
    mock_instance.ingest_code_from_source.assert_called_once_with(test_url, cleanup_github_after=False)

def test_ingest_code_success_local_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "dummy_code_path/"
    mock_path_obj = mocker.MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.__str__.return_value = test_path
    mocker.patch("main.Path", lambda p_arg: mock_path_obj if str(p_arg) == test_path else mocker.MagicMock(exists=lambda:False))

    result = runner.invoke(app, ["ingest-code", test_path, "--no-cleanup-github"])
    assert result.exit_code == 0, result.stdout
    default_col_name = global_app_settings.chroma_db.default_collection_name
    # Check new success message
    assert f"Successfully ingested code from '{test_path}'. Collection '{default_col_name}' now contains 0 items." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name=default_col_name)
    mock_instance.ingest_code_from_source.assert_called_once_with(test_path, cleanup_github_after=False) # cleanup_github_after is False due to --no-cleanup-github

def test_ingest_code_non_existent_local_path(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    test_path = "non_existent_code_path/"
    mocker.patch("main.Path", lambda p: mocker.MagicMock(exists=lambda: False, __str__=lambda:p))

    result = runner.invoke(app, ["ingest-code", test_path])
    assert result.exit_code == 1
    assert f"Error: Local source path '{test_path}' not found or not accessible." in result.stdout # Updated error message
    mock_instance.ingest_code_from_source.assert_not_called()

# --- Tests for 'generate' command ---
def test_generate_success(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 5 # Simulate non-empty collection
    query = "Test query for generation"
    result = runner.invoke(app, ["generate", query, "--num-retrieved", "3", "--collection-name", "gen_coll"])
    assert result.exit_code == 0, result.stdout
    assert "Generated test cases from mock agent." in result.stdout
    mock_agent_class_for_main.assert_called_with(collection_name="gen_coll")
    mock_instance.generate_test_cases.assert_called_once_with(query, n_retrieved_docs=3)

def test_generate_empty_collection_confirm_yes(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    mocker.patch("typer.confirm", return_value=True) # Simulate user confirming "yes"
    default_col_name = global_app_settings.chroma_db.default_collection_name
    result = runner.invoke(app, ["generate", "query"]) # Uses default collection
    assert result.exit_code == 0, result.stdout
    assert f"Warning: Collection '{default_col_name}' is empty. Generation will rely on LLM's general knowledge." in result.stdout # Updated message
    assert "Generated test cases from mock agent." in result.stdout
    mock_instance.generate_test_cases.assert_called_once_with("query", n_retrieved_docs=5)

def test_generate_empty_collection_confirm_no(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    mocker.patch("typer.confirm", return_value=False) # Simulate user confirming "no"
    result = runner.invoke(app, ["generate", "query"])
    assert result.exit_code == 1, result.stdout # typer.Abort() exits with 1
    assert "Aborted." in result.stdout # Typer's default message for Abort
    mock_instance.generate_test_cases.assert_not_called()

# --- Tests for 'status' command ---
def test_status_with_data(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 120
    result = runner.invoke(app, ["status", "--collection-name", "status_coll"])
    assert result.exit_code == 0, result.stdout
    assert "Collection 'status_coll' contains 120 ingested items." in result.stdout # Updated message
    assert "ChromaDB persistent path: mock/db/path_from_fixture" in result.stdout # Check mock path
    mock_agent_class_for_main.assert_called_with(collection_name="status_coll")
    mock_instance.get_ingested_data_count.assert_called_once()

def test_status_empty_kb(runner, mock_agent_class_for_main):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 0
    default_col_name = global_app_settings.chroma_db.default_collection_name
    result = runner.invoke(app, ["status"]) # Uses default collection
    assert result.exit_code == 0, result.stdout
    assert f"Collection '{default_col_name}' contains 0 ingested items." in result.stdout # Updated message

# --- Tests for 'clear-data' command ---
def test_clear_data_force(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    confirm_mock = mocker.patch("typer.confirm") # To ensure it's NOT called
    result = runner.invoke(app, ["clear-data", "--collection-name", "clear_coll", "--force"])
    assert result.exit_code == 0, result.stdout
    assert "Successfully cleared data from collection 'clear_coll'." in result.stdout # Updated message
    mock_agent_class_for_main.assert_called_with(collection_name="clear_coll")
    mock_instance.clear_ingested_data.assert_called_once()
    confirm_mock.assert_not_called()

def test_clear_data_confirm_yes(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_confirm = mocker.patch("typer.confirm", return_value=True) # Simulate user saying yes
    default_col_name = global_app_settings.chroma_db.default_collection_name
    result = runner.invoke(app, ["clear-data"]) # Uses default collection
    assert result.exit_code == 0, result.stdout
    assert f"Successfully cleared data from collection '{default_col_name}'." in result.stdout # Updated message
    mock_confirm.assert_called_once()
    mock_instance.clear_ingested_data.assert_called_once()

def test_clear_data_confirm_no(runner, mock_agent_class_for_main, mocker):
    mock_instance = mock_agent_class_for_main.return_value
    mock_confirm = mocker.patch("typer.confirm", return_value=False) # Simulate user saying no
    result = runner.invoke(app, ["clear-data"])
    assert result.exit_code == 1, result.stdout # typer.Abort() exits with 1
    assert "Aborted." in result.stdout # Typer's default message for Abort
    mock_confirm.assert_called_once()
    mock_instance.clear_ingested_data.assert_not_called()

# --- Test missing arguments (Typer handling) ---
def test_missing_argument_ingest_docs(runner):
    result = runner.invoke(app, ["ingest-docs"])
    assert result.exit_code != 0
    assert "Missing argument 'PATH'." in result.stdout

def test_missing_argument_generate(runner):
    result = runner.invoke(app, ["generate"])
    assert result.exit_code != 0
    assert "Missing argument 'QUERY'." in result.stdout

# --- New Tests ---

def test_get_agent_failure_exits(runner, mock_agent_class_for_main, mocker, caplog):
    caplog.set_level(logging.ERROR)
    # Configure the class mock to raise an exception when an instance is created
    mock_agent_class_for_main.side_effect = Exception("Agent init failed")

    # Any command that calls _get_agent, e.g., status
    result = runner.invoke(app, ["status", "--collection-name", "fail_coll"])

    assert result.exit_code == 1
    # Check the updated error message from main.py's _get_agent
    assert "Error: Could not initialize agent. Check logs and GOOGLE_API_KEY. Details: Agent init failed" in result.stdout

    # Check logs
    assert any("Failed to initialize TestCaseAgent for collection 'fail_coll': Agent init failed" in record.message for record in caplog.records)

def test_generate_success_with_output_file(runner, mock_agent_class_for_main, mocker, tmp_path):
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 1 # Non-empty collection
    mock_instance.generate_test_cases.return_value = "Awesome test cases."

    output_file = tmp_path / "tests.txt"

    mock_open_func = mocker.patch("builtins.open", mocker.mock_open())

    result = runner.invoke(app, ["generate", "query", "--output-file", str(output_file)])

    assert result.exit_code == 0, result.stdout
    assert "Awesome test cases." in result.stdout
    assert f"Test cases saved to: {str(output_file)}" in result.stdout
    mock_open_func.assert_called_once_with(str(output_file), 'w', encoding='utf-8')
    mock_open_func().write.assert_called_once_with("Awesome test cases.")

def test_generate_success_with_output_file_error_saving(runner, mock_agent_class_for_main, mocker, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 1
    mock_instance.generate_test_cases.return_value = "Valid test cases."

    output_file = tmp_path / "protected_tests.txt"

    # Make open() raise an IOError
    mocker.patch("builtins.open", side_effect=IOError("Permission denied"))

    result = runner.invoke(app, ["generate", "query", "--output-file", str(output_file)])

    assert result.exit_code == 0, result.stdout # Command itself might succeed, but prints error
    assert "Valid test cases." in result.stdout # Test cases are still printed
    assert f"Error: Could not save test cases to {str(output_file)}: Permission denied" in result.stdout
    assert any(f"Failed to save test cases to {str(output_file)}: Permission denied" in record.message for record in caplog.records)


def test_generate_error_response_no_output_file(runner, mock_agent_class_for_main, mocker, tmp_path, caplog):
    caplog.set_level(logging.WARNING) # Client logs warning when not saving due to error
    mock_instance = mock_agent_class_for_main.return_value
    mock_instance.get_ingested_data_count.return_value = 1
    # Simulate agent returning an error message
    mock_instance.generate_test_cases.return_value = "Error: LLM failed to generate content."

    output_file = tmp_path / "error_tests.txt"

    mock_open_func = mocker.patch("builtins.open", mocker.mock_open())

    result = runner.invoke(app, ["generate", "query", "--output-file", str(output_file)])

    assert result.exit_code == 0, result.stdout
    assert "Error: LLM failed to generate content." in result.stdout # Agent's error is printed
    # Ensure file saving was not attempted or was warned about
    mock_open_func.assert_not_called()
    assert f"Warning: Test case generation seems to have failed. Not saving to {output_file}" in result.stdout
    assert any(f"LLM generation resulted in an error, not saving to file" in record.message for record in caplog.records)
