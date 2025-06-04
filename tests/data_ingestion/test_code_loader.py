import pytest
import asyncio
import os
import shutil
from unittest import mock
from pathlib import Path
import sys
import git
import types

try:
    import aiofiles
    import aiofiles.os as aio_os
except ImportError:
    aiofiles = mock.MagicMock()
    aio_os = mock.MagicMock()

from git import exc as ActualGitExceptions

git_module_mock = mock.MagicMock(name="git_module_mock")
git_repo_class_mock = mock.MagicMock(name="Repo_class_mock")
git_module_mock.Repo = git_repo_class_mock
git_module_mock.exc = ActualGitExceptions
git_repo_instance_mock = mock.MagicMock(name="Repo_instance_mock")
git_repo_class_mock.clone_from = mock.MagicMock(name="clone_from_mock_global")

original_sys_modules_git = sys.modules.get('git')
sys.modules['git'] = git_module_mock

from data_ingestion.code_loader import CodeLoader
from data_ingestion.code_loader import Repo as CodeLoaderRepo # Get the Repo CodeLoader uses

from config import CodeLoaderSettings, ApiKeysSettings

if original_sys_modules_git:
    sys.modules['git'] = original_sys_modules_git
else:
    if 'git' in sys.modules:
        del sys.modules['git']

import logging

@pytest.fixture
def mock_code_loader_settings(monkeypatch, mocker):
    mock_api_keys_instance = mocker.MagicMock(spec=ApiKeysSettings)
    configured_mock_github_secret_str = mocker.MagicMock()
    configured_mock_github_secret_str.get_secret_value.return_value = "mock_token"
    mock_api_keys_instance.github_token = configured_mock_github_secret_str
    mock_google_api_key_secretstr = mocker.MagicMock()
    mock_google_api_key_secretstr.get_secret_value.return_value = "dummy_google_key"
    mock_api_keys_instance.google_api_key = mock_google_api_key_secretstr

    mock_settings_object = mocker.MagicMock()
    mock_settings_object.code_loader = CodeLoaderSettings(
        code_extensions=['.py', '.js'],
        temp_clone_dir_base='./mock_temp_clone_dir_test'
    )
    mock_settings_object.api_keys = mock_api_keys_instance

    monkeypatch.setattr("data_ingestion.code_loader.settings", mock_settings_object)
    return mock_settings_object

@pytest.fixture
def code_loader(mock_code_loader_settings, tmp_path, mocker):
    CodeLoader.cloned_repos.clear()
    loader = CodeLoader()
    instance_specific_temp_dir = tmp_path / "test_cloned_repos"
    instance_specific_temp_dir.mkdir()
    loader.temp_clone_dir_base = instance_specific_temp_dir
    yield loader

@pytest.fixture
def mock_aiofiles_open_fixture(mocker): # Renamed to avoid conflict if test uses same name
    mock_open_func = mocker.MagicMock(name="mock_aiofiles_open_function_for_fixture")
    mocker.patch("data_ingestion.code_loader.aiofiles.open", new=mock_open_func)
    return mock_open_func


def test_codeloader_init(code_loader, mock_code_loader_settings):
    assert code_loader.code_extensions == mock_code_loader_settings.code_loader.code_extensions
    assert code_loader.temp_clone_dir_base.name == "test_cloned_repos"
    assert CodeLoader.cloned_repos == set()

@pytest.mark.parametrize("filepath, is_supported_in_mock", [
    ("test.py", True), ("path/to/file.js", True), ("archive.zip", False),
    ("nodotextension", False), ("image.JPG", False), ("path/to/.env", False),
    ("file.with.dots.py", True), ("file.with.dots.txt", False),
    ("unsupported.with.dots.exe", False),
])
def test_is_supported_file(code_loader, filepath, is_supported_in_mock):
    assert code_loader._is_supported_file(filepath) == is_supported_in_mock

@pytest.mark.asyncio
async def test_read_code_files_from_path_mixed_files(code_loader, caplog, mocker):
    base_path_str = "/testrepo"

    file_mocks_data = [
        {"path_str": "/testrepo/file1.py", "is_file": True, "content": "python content"},
        {"path_str": "/testrepo/module/file2.js", "is_file": True, "content": "javascript content"},
        {"path_str": "/testrepo/data.unsupported", "is_file": True},
        {"path_str": "/testrepo/docs", "is_file": False},
        {"path_str": "/testrepo/empty.py", "is_file": True, "content": ""},
        {"path_str": "/testrepo/error.py", "is_file": True, "error": IOError("mock read error")},
    ]

    rglob_results = []
    for data in file_mocks_data:
        p_mock = mocker.MagicMock(spec=Path)
        p_mock.is_file.return_value = data["is_file"]
        p_mock.__str__.return_value = data["path_str"]
        p_mock.name = os.path.basename(data["path_str"])
        p_mock.suffix = Path(data["path_str"]).suffix
        rglob_results.append(p_mock)

    # Patch aiofiles.open directly in the module where CodeLoader uses it
    mock_aiofiles_open_in_module = mocker.patch("data_ingestion.code_loader.aiofiles.open")

    def custom_aio_open_factory(filepath_obj, mode, encoding, errors):
        file_data = next((d for d in file_mocks_data if str(filepath_obj) == d["path_str"]), None)

        # This is the mock for the file object itself (like 'f' in 'async with ... as f:')
        async_file_reader_mock = mocker.AsyncMock()
        if file_data and "error" in file_data:
            async_file_reader_mock.read = mocker.AsyncMock(side_effect=file_data["error"])
        elif file_data:
            async_file_reader_mock.read = mocker.AsyncMock(return_value=file_data.get("content", "default_factory_content"))
        else:
            async_file_reader_mock.read = mocker.AsyncMock(return_value="unspecified_file_content")

        # This is the mock for the async context manager object that aiofiles.open() returns
        # It needs __aenter__ and __aexit__ methods.
        # __aenter__ should be an async method returning the file_reader_mock.
        async_context_manager_mock = mocker.AsyncMock()
        async_context_manager_mock.__aenter__.return_value = async_file_reader_mock
        async_context_manager_mock.__aexit__ = mocker.AsyncMock(return_value=None)

        return async_context_manager_mock

    mock_aiofiles_open_in_module.side_effect = custom_aio_open_factory
    caplog.set_level(logging.INFO)

    mock_base_path_instance = mocker.MagicMock(spec=Path)
    mock_base_path_instance.rglob.return_value = rglob_results
    mock_base_path_instance.__str__.return_value = base_path_str

    def mock_relative_to(self_path_mock, other_base_path_mock):
        if other_base_path_mock is mock_base_path_instance:
            item_path_str = str(self_path_mock)
            base_path_str_to_replace = str(other_base_path_mock)
            if item_path_str.startswith(base_path_str_to_replace):
                return Path(item_path_str[len(base_path_str_to_replace):].lstrip("/"))
            return Path(item_path_str)
        raise ValueError("relative_to called with unexpected base")

    for p_mock in rglob_results:
        if p_mock.is_file():
             p_mock.relative_to = mock_relative_to.__get__(p_mock, type(p_mock))

    results_list = await code_loader._read_code_files_from_path(mock_base_path_instance, "mock_source_id")

    mock_base_path_instance.rglob.assert_called_once_with("*")

    successful_reads = {item[0]: item[1] for item in results_list}
    assert ("mock_source_id/file1.py", "python content") in successful_reads.items()
    assert ("mock_source_id/module/file2.js", "javascript content") in successful_reads.items()
    assert ("mock_source_id/empty.py", "") in successful_reads.items()
    assert len(results_list) == 3

    assert "Could not read file /testrepo/error.py" in caplog.text # For IOError

@pytest.mark.asyncio
async def test_read_code_files_from_path_unicode_error(code_loader, caplog, mocker):
    base_path_str = "/testrepo_unicode"
    unicode_error_file = {"path_str": "/testrepo_unicode/unicode_error.py", "is_file": True, "error": UnicodeDecodeError("utf-8", b"\x80abc", 0, 1, "invalid start byte")}

    file_mocks_data = [ unicode_error_file ]
    rglob_results = []
    for data in file_mocks_data:
        p_mock = mocker.MagicMock(spec=Path)
        p_mock.is_file.return_value = data["is_file"]
        p_mock.__str__.return_value = data["path_str"]
        p_mock.name = os.path.basename(data["path_str"])
        p_mock.suffix = Path(data["path_str"]).suffix
        rglob_results.append(p_mock)

    mock_aiofiles_open_in_module = mocker.patch("data_ingestion.code_loader.aiofiles.open")

    def custom_aio_open_factory(filepath_obj, mode, encoding, errors):
        file_data = next((d for d in file_mocks_data if str(filepath_obj) == d["path_str"]), None)
        async_file_reader_mock = mocker.AsyncMock()
        if file_data and "error" in file_data and isinstance(file_data["error"], UnicodeDecodeError):
            # Make .read() raise the UnicodeDecodeError
            async_file_reader_mock.read = mocker.AsyncMock(side_effect=file_data["error"])
        else: # Should not happen in this test
            async_file_reader_mock.read = mocker.AsyncMock(return_value="valid content")

        async_context_manager_mock = mocker.AsyncMock()
        async_context_manager_mock.__aenter__.return_value = async_file_reader_mock
        async_context_manager_mock.__aexit__ = mocker.AsyncMock(return_value=None)
        return async_context_manager_mock

    mock_aiofiles_open_in_module.side_effect = custom_aio_open_factory
    caplog.set_level(logging.WARNING) # Changed to WARNING to see if the generic handler catches it

    mock_base_path_instance = mocker.MagicMock(spec=Path)
    mock_base_path_instance.rglob.return_value = rglob_results
    mock_base_path_instance.__str__.return_value = base_path_str

    def mock_relative_to(self_path_mock, other_base_path_mock):
        if other_base_path_mock is mock_base_path_instance:
            item_path_str = str(self_path_mock)
            base_path_str_to_replace = str(other_base_path_mock)
            if item_path_str.startswith(base_path_str_to_replace):
                return Path(item_path_str[len(base_path_str_to_replace):].lstrip("/"))
            return Path(item_path_str)
        raise ValueError("relative_to called with unexpected base")

    for p_mock in rglob_results:
        if p_mock.is_file():
             p_mock.relative_to = mock_relative_to.__get__(p_mock, type(p_mock))

    results_list = await code_loader._read_code_files_from_path(mock_base_path_instance, "mock_source_unicode_error")

    assert len(results_list) == 0 # File with UnicodeDecodeError should be skipped
    # Check for the generic warning message and parts of the actual UnicodeDecodeError string representation
    assert "Could not read file /testrepo_unicode/unicode_error.py (source: mock_source_unicode_error/unicode_error.py)" in caplog.text
    assert "'utf-8' codec can't decode byte 0x80" in caplog.text # Specific to the error raised


@pytest.mark.asyncio
async def test_load_code_from_local_folder_valid_path(code_loader, monkeypatch, mocker):
    local_path_str = "/mocked/local/folder"
    mock_path_instance = mocker.MagicMock(spec=Path)
    mock_path_instance.is_dir.return_value = True
    mock_path_instance.resolve.return_value = mock_path_instance
    monkeypatch.setattr("data_ingestion.code_loader.Path", lambda p: mock_path_instance if p == local_path_str else Path(p))
    expected_files_data = [("local:/mocked/local/folder/file1.py", "content")]
    mocker.patch.object(code_loader, '_read_code_files_from_path', return_value=expected_files_data)
    results = await code_loader.load_code_from_local_folder(local_path_str)
    assert results == expected_files_data
    code_loader._read_code_files_from_path.assert_called_once_with(
        mock_path_instance, source_identifier=f"local:{str(mock_path_instance)}"
    )

@pytest.mark.asyncio
async def test_load_code_from_local_folder_non_existent(code_loader, monkeypatch, caplog):
    local_path_str = "/non/existent/folder"
    mock_path_instance = mock.MagicMock(spec=Path);
    mock_path_instance.is_dir.return_value = False
    mock_path_instance.exists.return_value = False
    monkeypatch.setattr("data_ingestion.code_loader.Path", lambda p: mock_path_instance)
    results = await code_loader.load_code_from_local_folder(local_path_str)
    assert results == []
    assert "Provided local path is not a directory or does not exist" in caplog.text


class TestCodeLoaderClonePullSuite:
    @pytest.mark.asyncio
    async def test_clone_or_pull_repo_clones_new_with_token(self, code_loader, mocker, caplog, mock_code_loader_settings):
        mock_code_loader_settings.api_keys.github_token.get_secret_value.return_value = "mock_token_for_test"
        repo_url = "https://github.com/user/repo.git"
        repo_name = "repo"
        expected_path = code_loader.temp_clone_dir_base / repo_name

        def path_side_effect(p):
            path_str = str(p)
            if path_str == str(expected_path):
                mp = mocker.MagicMock(spec=Path)
                mp.exists.return_value = False
                mp.__str__.return_value = str(expected_path)
                return mp
            elif path_str == str(code_loader.temp_clone_dir_base):
                 mp_base = mocker.MagicMock(spec=Path)
                 mp_base.exists.return_value = True
                 mp_base.is_dir.return_value = True
                 mp_base.__truediv__ = lambda s, k: expected_path
                 return mp_base
            return Path(p)

        mocker.patch("data_ingestion.code_loader.Path", side_effect=path_side_effect)

        git_module_mock.Repo.clone_from.reset_mock()
        git_module_mock.Repo.clone_from.side_effect = None
        git_module_mock.Repo.clone_from.return_value = None

        returned_path = await code_loader.clone_or_pull_repo(repo_url)
        assert returned_path == expected_path
        expected_clone_url = f"https://oauth2:mock_token_for_test@github.com/user/repo.git"
        git_module_mock.Repo.clone_from.assert_called_once_with(expected_clone_url, str(expected_path))

    @pytest.mark.asyncio
    async def test_clone_or_pull_repo_pulls_existing(self, code_loader, mocker, caplog):
        repo_url = "https://github.com/user/repo2.git"
        repo_name = "repo2"
        expected_path = code_loader.temp_clone_dir_base / repo_name

        # Ensure the directory exists, so CodeLoader takes the "pull" path
        expected_path.mkdir(parents=True, exist_ok=True)

        # mock_path_instance_repo is not needed if we make the path actually exist.
        # The Path patching is removed for this test to simplify and use real Path object interactions.

        mock_repo_instance_local = mocker.MagicMock(spec=git.Repo) # This will be the return of Repo(path)
        mock_repo_instance_local.remotes = mocker.MagicMock()
        mock_origin_remote = mocker.MagicMock()
        mock_repo_instance_local.remotes.origin = mock_origin_remote

        # Create an explicit MagicMock for the pull method
        mock_pull_fn = mocker.MagicMock(name="ExplicitPullFunctionMock")
        # Assign this explicit mock to the 'pull' attribute of mock_origin_remote
        mock_origin_remote.pull = mock_pull_fn

        async def git_wrapper_side_effect(func, *args_wrapper, **kwargs_wrapper):
            if func is CodeLoaderRepo:
                assert str(args_wrapper[0]) == str(expected_path), \
                    f"Repo constructor called with {args_wrapper[0]} instead of {expected_path}"
                return mock_repo_instance_local
            elif func is mock_pull_fn: # Crucially, check identity with our explicit mock
                # This path should be taken.
                # The side_effect (this function) must now simulate the execution of mock_pull_fn
                # if we want mock_pull_fn to register a call.
                return func(*args_wrapper, **kwargs_wrapper) # Call mock_pull_fn and return its result
            elif func is git_module_mock.Repo.clone_from:
                 raise AssertionError("Repo.clone_from called unexpectedly in pull path.")
            raise AssertionError(f"Unexpected call to _git_command_wrapper with {func}")

        mocker.patch.object(code_loader, '_git_command_wrapper', side_effect=git_wrapper_side_effect)

        returned_path = await code_loader.clone_or_pull_repo(repo_url)
        assert returned_path == expected_path
        mock_pull_fn.assert_called_once() # Assert that our explicit mock was called

    @pytest.mark.asyncio
    async def test_clone_or_pull_repo_pull_fails(self, code_loader, mocker, caplog):
        repo_url = "https://github.com/user/repo_pull_fail.git"
        repo_name = "repo_pull_fail"
        expected_path = code_loader.temp_clone_dir_base / repo_name
        expected_path.mkdir(parents=True, exist_ok=True) # Path exists

        mock_repo_instance_local = mocker.MagicMock(spec=git.Repo)
        mock_repo_instance_local.remotes = mocker.MagicMock()
        mock_origin_remote = mocker.MagicMock()
        mock_repo_instance_local.remotes.origin = mock_origin_remote

        mock_pull_fn = mocker.MagicMock(name="ExplicitPullFunctionMockFail")
        # Simulate pull failing with a GitCommandError
        mock_pull_fn.side_effect = ActualGitExceptions.GitCommandError("pull", "failed pull", stderr="simulated pull error")
        mock_origin_remote.pull = mock_pull_fn

        async def git_wrapper_side_effect(func, *args_wrapper, **kwargs_wrapper):
            if func is CodeLoaderRepo:
                return mock_repo_instance_local
            elif func is mock_pull_fn:
                # This will now raise the GitCommandError because of mock_pull_fn.side_effect
                return func(*args_wrapper, **kwargs_wrapper)
            raise AssertionError(f"Unexpected call to _git_command_wrapper with {func}")

        mocker.patch.object(code_loader, '_git_command_wrapper', side_effect=git_wrapper_side_effect)
        caplog.set_level(logging.ERROR)

        returned_path = await code_loader.clone_or_pull_repo(repo_url)

        assert returned_path is None # Should return None on pull failure
        mock_pull_fn.assert_called_once()
        # Check for the specific log message for GitCommandError
        assert f"Git command error for {repo_url}:" in caplog.text
        assert "simulated pull error" in caplog.text # Check if stderr from exception is in log


    @pytest.mark.asyncio
    async def test_clone_or_pull_repo_git_auth_failure(self, code_loader, mocker, caplog, mock_code_loader_settings):
        caplog.set_level(logging.ERROR)
        mock_code_loader_settings.api_keys.github_token.get_secret_value.return_value = "mock_token_for_auth_fail"
        repo_url = "https://github.com/user/repo_auth_fail.git"

        mocker.patch("data_ingestion.code_loader.Path.exists", return_value=False)

        async def mock_wrapper_that_fails(func, *args, **kwargs):
            if func == git_module_mock.Repo.clone_from:
                raise ActualGitExceptions.GitCommandError(
                    command=['clone'], status=128, stderr='Authentication failed', stdout=''
                )
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
        mocker.patch.object(code_loader, '_git_command_wrapper', side_effect=mock_wrapper_that_fails)

        returned_path = await code_loader.clone_or_pull_repo(repo_url)
        assert returned_path is None
        assert "Authentication failed. Ensure GITHUB_TOKEN is valid" in caplog.text

class TestCodeLoaderLoadRepoSuite:
    @pytest.mark.asyncio
    async def test_load_code_from_repo_success(self, code_loader, mocker, caplog):
        repo_url = "mock_repo_url"
        mock_cloned_path = code_loader.temp_clone_dir_base / "mock_repo"
        mocker.patch.object(code_loader, 'clone_or_pull_repo', new_callable=mocker.AsyncMock, return_value=mock_cloned_path)
        mocker.patch.object(code_loader, '_read_code_files_from_path', new_callable=mocker.AsyncMock, return_value=[("file.py", "content")])
        results = await code_loader.load_code_from_repo(repo_url)
        assert results == [("file.py", "content")]
        code_loader.clone_or_pull_repo.assert_called_once_with(repo_url)
        code_loader._read_code_files_from_path.assert_called_once_with(mock_cloned_path, source_identifier=repo_url)

    @pytest.mark.asyncio
    async def test_load_code_from_repo_clone_fails(self, code_loader, mocker, caplog):
        repo_url = "mock_repo_url_fail_clone"
        mocker.patch.object(code_loader, 'clone_or_pull_repo', new_callable=mocker.AsyncMock, return_value=None)
        mock_read_files = mocker.patch.object(code_loader, '_read_code_files_from_path', new_callable=mocker.AsyncMock)
        results = await code_loader.load_code_from_repo(repo_url)
        assert results == []
        code_loader.clone_or_pull_repo.assert_called_once_with(repo_url)
        mock_read_files.assert_not_called()

@pytest.mark.asyncio
async def test_cleanup_repo(code_loader, monkeypatch, mocker):
    repo_url = "httpsgitscheme://github.com/user/repo_cleanup.git"
    repo_name = code_loader._get_repo_name_from_url(repo_url)
    temp_repo_path = code_loader.temp_clone_dir_base / repo_name
    temp_repo_path.mkdir(parents=True, exist_ok=True)
    CodeLoader.cloned_repos.add(str(temp_repo_path))
    mock_rmtree = mocker.patch("shutil.rmtree")

    await code_loader.cleanup_repo(repo_url)
    mock_rmtree.assert_called_once_with(temp_repo_path)
    assert str(temp_repo_path) not in CodeLoader.cloned_repos

@pytest.mark.asyncio
async def test_cleanup_repo_shutil_error(code_loader, mocker, caplog): # Removed self if not in a class
    repo_url = "httpsgitscheme://github.com/user/repo_cleanup_error.git"
    repo_name = code_loader._get_repo_name_from_url(repo_url)
    temp_repo_path = code_loader.temp_clone_dir_base / repo_name

    # Ensure the path exists and is tracked for cleanup
    temp_repo_path.mkdir(parents=True, exist_ok=True)
    CodeLoader.cloned_repos.add(str(temp_repo_path))

    mock_rmtree = mocker.patch("shutil.rmtree", side_effect=OSError("Disk full"))
    caplog.set_level(logging.ERROR)

    await code_loader.cleanup_repo(repo_url)

    mock_rmtree.assert_called_once_with(temp_repo_path)
    assert f"Error cleaning up repository {temp_repo_path}: Disk full" in caplog.text
    # If shutil.rmtree fails, the path should remain in cloned_repos as the .remove() line is skipped.
    assert str(temp_repo_path) in CodeLoader.cloned_repos


@pytest.mark.asyncio
async def test_cleanup_all_repos(code_loader, mocker):
    (code_loader.temp_clone_dir_base / "repo1").mkdir(parents=True, exist_ok=True)
    (code_loader.temp_clone_dir_base / "repo2").mkdir(parents=True, exist_ok=True)
    mock_rmtree = mocker.patch("shutil.rmtree")

    await code_loader.cleanup_all_repos()
    mock_rmtree.assert_called_once_with(code_loader.temp_clone_dir_base)
    assert code_loader.temp_clone_dir_base.exists() # Base dir is recreated

@pytest.mark.asyncio
async def test_cleanup_all_repos_shutil_error(code_loader, mocker, caplog):
    # Ensure the base directory exists to attempt cleanup
    if not code_loader.temp_clone_dir_base.exists():
        code_loader.temp_clone_dir_base.mkdir(parents=True, exist_ok=True)

    mock_rmtree = mocker.patch("shutil.rmtree", side_effect=OSError("Cleanup all failed"))
    caplog.set_level(logging.ERROR)

    await code_loader.cleanup_all_repos()

    mock_rmtree.assert_called_once_with(code_loader.temp_clone_dir_base)
    assert "Error cleaning up all repositories: Cleanup all failed" in caplog.text
    # The base directory might still exist if rmtree fails before recreation,
    # or it might be recreated if failure is after rmtree but before mkdir.
    # Current code: recreates mkdir even if rmtree fails (try/except doesn't prevent mkdir).
    # So, it should exist.
    assert code_loader.temp_clone_dir_base.exists()
