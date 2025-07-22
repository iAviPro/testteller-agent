"""
Code loading utilities for ingesting code from various sources.
"""
import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from git import Repo

from testteller.config import ApiKeysSettings, CodeLoaderSettings, settings

logger = logging.getLogger(__name__)


class CodeLoader:
    """Handles loading code from various sources."""

    def __init__(self):
        """Initialize the code loader."""
        # Use system temporary directory with testteller prefix
        self.temp_dir = tempfile.mkdtemp(prefix="testteller_repos_")
        self.clone_dir_base = Path(self.temp_dir)
        logger.debug("Created temporary directory for repositories: %s", self.clone_dir_base)

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        # Remove .git extension if present
        repo_url = repo_url.rstrip(".git")
        # Get the last part of the URL/path
        return repo_url.split("/")[-1]

    async def _git_command_wrapper(self, func, *args, **kwargs):
        """Wrapper to run git commands asynchronously."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def clone_or_pull_repo(self, repo_url: str) -> Optional[Path]:
        """Clone a repository or pull latest changes if already cloned."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self.clone_dir_base / repo_name

        try:
            if local_repo_path.exists():
                logger.info(
                    "Repository %s exists. Pulling latest changes from %s.", repo_name, repo_url)
                repo = await self._git_command_wrapper(Repo, str(local_repo_path))
                origin = repo.remotes.origin
                await self._git_command_wrapper(origin.pull)
            else:
                logger.info(
                    "Cloning repository %s to %s.", repo_url, local_repo_path)
                clone_url = repo_url

                # Check if we have GitHub token configured
                github_token = None
                try:
                    if (settings and
                        settings.api_keys and
                            isinstance(settings.api_keys, ApiKeysSettings)):
                        settings_dict = settings.api_keys.__dict__
                        if settings_dict.get('github_token'):
                            github_token = settings_dict['github_token'].get_secret_value(
                            )
                except Exception as e:
                    logger.warning("Failed to access GitHub token: %s", e)

                # Use GitHub token for HTTPS URLs if available
                if github_token and "github.com" in repo_url and not repo_url.startswith("git@"):
                    # ensure token is not already in URL
                    if "@" not in repo_url.split("://")[1]:
                        protocol, rest = repo_url.split("://")
                        clone_url = f"{protocol}://oauth2:{github_token}@{rest}"
                        logger.info(
                            "Using GITHUB_TOKEN for HTTPS clone. Ensure token is not logged if clone_url is logged elsewhere.")

                await self._git_command_wrapper(Repo.clone_from, clone_url, str(local_repo_path))

            logger.info(
                "Repository %s is up to date at %s.", repo_name, local_repo_path)
            return local_repo_path
        except Exception as e:
            logger.error(
                "Failed to clone/pull repository %s: %s", repo_url, e, exc_info=True)
            # Clean up failed clone
            if local_repo_path.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, local_repo_path)
                except Exception as cleanup_e:
                    logger.warning(
                        "Failed to clean up failed clone at %s: %s", local_repo_path, cleanup_e)
            return None

    async def _read_code_files_from_path(self, base_path: Path, source_identifier: str) -> List[Tuple[str, str]]:
        """Read code files from a directory path."""
        if not settings:
            raise ValueError(
                "Settings not initialized. Please ensure .env file exists with required configurations.")

        if not isinstance(settings.code_loader, CodeLoaderSettings):
            raise ValueError("Invalid code_loader settings configuration")

        code_files = []
        try:
            # Get code extensions from settings
            settings_dict = settings.code_loader.__dict__
            extensions = settings_dict.get('code_extensions', [".py"])
            if not extensions:
                logger.warning(
                    "No code extensions configured. Using default: .py")
                extensions = [".py"]

            for root, _, files in os.walk(base_path):
                for file in files:
                    file_path = Path(root) / file
                    if any(file.endswith(ext) for ext in extensions):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if not content.strip():
                                    logger.debug(
                                        "Skipping empty file: %s", file_path)
                                    continue
                                relative_path = str(
                                    file_path.relative_to(base_path))
                                code_files.append(
                                    (f"{source_identifier}:{relative_path}", content))
                                logger.debug("Read file: %s", relative_path)
                        except Exception as e:
                            logger.warning(
                                "Failed to read file %s: %s", file_path, e)
                            continue

            if not code_files:
                logger.warning(
                    "No code files found in %s with extensions: %s", base_path, extensions)
            else:
                logger.info("Found %d code files in %s",
                            len(code_files), base_path)

        except Exception as e:
            logger.error("Error walking directory %s: %s", base_path, e)
            raise

        return code_files

    async def load_code_from_repo(self, repo_url: str) -> List[Tuple[str, str]]:
        """Load code from a remote GitHub repository."""
        local_repo_path = await self.clone_or_pull_repo(repo_url)
        if not local_repo_path:
            logger.error("Failed to clone/pull repository: %s", repo_url)
            return []
        return await self._read_code_files_from_path(local_repo_path, source_identifier=repo_url)

    async def load_code_from_local_folder(self, folder_path: str) -> List[Tuple[str, str]]:
        """Load code from a local folder path."""
        local_path = Path(folder_path)
        if not local_path.is_dir():
            logger.error(
                "Provided local path is not a directory or does not exist: %s", folder_path)
            return []

        # Use the absolute path of the folder as the source_identifier for uniqueness and clarity
        abs_folder_path_str = str(local_path.resolve())
        return await self._read_code_files_from_path(local_path, source_identifier=f"local:{abs_folder_path_str}")

    async def cleanup_repo(self, repo_url: str) -> None:
        """Clean up a cloned repository."""
        repo_name = self._get_repo_name_from_url(repo_url)
        local_repo_path = self.clone_dir_base / repo_name
        if local_repo_path.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, local_repo_path)
                logger.info("Cleaned up cloned repository: %s",
                            local_repo_path)
            except Exception as e:
                logger.error(
                    "Error cleaning up repository %s: %s", local_repo_path, e, exc_info=True)

    async def cleanup_all_repos(self) -> None:
        """Clean up all cloned repositories."""
        if self.clone_dir_base.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, self.clone_dir_base)
                self.clone_dir_base.mkdir(exist_ok=True)
                logger.info("Cleaned up all cloned repositories")
            except Exception as e:
                logger.error("Error cleaning up repositories: %s",
                             e, exc_info=True)
