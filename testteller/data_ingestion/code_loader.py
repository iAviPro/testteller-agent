"""Module for loading code files from GitHub repositories or local folders asynchronously."""

import shutil
import logging
import asyncio
from pathlib import Path
import aiofiles  # For async file reading
from git import Repo, exc as GitExc
from testteller.config import settings


logger = logging.getLogger(__name__)


class CodeLoader:
    def __init__(self):
        self.clone_dir_base = Path(settings.code_loader.temp_clone_dir_base)
        self.clone_dir_base.mkdir(parents=True, exist_ok=True)
        logger.info(
            "CodeLoader initialized. "
            "GitHub repos cloned to: %s. "
            "Local paths processed directly.",
            self.clone_dir_base
        )

    def _get_repo_name_from_url(self, repo_url: str) -> str:
        return repo_url.split('/')[-1].replace('.git', '')

    async def _git_command_wrapper(self, func, *args, **kwargs):
        """Wraps synchronous GitPython calls to run in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def clone_or_pull_repo(self, repo_url: str) -> Path | None:
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
                if settings.github_token and "github.com" in repo_url and not repo_url.startswith("git@"):
                    token = settings.github_token
                    # ensure token is not already in URL
                    if "@" not in repo_url.split("://")[1]:
                        protocol, rest = repo_url.split("://")
                        clone_url = f"{protocol}://oauth2:{token}@{rest}"
                        logger.info(
                            "Using GITHUB_TOKEN for HTTPS clone. Ensure token is not logged if clone_url is logged elsewhere.")

                await self._git_command_wrapper(Repo.clone_from, clone_url, str(local_repo_path))

            logger.info(
                "Repository %s is up to date at %s.", repo_name, local_repo_path)
            return local_repo_path
        except GitExc.GitCommandError as e:
            # Log stderr for more details
            logger.error(
                "Git command error for %s: %s", repo_url, e.stderr or e, exc_info=True)
            if "Authentication failed" in str(e.stderr):
                logger.error(
                    "Authentication failed. Ensure GITHUB_TOKEN is valid with 'repo' scope, or SSH keys are configured for git.")
            return None
        except Exception as e:
            logger.error(
                "Failed to clone/pull repository %s: %s", repo_url, e, exc_info=True)
            return None

    async def _read_code_files_from_path(self, base_path: Path, source_identifier: str) -> list[tuple[str, str]]:
        """
        Helper function to read code files from a given base_path.
        source_identifier is used for creating unique "source" metadata (e.g., repo URL or local folder path).
        """
        code_files_content = []
        logger.info(
            "Loading code files from %s with extensions: %s", base_path, settings.code_extensions)

        file_paths_to_read = []
        for item in base_path.rglob('*'):  # rglob for recursive search
            if item.is_file() and item.suffix.lower() in settings.code_extensions:
                file_paths_to_read.append(item)

        async def read_file_content(item_path: Path):
            # For local paths, relative_path is relative to the input base_path.
            # For GitHub, it's relative to the repo root.
            relative_path_str = str(item_path.relative_to(base_path))
            # Consistent source format
            full_source_path = f"{source_identifier}/{relative_path_str}"

            try:
                async with aiofiles.open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                return (full_source_path, content)
            except Exception as e:
                logger.warning(
                    "Could not read file %s (source: %s): %s", item_path, full_source_path, e, exc_info=True)
                return None

        tasks = [read_file_content(p) for p in file_paths_to_read]
        # Catch exceptions per task
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res_or_exc in enumerate(results):
            if isinstance(res_or_exc, Exception):
                logger.error(
                    "Failed to process file %s: %s", file_paths_to_read[i], res_or_exc)
            elif res_or_exc:  # If result is not None and not an exception
                code_files_content.append(res_or_exc)
                logger.debug(
                    "Loaded code file: %s", res_or_exc[0].split('/')[-1])

        logger.info(
            "Loaded %d code files from %s (path: %s)", len(code_files_content), source_identifier, base_path)
        return code_files_content

    async def load_code_from_repo(self, repo_url: str) -> list[tuple[str, str]]:
        """Loads code from a remote GitHub repository."""
        local_repo_path = await self.clone_or_pull_repo(repo_url)
        if not local_repo_path:
            return []
        # Use repo_url as the source_identifier for consistent metadata
        return await self._read_code_files_from_path(local_repo_path, source_identifier=repo_url)

    async def load_code_from_local_folder(self, folder_path: str) -> list[tuple[str, str]]:
        """Loads code from a local folder path."""
        local_path = Path(folder_path)
        if not local_path.is_dir():
            logger.error(
                "Provided local path is not a directory or does not exist: %s", folder_path)
            return []

        # Use the absolute path of the folder as the source_identifier for uniqueness and clarity
        abs_folder_path_str = str(local_path.resolve())
        return await self._read_code_files_from_path(local_path, source_identifier=f"local:{abs_folder_path_str}")

    async def cleanup_repo(self, repo_url: str):
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

    async def cleanup_all_repos(self):
        # This method primarily cleans up the temp_clone_dir_base, so it's fine as is.
        # It doesn't affect local folders that were read directly.
        if self.clone_dir_base.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, self.clone_dir_base)
                logger.info(
                    "Cleaned up all cloned repositories in: %s", self.clone_dir_base)
                # Recreate base dir (sync, fast)
                self.clone_dir_base.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    "Error cleaning up all repositories: %s", e, exc_info=True)
