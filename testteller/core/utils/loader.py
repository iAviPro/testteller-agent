import asyncio
import sys
import threading
import time
from typing import Any, Callable, Optional


class LoadingProgressBar:
    """A progress bar for CLI operations."""

    def __init__(self, message: str = "Loading...", width: int = 30):
        self.message = message
        self.width = width
        self.is_running = False
        self._thread = None
        self.start_time = None

    def _animate(self):
        """Internal method to display the progress bar animation."""
        position = 0
        direction = 1

        while self.is_running:
            # Create progress bar
            bar = ['░'] * self.width

            # Add animated block
            for i in range(3):  # 3-character wide animated block
                if 0 <= position + i < self.width:
                    bar[position + i] = '█'

            # Calculate elapsed time
            elapsed = time.time() - self.start_time if self.start_time else 0
            elapsed_str = f"{elapsed:.1f}s"

            # Display just the progress bar (message already printed in start())
            bar_str = ''.join(bar)
            output = f"\r[{bar_str}] {elapsed_str}"
            
            # Write output and immediately flush
            sys.stdout.write(output)
            sys.stdout.flush()

            # Update position
            position += direction
            if position >= self.width - 3:
                direction = -1
            elif position <= 0:
                direction = 1

            time.sleep(0.1)  # 100ms delay

        # Clear the progress bar when done
        sys.stdout.write("\r")
        sys.stdout.flush()


    def start(self):
        """Start the progress bar in a separate thread."""
        if not self.is_running:
            # Print initial message on new line
            print(f"{self.message}...")
            self.is_running = True
            self.start_time = time.time()
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the progress bar."""
        if self.is_running:
            self.is_running = False
            if self._thread:
                self._thread.join()
            # Clear the progress bar line and print completion
            sys.stdout.write("\r")
            sys.stdout.flush()
            print("✓ Complete")


class DeterminateProgressBar:
    """A determinate progress bar for operations with known progress."""

    def __init__(self, message: str = "Processing...", width: int = 30):
        self.message = message
        self.width = width
        self.progress = 0
        self.start_time = time.time()

    def update(self, progress: float):
        """Update progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        self._display()

    def _display(self):
        """Display the current progress."""
        filled = int(self.width * self.progress)
        progress_bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        percentage = int(self.progress * 100)

        sys.stdout.write(
            f"\r{self.message} [{progress_bar}] {percentage}% ({elapsed:.1f}s)")
        sys.stdout.flush()

    def finish(self):
        """Finish the progress bar."""
        self.update(1.0)
        sys.stdout.write("\n")
        sys.stdout.flush()


async def with_progress_bar(coro: Callable, message: str = "Processing...") -> Any:
    """
    Execute an async function with a progress bar.

    Args:
        coro: The async function to execute
        message: The loading message to display

    Returns:
        The result of the async function
    """
    progress_bar = LoadingProgressBar(message)

    try:
        progress_bar.start()
        result = await coro
        return result
    finally:
        progress_bar.stop()


def with_progress_bar_sync(func: Callable, message: str = "Processing...") -> Any:
    """
    Execute a sync function with a progress bar.

    Args:
        func: The sync function to execute
        message: The loading message to display

    Returns:
        The result of the function
    """
    progress_bar = LoadingProgressBar(message)

    try:
        progress_bar.start()
        result = func()
        return result
    finally:
        progress_bar.stop()


# Keep the old function names for backward compatibility
async def with_spinner(coro: Callable, message: str = "Processing...") -> Any:
    """Backward compatibility wrapper for with_progress_bar."""
    return await with_progress_bar(coro, message)


def with_spinner_sync(func: Callable, message: str = "Processing...") -> Any:
    """Backward compatibility wrapper for with_progress_bar_sync."""
    return with_progress_bar_sync(func, message)
