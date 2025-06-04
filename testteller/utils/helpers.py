import logging
import sys
from pythonjsonlogger import jsonlogger
from testteller.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(
            log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = record.created
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


def setup_logging():
    """Configures logging based on settings."""
    log_level_str = str(settings.logging.log_level).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a handler (console for CLI)
    handler = logging.StreamHandler(sys.stdout)

    if str(settings.logging.log_format).lower() == "json":
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s')
    else:  # text format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Silence overly verbose third-party loggers
    # httpx is used by google-generativeai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
    logging.getLogger("git.cmd").setLevel(
        logging.INFO)  # GitPython can be verbose

    # Initial log to confirm setup
    initial_logger = logging.getLogger(__name__)
    log_format = getattr(getattr(settings, "logging", {}),
                         "log_format", "text")
    initial_logger.info(
        "Logging initialized. Level: %s, Format: %s", log_level_str, log_format)

# Call setup_logging when this module is imported so it's configured early
# setup_logging() # Or call explicitly in main.py before anything else
