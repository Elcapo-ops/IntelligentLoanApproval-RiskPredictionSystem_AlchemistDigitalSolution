"""Structured project-wide logging setup."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from config.paths import get_project_paths
from config.settings import validate_config

_MANAGED_HANDLER_ATTRIBUTE = "_loan_prediction_managed_handler"


class JsonLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize timestamp, severity, logger, message, and exception."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for field in ("event", "seed", "component"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: dict[str, Any]) -> logging.Logger:
    """Configure console and file logging from validated settings.

    Existing handlers created by this helper are replaced to honor changed
    settings without duplicating output. The configured log directory is
    created through the shared project path manager.
    """
    validate_config(config)
    logging_settings = config["logging"]
    paths = get_project_paths(config)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("loan_prediction")
    logger.setLevel(logging_settings["level"].upper())
    logger.propagate = False

    for handler in logger.handlers[:]:
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()

    if logging_settings["format"] == "json":
        formatter: logging.Formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(logging_settings["text_format"])

    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(paths.log_file, encoding="utf-8"),
    )
    for handler in handlers:
        setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
