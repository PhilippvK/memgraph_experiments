import logging

from ..settings import Settings

logger = logging.getLogger("validate_settings")


def validate_settings(settings: Settings):
    logger.info("Validating settings...")
    settings.validate()
