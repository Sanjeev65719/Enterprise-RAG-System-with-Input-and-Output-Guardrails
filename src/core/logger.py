import logging
import sys
import yaml
from pathlib import Path

def setup_logging():
    """Sets up the system-wide logging configuration."""
    config_path = Path("config/config.yaml")

    # Default config if file is missing
    log_level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_cfg = config.get('system', {}).get('logging', {})
            log_level = log_cfg.get('level', 'INFO')
            log_format = log_cfg.get('format', log_format)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("system.log")
        ]
    )

    logger = logging.getLogger("EnterpriseRAG")
    logger.info("System logging initialized successfully.")
    return logger

# Global logger instance
logger = setup_logging()
