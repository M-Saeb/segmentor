import logging
import datetime
import os

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output based on log level"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def get_dual_logger(name: str):
    """
    Create a logger that writes to both the console (bash) and a log file under 'logs/'.
    Args:
        name: Logger name
        log_filename: Name of the log file (default: 'app.log')
    Returns:
        logging.Logger instance
    """
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Get debug level from environment variables (default: INFO)
    debug_level_str = os.environ.get('DEBUG_LEVEL', 'INFO').upper()
    debug_level = getattr(logging, debug_level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(debug_level)
    logger.handlers.clear()  # Avoid duplicate handlers if called multiple times

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(debug_level)
    ch_formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(debug_level)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
