import logging
import sys
from pathlib import Path

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_filepath = Path(log_dir) / "continuos_logs.log"

Path(log_dir).mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("summarizerlogger")
