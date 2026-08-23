import logging
import os
from datetime import datetime

LOG_DIR = "logs"

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y-%m-%d')}.log"
)

# Log_FILE = F"{datetime.now().strftime('%Y-%m-%d')}.log"
# logs_path = os.path.join(os.getcwd(),"logs",Log_FILE)
# os.makedirs(logs_path,exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("mlProject")