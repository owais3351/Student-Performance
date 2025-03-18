import logging
import os
from datetime import datetime

# create LOG file with current date and time
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#Full file path for storing logfile inside directory-->{logs} within CWD:
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

#create a directory(logs) if it doesn't exist
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_path=os.path.join(logs_path,LOG_FILE)

#basic logging Configuration
logging.basicConfig(
    filename=LOG_FILE_path,
    format="[%(asctime)s] %(lineno)s %(name)s-%(levelname)s-%(message)s",
    level=logging.INFO
)
