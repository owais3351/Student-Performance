import logging
import os
from datetime import datetime

#create log-file based on current date and time:
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#Full file path for Storing log-file inside Logs directory within CWD:
logs_path=os.path.join(os.getcwd(),"Logs",LOG_FILE)

#create directory(Logs) if it do not exist:
os.makedirs(logs_path,LOG_FILE) 

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)


#Basic logging configuration:
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("logging has started")
