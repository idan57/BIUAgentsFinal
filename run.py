import logging
from pathlib import Path

from view.hr_app import open_gui
import numpy as np

MIN_RAND = 1000
MAX_RAND = 1000000


def set_logger():
    logs_folder = Path("logs")

    if not logs_folder.is_dir():
        logs_folder.mkdir()
    log_file = str(logs_folder / f"example_{np.random.randint(MIN_RAND, MAX_RAND)}.log")
    logging.basicConfig(format="[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s",
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()],
                        level=logging.DEBUG)


if __name__ == '__main__':
    set_logger()
    logging.info("********************************** OPENING APP **********************************")
    open_gui()
    logging.info("********************************** CLOSING APP **********************************")
