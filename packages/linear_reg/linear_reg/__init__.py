import logging
import os

from linear_reg.config import config
from linear_reg.config import logging_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False

fh = logging.FileHandler('log.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

with open(os.path.join(config.PACKAGE_ROOT,'VERSION')) as version_file:
    __version__ = version_file.read().strip()
    
