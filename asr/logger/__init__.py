import logging

from .utils import *


logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s - %(name)s - %(levelname)s] :: %(message)s'
)
ch.setFormatter(formatter)
logger.addHandler(ch)
