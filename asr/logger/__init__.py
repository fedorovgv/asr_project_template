import logging

from .utils import *


asr_logger = logging.getLogger('asr_logger')
asr_logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s - %(name)s - %(levelname)s] :: %(message)s'
)
ch.setFormatter(formatter)
asr_logger.addHandler(ch)
