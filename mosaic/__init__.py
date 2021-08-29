import logging

from .master import Master
from .mosaic import MosaicRegular
from .pool import Pool

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
