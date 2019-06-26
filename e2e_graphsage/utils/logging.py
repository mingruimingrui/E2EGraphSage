from __future__ import absolute_import

import logging


def setup_logging(log_path, mode='w'):
    fmt = '%(asctime)s %(levelname)-4.4s %(filename)s:%(lineno)d: %(message)s'
    logging.root.handlers = []
    logging.basicConfig(
        filename=log_path,
        filemode=mode,
        format=fmt,
        datefmt='%m-%d %H:%M',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    return logging.getLogger(__name__)
