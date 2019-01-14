import logging
import logging.config

import os

from data_loader import kaldi_io


class Logger(object):

    def configure_logger(self, out_folder):
        name = 'default'
        log_path = os.path.join(out_folder, 'log.log')
        logging.config.dictConfig({
            'version': 1,
            'formatters': {
                'default': {'format': '%(asctime)s [%(levelname)s] %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'},
                'brief': {'format': "[%(levelname)s]  %(message)s",
                          "class": "utils.logger_format.ColoredFormatter"}
            },
            'handlers': {
                'console': {
                    'level': 'INFO',
                    'class': 'logging.StreamHandler',
                    'formatter': 'brief',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'level': 'DEBUG',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'default',
                    'filename': log_path,
                }
            },
            # matplotlib logs to the root logger too, in debug mode
            # 'root': {
            #     'level': 'DEBUG',
            #     'handlers': ['console', 'file']
            #
            # },
            'loggers': {
                'default': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file']
                }
            },
            'disable_existing_loggers': False
        })
        self.logger = logging.getLogger(name)
        kaldi_io.logger = self

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level):
        return self.logger.isEnabledFor(level)


logger = Logger()