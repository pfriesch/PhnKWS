import logging
import logging.config
import os

from data import kaldi_io


class Logger(object):

    def __init__(self) -> None:
        super().__init__()
        self.configure_logger(out_folder=None)

    def configure_logger(self, out_folder):
        if out_folder is not None:
            name = 'default'
            log_path = os.path.join(out_folder, 'log.log')
            logging.config.dictConfig({
                'version': 1,
                'formatters': {
                    'default': {'format': '%(asctime)s [%(levelname)s] %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'},
                    'brief': {'format': "%(asctime)s [%(levelname)s] %(message)s", 'datefmt': '%H:%M:%S',
                              "class": "utils.logger_format.ColoredFormatter"}
                },
                'handlers': {
                    'console': {
                        'level': 'DEBUG' if 'DEBUG_MODE' in os.environ and bool(int(os.environ['DEBUG_MODE']))
                        else 'INFO',
                        'class': 'utils.logger_format.TqdmLoggingHandler',
                        'formatter': 'brief'
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
        else:
            name = 'default'
            logging.config.dictConfig({
                'version': 1,
                'formatters': {
                    'brief': {'format': "[%(levelname)s]  %(message)s",
                              "class": "utils.logger_format.ColoredFormatter"}
                },
                'handlers': {
                    'console': {
                        'level': 'DEBUG',
                        'class': 'utils.logger_format.TqdmLoggingHandler',
                        'formatter': 'brief'
                    }
                },
                'loggers': {
                    'default': {
                        'level': 'DEBUG',
                        'handlers': ['console']
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
