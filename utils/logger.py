import logging
import logging.config


def configure_logger(name, log_path):
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
    return logging.getLogger(name)

#
#
# class Logger:
#     """
#     Training process logger
#     Note:
#         Used by BaseTrainer to save training history.
#     """
#
#     def __init__(self):
#         self.entries = {}
#
#     def add_entry(self, entry):
#         logging.info('Hey, this is working!')
#
#         self.entries[len(self.entries) + 1] = entry
#
#     def __str__(self):
#         return json.dumps(self.entries, sort_keys=True, indent=4)
