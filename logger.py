import logging
import sys
from config import Config


class Logger():

    logger_stdout = None

    [staticmethod]
    def init():

        handler=logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s'))
        Logger.logger_stdout=logging.getLogger('NER')
        Logger.logger_stdout.propagate = False
        Logger.logger_stdout.addHandler(handler)
        Logger.logger_stdout.setLevel(logging.getLevelName(Config.getstr('general', 'log_level')))

    [staticmethod]
    def debug(*arg):
        if len(arg)>1:
            Logger.logger_stdout.debug(', '.join([str(a) for a in arg]))
        else:
            Logger.logger_stdout.debug(arg[0] if len(arg)>0 else '')

    [staticmethod]
    def info(*arg):
        if len(arg)>1:
            Logger.logger_stdout.info(', '.join([str(a) for a in arg]))
        else:
            Logger.logger_stdout.info(arg[0] if len(arg)>0 else '')

    [staticmethod]
    def warn(*arg):
        if len(arg)>1:
            Logger.logger_stdout.warn(', '.join([str(a) for a in arg]))
        else:
            Logger.logger_stdout.warn(arg[0] if len(arg)>0 else '')

    [staticmethod]
    def error(*arg):
        if len(arg)>1:
            Logger.logger_stdout.error(', '.join([str(a) for a in arg]))
        else:
            Logger.logger_stdout.error(arg[0] if len(arg)>0 else '')
        sys.exit()


    