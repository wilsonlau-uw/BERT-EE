import configparser
import datetime
import time
import os
from pathlib import Path


class Config:
    parser =None
    result_folder=None
    args=None

    [staticmethod]
    def init(args):
        Config.args=vars(args)
        Config.parser = configparser.ConfigParser(allow_no_value=True)
        Config.parser.read('./config.ini')
        if Config.getstr("general", "results_folder") != '':
            result_folder =  Config.getstr("general", "results_folder")
        else:
            result_folder = Config.getstr('general','tasks').replace(' ','').replace(',','_')+'_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        Config.result_folder=os.path.join(Config.getstr("general", "results_path"),result_folder )

    [staticmethod]
    def get_args(section, opt):
        return Config.args['{}_{}'.format(section,opt)]

    [staticmethod]
    def get_resultfolder(subfolder=None ):
        folder =Config.result_folder
        if subfolder is not None:
            folder=os.path.join(Config.result_folder,subfolder)
        Path(folder).mkdir(parents=True, exist_ok=True)
        return folder

    [staticmethod]
    def getboolean(section, opt, usefallback=False, fallback=None):
        args_opt= Config.get_args(section, opt)
        if usefallback and Config.getstr(section, opt) == '':
            return fallback
        elif args_opt !=None:
            return args_opt
        else:
            return Config.parser.getboolean(section, opt)

    [staticmethod]
    def getstr(section, opt):
        args_opt = Config.get_args(section, opt)
        if args_opt != None:
            return args_opt
        else:
            return Config.parser.get(section, opt)

    [staticmethod]
    def getfloat(section, opt, usefallback=False, fallback=None):
        args_opt = Config.get_args(section, opt)

        if usefallback and Config.getstr(section, opt) == '':
            return fallback
        elif args_opt !=None:
            return args_opt
        else:
            return Config.parser.getfloat(section, opt)

    [staticmethod]
    def getint(section, opt, usefallback=False, fallback=None):
        args_opt = Config.get_args(section, opt)
        if usefallback and Config.getstr(section, opt) == '':
            return fallback
        elif args_opt !=None:
            return args_opt
        else:
            return Config.parser.getint(section, opt)

    [staticmethod]
    def getlist(section, opt, usefallback=False, fallback=None):
        args_opt = Config.get_args(section, opt)
        if usefallback and Config.getstr(section, opt) == '':
            return fallback
        elif args_opt !=None:
            return args_opt
        else:
            return Config.getstr(section, opt).split(',')

