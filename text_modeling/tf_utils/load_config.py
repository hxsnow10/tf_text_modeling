# encoding=utf-8
import importlib
import os,sys

def load_config(path='.'):
    path=os.path.abspath(path)
    if os.path.isdir(path):
        dir_path=path
        config_name="config"
    else:
        dir_path=os.path.dirname(path)
        config_name=os.path.basename(path).split('.')[0]

    sys.path.insert(0, dir_path)
    mo =importlib.import_module(config_name) 
    print "load config from {}/{}".format(dir_path, config_name)
    return mo.config
