import tensorflow as tf
import os
from os import makedirs
from shutil import rmtree
from load_config import load_config

        
cpu_conf = tf.ConfigProto(
      device_count = {'CPU': 12, 'GPU':0}, 
      allow_soft_placement=True,
      log_device_placement=False,)
def cpu_sess():
    return tf.Session(config=cpu_conf)

def check_dir(dir_path, ask_for_del=False, restore=True):
    if os.path.exists(dir_path):
        if restore:return
        y=''
        if ask_for_del:
            y=raw_input('new empty {}? y/n:'.format(dir_path))
        if y.strip()=='y' or not ask_for_del:
            rmtree(dir_path)
        else:
            print('use a clean summary_dir')
            quit()
    makedirs(dir_path)
    '''
    oo=open(os.path.join(dir_path,'config.txt'),'w')
    d={}
    for name in dir(config):
        if '__' in name:continue
        d[name]=getattr(config,name)
    try:
        oo.write(json.dumps(d,ensure_ascii=False))
    except:
        pass
    '''
