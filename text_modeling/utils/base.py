#encoding=utf-8
import os
from collections import OrderedDict

def dict_reverse(d):
    new_d={}
    for key in d:
        value=d[key]
        new_d[value]=key
    return new_d

def dict_sub(d,s):
    return {k:d[k] for k in s if k in d}

def leave_values(d, keys):
    if type(d)==type([1,2,3]):
        rval=[]
        for k in d:
            rval+=leave_values(k,keys)
    elif type(d)==type({1:2}):
        rval=[]
        for key in d:
            if key in keys:
                rval.append(d[key])
            else:
                rval+=leave_values(d[key],keys)
    else:
        rval=[]
    return rval

def get_vocab(vocab_path):
    ii=open(vocab_path, 'r')
    vocab,reverse_vocab=OrderedDict(), OrderedDict()
    for line in ii: 
        word=line.strip()
        k=len(vocab)
        reverse_vocab[word]=k
        vocab[k]=word
    return vocab,reverse_vocab

import inspect
def get_func_args(depth=1):
    frame = inspect.currentframe(depth)
    args, name1, name2, values = inspect.getargvalues(frame)
    rval={}
    for d in [args, {}, {}]:
        for arg in d:
            rval[arg]=values.get(arg,None)
    return rval

from collections import OrderedDict, defaultdict
class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        #in python3 you can omit the args to super
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory

def get_file_paths(data_dir):
    data_paths=[]
    for dir_, _, files in os.walk(data_dir):
        for file_name in files:
            path=os.path.join(dir_, file_name)
            data_paths.append(path)
    return data_paths
