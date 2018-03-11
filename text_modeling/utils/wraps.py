# encoding=utf-8
import time
from functools import wraps
import traceback
import os

def count_time(func_name='func'):
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            start=time.time()
            rval=func(*args, **kwargs)
            print "{}.finished,in {},  use time {}".format(func_name, os.getpid(), time.time()-start)
            return rval
        return wrap
    return decorator

def ifrepeat(n,m=5):
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            for i in range(n):
                print "repeat {}".format(i)
                break_flag=func(*args, **kwargs)
                if m<i and not break_flag:break
            return None
        return wrap
    return decorator

def tryfunc(except_return=None):
    def decorator(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            try:
                rval = func(*args, **kwargs)
            except:
                traceback.print_exc()
                rval = except_return
            return rval
        return wrap
    return decorator

    
