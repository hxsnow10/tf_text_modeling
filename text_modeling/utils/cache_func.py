# encoding=utf-8
import json
import time

def Cache(object):

    def __init__(self, expire_time=15):
        self.c={}
        self.last_minutes=int(time.strftime("%M",time.localtime(time.time())))
    
    def clear(self):
        self.c={}
    
    def add(self, key, value):
        key=json.dumps(key, ensure_ascii=False)
        self.c[key]=value

    def search(self,key):
        key=json.dumps(key, ensure_ascii=False)
        rval=self.c.get(key,None)
        minutes=int(time.strftime("%M",time.localtime(time.time())))
        if minutes-self.last_minutes>=10:
            self.clear()
            selfminutes=minutes
        return rval

class CacheFunc(object):
    def __init__(self, func):
        self.cache=Cache()
        self.func=func
    
    def __call__(self, *args, **kwargs):
        key=(args,kwargs)
        rval=self.search(key)
        if not rval:
            rval=self.func(*args, **kwargs)
            self.cache.add(key,value)
        return rval
