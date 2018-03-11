# encoding=utf-8
import multiprocessing as mp
from multiprocessing import Pool
import sys
import itertools
import time
from time import sleep

class batched_process():
    def __init__(self, func):
        self.func=func
        print self.func

    def __call__(self,data):
        rval = [ self.func(sample) for sample in data ]
        return rval

    def finish(self):
        self.func.finish()

def batched_iterator(data,N=100,q=None):
    batch=[]
    pre_num, num=1, 0
    for item in data:
        num+=1
        if num==10*pre_num:
            print num
            pre_num=num
        if len(batch)==N:
            if q: yield [batch, q]
            else:yield batch
            batch=[]
        batch.append(item)
    if q: yield [batch, q]
    else:yield batch

def BatchedMpWork(data, process, listener, workers=20):
    '''Batched Multiprocess Worker.
    data is origianl sample generator
    process is func about how to map sample to output_sample
    listener is func about how to write sample_output to output_file
    批量读入，批量处理。
    '''
    data = batched_iterator(data)
    #global process_
    #global listener_
    process_ = batched_process(process)
    listener_ = batched_process(listener)
    MpWork(data, process_, listener_, workers)

#@profile
def MpWork(data, process, listener, workers=20, batch_num=10000):
    print 'data={}\nprocess={}\nwriter={}'.format(data, process, listener)
    manager = mp.Manager() 
    q = manager.Queue()
    global listener_thread
    def listener_thread():
        while True:
            m=q.get()
            if m == 'kill':
                listener.finish()
                break
            else:
                listener(m)
        return 'finished'

    global process_thread
    def process_thread(inp):
        #print inp
        rval=process(inp)
        q.put(rval)
        return 'ok'
    print 'start listener'
    print 'all workers=', workers
    pool = mp.Pool(workers+1)
    print 'start processer'
    global process_
    watcher = pool.apply_async(listener_thread)
    solved = 0
    while True:# not read all data once time
        data_ = list(itertools.islice(data,batch_num))
        l = len(data_)
        g=pool.map(process_thread, data_)
        print 'finished one batch'
        if l==0:break
        solved += l
        print solved
        #import time
        #time.sleep(10)
    q.put('kill')
    print watcher.get()

class reader():

    def __init__(self, file_path):
        self.file=open(file_path, 'r')

    def __call__(self):
        return self.file.readline()

    def __iter__(self):
        for line in self.file:
            yield line

class writer():

    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.c=0
    
    def __call__(self, line):
        if line.strip():
            #print self.file
            self.c+=1
            self.file.write(line)
            if self.c%1000==0:
                self.file.flush()
                self.c=0
    def finish(self):
        print 'FINISH'
        self.file.flush()


