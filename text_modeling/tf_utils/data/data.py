#encoding=utf-8
'''
Text2Tensor utils
'''
import numpy as np
import Queue
import random
from functools import wraps
import json

from utils.base import dict_reverse,get_file_paths
from utils import byteify
from utils.wraps import tryfunc
UNK='</s>'
PAD='</pad>'

class Vocab(object):

    def __init__(self, words={}, vocab_path=None, unk=UNK, pad=PAD):
        '''
        words:id2word
        '''
        if words:
            self.vocab=words
            self.reverse_vocab = dict_reverse(words)
        else:
            self.vocab, self.reverse_vocab=self.get_vocab(vocab_path)
        if unk:
            self.unk=unk
            if self.unk not in self.reverse_vocab:
                k=len(self.vocab)
                self.vocab[k]=self.unk
                self.reverse_vocab[self.unk]=k
        if pad:
            self.pad=pad
            if self.pad not in self.reverse_vocab:
                k=len(self.vocab)
                self.vocab[k]=self.pad
                self.reverse_vocab[self.pad]=k
        self.size=1
            
    def get_vocab(self, vocab_path):
        ii=open(vocab_path, 'r')
        vocab,reverse_vocab={},{}
        for line in ii:
            word=line.strip()
            k=len(vocab)
            vocab[word]=k
            reverse_vocab[k]=word
        return vocab, reverse_vocab

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, x):
        return self.reverse_vocab[x]

    def __contains__(self, x):
        return x in self.reverse_vocab

    def get(self, word, default):
        if self.__contains__(word):
            return self.__getitem__(word)
        else:
            return default

    def __call__(self, word):
        if word == None:return self.pad
        return self.get(word, self[self.unk])
        
class seq_process(object):
    def __init__(self, tokenizer, processor, seq_len, return_length,
        with_batch, pad_value):
        '''
        import seq base process'
        use args to define various processors
        assume processor(tok) generate [t0, t1, t2...], 
            which ti is a tensor without batch dim.
        the seq_process output [T0, T1, ..., Lenghth], also without batch dim.
        
        '''
        print tokenizer, type(tokenizer)==str
        if type(tokenizer)==str:
            if tokenizer=='char':
                self.tokenizer=lambda line:[ch.encode('utf-8') for ch in line.decode('utf-8')]
            else:
                self.tokenizer=lambda line:line.split(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.processor = processor
        self.seq_len = seq_len
        self.return_length = return_length
        self.size=processor.size
        if return_length:self.size+=1
        self.with_batch=with_batch
        self.pad_value=pad_value

    def __call__(self, line):
        toks=self.tokenizer(line)
        rval=[]
        for tok in toks[:self.seq_len]:
            a=self.processor(tok)
            if a==None:continue
            if type(a)==int:a=[a]
            rval.append(a)
        if not rval:
            return None
        pad=[np.ones_like(rval[0][i])*self.pad_value for i in range(self.processor.size)]
        length=len(rval)
        rval=rval+[pad,]*(self.seq_len-len(rval))
        final=[[x[i] for x in rval] for i in range(self.processor.size)]
        if self.return_length:
            final.append(length)
        if self.with_batch:
            final=[[x] for x in final]
        return final

class label_line_processing():

    def __init__(self, words, split=' '):
        self.vocab=Vocab(words=words, unk=None, pad=None)
        self.size=1

    def __call__(self, line):
        words = line.strip().split()
        rval=[0,]*len(self.vocab)
        for word in words:
            if word in self.vocab:
                rval[self.vocab[word]]=1
        return [[rval]]

class sub_merged_vocab():

    def __init__(self, vocab, sub_vocab):
        self.vocab=vocab
        self.sub_vocab=sub_vocab
        self.sub_processor=seq_process("char", self.sub_vocab, char_len, use_char_length, False, self.sub_vocab[PAD])
        self.size=3

    def __call__(self, tok):
        s=self.sub_processor(tok)
        if not s:return None
        else:return [self.vocab(tok)]+s
        
class sequence_line_processing(seq_process):
    
    def __init__(self, words, seq_len=100, split=' ', return_length=True, 
        sub_words=None, char_len=10, use_char_length=True,
        sents_split=None, sents_num=30):
        self.tokenizer=split
        self.vocab= Vocab(words=words)
        if not sub_words:
            self.processor=self.vocab
        else:
            self.sub_vocab=Vocab(sub_words)
            self.processor=sub_merged_vocab(self.vocab, self.sub_vocab)
        if sents_split:
            self.sents_split=sents_split
            self.sents_processor=seq_process(self.tokenizer, self.processor, seq_len, return_length, False, self.vocab[PAD])
            super(sequence_line_processing, self).__init__(self.sents_split, self.sents_processor, sents_num, return_length, True, self.vocab[PAD])
        else:
            super(sequence_line_processing, self).__init__(self.tokenizer, self.processor, seq_len, return_length, True, self.vocab[PAD])

class split_line_processing():

    def __init__(self, line_processors=[], split='\t'):
        self.line_processors = line_processors
        self.size = sum([p.size for p in line_processors])
        self.split='\t'

    # @tryfunc()
    def __call__(self, line):
        parts = line.strip().split(self.split)
        if len(parts)!=len(self.line_processors):
            #print len(parts)
            pass
        n=len(self.line_processors)
        if len(parts)>n:
            parts[n-1]=' '.join(parts[n-1:])
            parts=parts[:n]
        rval = [self.line_processors[k](part) for k,part in enumerate(parts)]
        # print rval
        rval=sum(rval,[])
        return rval

class json_line_processing():

    def __init__(self, d_processors):
        self.d_processors = d_processors
        self.size=sum(x.size for x in d_processors.values())

    def __call__(self, line):
        rval=[]
        a=byteify(json.loads(line.strip()))
        for key in self.d_processors:
            process = self.d_processors[key]
            part=a[key]
            rval+=process(part)
        return rval

class text_dependency_processing():
    def __init__(self, split=' ', dpfunc=None, words={}, seq_len=20):
        self.vocab=Vocab(words)
        self.split=split
        self.dpfunc=dpfunc

    def __call__(self, line):
        words = line.split(self.split)
        mat=np.zeros([seq_len, seq_len], np.float32)
        for k,head,rel in self.dpfunc(words):
            mat[k][head]=1
        # TODO mat_sparse=[]
        return [[mat]]
 
class TfDataset():

    def __init__(self, py_iterator=None, py_packed=True, sess=None, shuffle_size=1000, batch_size=100):
        if py_packed:
            def sample_iterator():
                for samples in py_iterator:
                    for sample in samples:
                        yield sample
            self.py_iterator=sample_iterator()
        else:
            self.py_iterator=py_iterator
        self.tf_ds=tf.Dataset.from_generator(py_iterator)
        if shuffle_size:
            self.tf_ds=self.shuffle(shuffle_size)
        self.tf_ds=self.tf_ds.batch(batch_size)
        self.sess=sess

    def __iter__(self):
        value = self.tf_ds.make_one_shot_iterator().get_next()
        if not self.sess:self.sess=tf.get_default_session()
        while True:
            try:
                return self.sess.run(value)
            except Exception,e:
                print e
            except tf.errors.OutOfRangeError:
                print "epoch of data finish"
                break

class LineBasedDataset():

    def __init__(self, data_paths=[], data_dir="", line_processing=None,
        split="\n",
        queue_size=100000, save_all=False, batch_size=100, sampling=None):
        
        if not data_paths and data_dir:
            data_paths=get_file_paths(data_dir)
        elif type(data_paths)==str:
            data_paths=[data_paths]
        self.data_paths = data_paths
        
        self.queue_size=queue_size
        self.line_processing=line_processing
        self.size=line_processing.size
        self.save_all=save_all
        self.batch_size = batch_size
        self.l=None
        if save_all:
            self.all_data=list(self.epoch_data())
            self.l=len(self.all_data)
            print 'len===============',self.len
        self.sampling=sampling
        self.split=split
    
    def sample(self):
        return random.random()<self.sampling
    
    def iter(self, epoches):
        for i in range(epoches):
            for batch in self.__iter__():
                yield batch

    def __iter__(self):
        if self.save_all:
            for d in self.all_data:
                yield d
            return
        self.queue = Queue.Queue()
        file_readers = [open(data_path, 'r') for data_path in self.data_paths]
        while True:
            if False:
            # if not self.queue.empty():
                #yield self.queue.get()
                pass
            else:
                batch=[[],]*self.size
                k=0
                for file_reader in file_readers:
                    raw_lines=[]
                    for line in file_reader:
                        if self.sampling and not self.sample():continue
                        k+=1
                        if self.split=="\n":
                            items=self.line_processing(line)
                        elif self.split=="\n\n":
                            if not line.strip():
                                
                                items=self.line_processing(raw_lines)
                                raw_lines=[]
                            else:
                                raw_lines.append(line.strip())
                                continue 
                        if not items or len(items)!=self.size:
                            print items
                            print "get size={}, but except size={}".format(len(items), self.size)
                            print 'Not correct parsed:\t'
                            # print line,len(line)
                            # raw_input('xxxxxxxxxxxxx')
                            continue
                        for k,item in enumerate(items):
                            batch[k]=batch[k]+item
                        while len(batch[0])>=self.batch_size:
                            one_batch=[np.array(x[:self.batch_size]) for x in batch]
                            #self.queue.put( batch )
                            yield one_batch
                            batch=[x[self.batch_size:] for x in batch]
                        #if self.queue.qsize()==self.queue_size:
                        #    break
                if k==0:break

class StackDataset():
    def __init__(self, datasets, axis=0):
        # assume datasets has same size, and same shape except axis
        for i in range(len(datasets)):
            assert datasets[i].size==datasets[0].size
        self.size=datasets[0].size
        self.datasets=datasets
        self.axis=axis

    def iter(self, epoches, use_all=False):
        if use_all:
            for batchs in zip(*[d.iter(epoches) for d in self.datasets]):
                batch=np.concatenate(batchs,axis=axis)
                yield batch
        else:
            for i in range(epoches):
                for batch in self.__iter__():
                    yield batch
            

    def __iter__(self):
        for batchs in zip(*self.datasets):
            this=[None,]*self.size
            for i in range(self.size):
                this[i]=np.concatenate([batch[i] for batch in batchs])
            yield this

# ----------------------------------

class MultiDataset():

    def __init__(self, datasets):
        self.datasets=datasets
        self.size=sum(d.size for d in datasets)

    def epoch_data(self):
        self.iterators=[x.epoch_data() for x in self.datasets]
        while True:
            batch_item = []
            for it in self.iterators:
                item=it.next()
                if type(item)==type([1,2]):
                    batch_item+=item
                else:
                    batch_item.append(item)
            if not batch_item:
                yield bctch_item
            else:
                break

    def __len__(self):
        return len(self.datasets[0])

class SampleDataset():

    def __init__(self, datasets, ratios):
        self.datasets=datasets
        self.ratios=[sum(ratios[:i+1])*1.0/sum(ratios) for i in range(len(ratios))]
        self.size=sum(d.size for d in datasets)

    def select(self,r):
        for i in range(len(self.ratios)):
            if self.ratios[i]>=r:
                return i
        
    def epoch_data(self):
        iterators=[x.epoch_data() for x in self.datasets]
        finished=[0,]*len(self.ratios)
        finished[1]=1
        # TODO 这里手动让第二数据集直接结束
        while True:
            r=random.random()
            k=self.select(r)
            batch_item=[None,]*self.size
            item=None
            try:
                item=iterators[k].next()
                mask=[0,]*len(self.datasets)
                mask[k]=1
                if not isinstance(item, list): item=[item]
                size=sum(d.size for d in self.datasets[:k])
                batch_item[size:size+len(item)]=item
                batch_item[size+len(item):size+2*len(item)]=item
                # 上面一行指导了无效数据
                batch_item.append(np.array(mask))
                yield batch_item
            except Exception,e:
                print e
                finished[k]=1
                if sum(finished)==len(finished):
                    break
    
    def __len__(self):
        self.len=sum(len(d) for d in  self.datasets)
        return self.len


class GraphCnnProcessing():

    def __init__(self):
        pass

    def __call__(self, inputs):
        #for input in inputs:
        #    generate conv2, conv3, ..
        pass


