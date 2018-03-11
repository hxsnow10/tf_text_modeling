# encoding=utf-8
import numpy as np
import sys
sys.path.append("..")
from tf_utils.data import Vocab
from tf_utils.data import LineBasedDataset


class word2vec_line_processing():

    def __init__(self, d_id2words, window, skip=True):
        self.vocab = Vocab(words=d_id2words)
        self.unk_id = self.vocab[self.vocab.unk]
        self.window = window
        self.skip=skip
        self.size=2
    
    def __call__(self, line):
        words=line.strip().split()
        ids=[ self.vocab.get(word, self.unk_id) for word in words]
        if self.skip:
            ids = [id_ for id_ in ids if id_!=self.unk_id]
        examples, labels=[], []
        for c in range(len(ids)):
            for k in range(max(-self.window, -c), min(self.window+1, len(ids)-c)):
                examples.append(ids[c])
                labels.append(ids[c+k])
        # examples = np.array(examples, dtype=np.int64)
        # labels = np.array(labels, dtype=np.int64)
        rval=[examples, labels]
        return rval

def load_data_w2v(config):
    window=5
    d_id2words={k:line.strip().split()[0] for k,line in enumerate(open(config.dict_path, 'r'))}
    p = word2vec_line_processing(d_id2words, config.window)
    dataset = LineBasedDataset(config.train_paths,line_processing=p)
    return dataset
    
if __name__=="__main__": 
    from tf_utils import load_config
    config=load_config('../example_configs/config4.py')
    k=0
    dataset=load_data_w2v(config)
    for k,s in enumerate(dataset):
        print k
        for ele in s:
            print '\t',ele.shape
        k+=1
        if k>=4:break

