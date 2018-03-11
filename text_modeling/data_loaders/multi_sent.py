# encoding=utf-8
'''
generate multi sent data for model like hs_rnn_attn
'''
import os,sys
sys.path.append('..')
from tf_utils.data import LineBasedDataset
from tf_utils.data import split_line_processing, sequence_line_processing, label_line_processing
from itertools import islice

def load_data(config, mode="train"):
    class Data():
        tags={k:word.strip() for k,word in enumerate(open(config.tags_path))}
        def load_words(path, skip_head, max_vocab_size):
            if not path:return {}
            print path
            ff=open(path)
            if skip_head:
                ff.readline()
            words={k:word.split()[0] for k,word in enumerate(islice(ff,max_vocab_size)) if k<max_vocab_size or max_vocab_size is None}
            return words
        print config.words_vec.vocab_path
        words=config.words_vec and load_words(config.words_vec.vocab_path, config.words_vec.vocab_skip_head, config.words_vec.max_vocab_size) 
        sub_words= config.chars_vec and load_words(config.chars_vec.vocab_path, config.chars_vec.vocab_skip_head, config.chars_vec.max_vocab_size)
        label_p=label_line_processing(tags)
        seq_p=sequence_line_processing(words, return_length=config.use_seq_length, seq_len=config.seq_len, split=config.split, sub_words=sub_words, char_len=2)
        words=seq_p.vocab.vocab
        import json
        print "TAGS=\t",json.dumps(tags, ensure_ascii=False)
        print "WORDS_NUM=\t",len(words)
        print "SUB_WORDS_NUM=\t".format(sub_words and len(sub_words))
        if mode=='train': 
            line_p = split_line_processing([label_p,seq_p])
            train_data=LineBasedDataset(config.train_paths, line_processing=line_p, batch_size=config.batch_size)
            dev_data=LineBasedDataset(config.dev_paths, line_processing=line_p, batch_size=config.batch_size)

            train_label_weights=[0,]*len(tags)
            if config.use_label_weights:
                for inputs in train_data:
                    y=inputs[0]
                    for k in y.tolist():
                        y[k]+=1
                s=su(train_label_weights)
                train_label_weights=[x/s for x in train_label_weights]
    return Data 

if __name__=="__main__":
    from tf_utils import load_config
    config=load_config('.')
    data=load_data(config)
    for dd in [data.train_data, data.dev_data]:
        print dd
        for k,inputs in enumerate(dd):
            print '-'*20,'batch ',k,'-'*20
            for inp in inputs:
                print inp.shape
            if k>=3:break
