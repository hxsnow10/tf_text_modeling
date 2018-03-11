# encoding=utf-8
import os,sys
sys.path.append('..')
from tf_utils.data import LineBasedDataset
from tf_utils.data import split_line_processing, sequence_line_processing, label_line_processing
from itertools import islice

'''
Data Loader for tags\tsequence like data
Now following config parameters may influence data_load:
    tags_path
    train_paths
    dev_paths

    words_vec
    chars_vec

    use_seq_lenghth
    seq_len
    split
    char_len

    batch_size
    use_label_weights
    }

'''
def load_data_clf(config, mode="train"):
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
        seq_p=sequence_line_processing(words, return_length=config.use_seq_length, seq_len=config.seq_len, split=config.split, sub_words=sub_words, char_len=config.char_len)
        words=seq_p.vocab.vocab
        import json
        print "TAGS=\t",json.dumps(tags, ensure_ascii=False)
        print "WORDS_NUM=\t",len(words)

        # sub_words=seq_p.sub_vocab.vocab
        # print "SUB_WORDS_NUM=\t",sub_words and len(sub_words)
        if mode=='train': 
            line_p = split_line_processing([label_p,seq_p])
            train_data=LineBasedDataset(config.train_paths, line_processing=line_p, batch_size=config.batch_size)
            dev_data=LineBasedDataset(config.dev_paths, line_processing=line_p, batch_size=config.batch_size)

            train_label_weights=[0,]*len(tags)
            if config.use_label_weights:
                for inputs in train_data:
                    y=inputs[0]
                    for k in y.tolist():
                        for kk in k:
                            train_label_weights[kk]+=1
                s=sum(train_label_weights)
                class_weights=[x*1.0/s for x in train_label_weights]
    return Data 

