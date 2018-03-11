# encoding=utf-8
import sys
sys.path.append('..')
from tf_utils.data import LineBasedDataset, Vocab, sub_merged_vocab
from itertools import islice

class conll_lines_processing():
    def __init__(self, split, words, tagss, seq_len, sub_words=None):
        self.split=split
        self.vocabs=[Vocab(words)]+[Vocab(tags) for tags in tagss]
        self.seq_len=seq_len
        self.size=len(self.vocabs)+1

    def __call__(self, lines):
        rval=[[] for _ in range(self.size-1)]
        for k,line in enumerate(lines):
            if k>=self.seq_len:continue
            toks=line.rstrip().split(self.split)
            if len(toks)!=self.size-1:continue
            ids=[self.vocabs[i](toks[i]) for i in range(len(toks))]
            for i in range(self.size-1):
                rval[i].append(ids[i])
        length=len(rval[0])
        for i in range(self.size-1):
            rval[i]=rval[i]+[self.vocabs[i](self.vocabs[i].pad),]*(self.seq_len-len(rval[i]))
            rval[i]=[rval[i]]
        rval.append([length])
        rval=[rval[1], rval[0], rval[-1]]
        return rval

def load_data_ner(config, mode="train"):
    '''every line like: tok[\ttag1..]\n or blackline
    specifically, LineBasedDataset, use '\n\n' as split
    '''
    class Data():
        def load_words(path, skip_head, max_vocab_size):
            if not path:return {}
            print path
            ff=open(path)
            if skip_head:
                ff.readline()
            words={k:word.split()[0] for k,word in enumerate(islice(ff,max_vocab_size)) if k<max_vocab_size or max_vocab_size is None}
            return words
        words=config.words_vec and load_words(config.words_vec.vocab_path, config.words_vec.vocab_skip_head, config.words_vec.max_vocab_size) 
        sub_words= config.chars_vec and load_words(config.chars_vec.vocab_path, config.chars_vec.vocab_skip_head, config.chars_vec.max_vocab_size)
        # if config.use_sub:pass
        tagss=[ load_words(tags_path, None, None) for tags_path in config.tags_paths]
        lines_processing=conll_lines_processing(' ', words, tagss, config.seq_len, sub_words=sub_words)
        tags=lines_processing.vocabs[1]
        train_data=LineBasedDataset(config.train_paths, line_processing=lines_processing, batch_size=config.batch_size, split='\n\n')
        dev_data=LineBasedDataset(config.dev_paths, line_processing=lines_processing, batch_size=config.batch_size, split='\n\n')
        class_weights=None
    return Data 
