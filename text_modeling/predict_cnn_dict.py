# encoding=utf-8
import numpy as np
import tensorflow as tf
from tf_utils.data import sequence_line_processing
from tf_utils.predict import TFModel
try:
    from nlp.tokenizer.tok import zh_tok
except:
    pass

use_char=True

tags_path="data/tags.txt"
if use_char:
    words_path="data/char_vec.txt"
else:
    words_path="data/vec.txt"
if use_title:
    input_names=["input_x_title:0", "input_x:0", "dropout_keep_prob:0"]
else:
    input_names=["input_x:0", "dropout_keep_prob:0"]

output_names=["output/predictions:0"]
model_path="model/model-732"
def rsplit(text):
    toks=[x for x in zh_tok(text) if x!='\t']
    new=' '.join(toks)
    return new
if use_char:
    split=lambda x:x.strip()
else:
    split=rsplit
class Classifier(object):

    def __init__(self, ):
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 1, 'GPU':0},
              allow_soft_placement=True,
              log_device_placement=False)
        sess=tf.Session(config=session_conf)
        self.tags = tags={k:word.strip() for k,word in enumerate(open(tags_path))}
        words={k-1:word.split()[0] for k,word in enumerate(open(words_path)) if k>=1 and k<200000}
        self.seq_p=sequence_line_processing(words, return_length=False, max_len=200, split="char")
        self.tf_model=TFModel(sess,model_path,input_names,output_names)

    def predict(self, title, content):
        # data=[np.array(self.seq_p(split(title))[0]),np.array(self.seq_p(split(content))[0]),1.0]
        data=[np.array(self.seq_p(split(title+content))[0]),1.0]
        k = int(self.tf_model.predict(data)[0])
        tag=self.tags[k]
        if tag=="其他":tag=''
        return tag

tag_model=Classifier()
if __name__=="__main__":
    title="出租房子啦"
    content="2000元一平米，西北旺，超便宜"
    print tag_model.predict(title, content)
    ii=open("dev.txt")
    result=[]
    import time
    st=time.time()
    for line in ii:
        try:
            tag, title, content = line.strip().split('\t')
            '''
            print tag
            print 'result============='
            '''
            p_tag=tag_model.predict(title, content)
            if tag==p_tag:
                result.append(1)
            else:
                print title
                print content 
                print tag
                print p_tag
                result.append(0)
        except Exception,e:
            print e
            pass
        if result and len(result)%100==0:
            print sum(result)*1.0/len(result)
            # raw_input('XXXXXXXXXXXX')
    et=time.time()
    print len(result)/(et-st)
