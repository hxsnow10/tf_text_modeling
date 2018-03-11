# encoding=utf-8
import numpy as np
import tensorflow as tf
from tf_utils.predict import TFModel
try:
    from nlp.tokenizer.tok import zh_tok
except:
    pass
from tf_utils import load_config
from data_utils import load_data

# See Nohup
input_names=["input_x:0", "input_l:0", "dropout_keep_prob:0"]
output_names=["output/Softmax:0"]

def rsplit(text):
    toks=[x for x in zh_tok(text) if x!='\t']
    new=' '.join(toks)
    return new

class Classifier(object):

    def __init__(self, model_path, use_gpu=False):
        config=load_config(model_path)
        if config.use_char:
            self.pre=lambda x:x.strip()
        else:
            self.pre=rsplit
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 1, 'GPU':0},
              allow_soft_placement=True,
              log_device_placement=False)
        sess=tf.Session(config=session_conf)
        data=load_data(config, mode='inference')
        self.tags=data.tags
        self.words=data.words
        self.seq_p=data.seq_p
        self.tf_model=TFModel(sess,model_path,input_names,output_names)

    def predict(self, line):
        data=[np.array(x) for x in self.seq_p(self.pre(line))]+[1.0]
        v=self.tf_model.predict(data)[0][0][1]
        return v 

if __name__=="__main__":
    model_path="SA/model1text_model=cnn-learning_method=adam_decay-rnn_cell=None/model"
    model_path="SA/model1text_model=cnn_rnn-learning_method=adam_decay-rnn_cell=gru/model"
    tag_model=Classifier(model_path)
    ii=open("bfd.txt")
    result=[]
    import time
    st=time.time()
    oo=open('bfd_result_cnn','w')
    for line in ii:
        try:
            line=line.decode('gbk').encode('utf-8')
            title = line.strip().split('\t')[0]
            print line
            v=tag_model.predict(title)
            oo.write(line.strip()+'\t'+str(v)+'\n')
            continue
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
