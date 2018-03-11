# encoding=utf-8
import numpy as np
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score

def evaluate_1(sess, model, eval_data, target_names=None, restore=False, dropout=True):
    total_y,total_predict_y = [], []
    print 'start evaluate...'
    for inputs in eval_data:
        if dropout:
            inputs=inputs+[1]
        fd=dict(zip(model.inputs, inputs))
        predict_y=\
            sess.run(model.predictions, feed_dict = fd)
        total_y = total_y + [np.argmax(inputs[0],-1)]
        total_predict_y = total_predict_y + [predict_y]
    total_y = np.concatenate(total_y,0)
    total_predict_y=np.concatenate(total_predict_y,0)
    # print total_y.shape, total_predict_y.shape
    print classification_report(total_y, total_predict_y)
    p,r,f=precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print p,r,f
    return f,{"precision":p,"recall":r,"f1":f}

def evaluate_2(sess, model, eval_data, target_names=None, restore=False, dropout=True):
    total_y,total_predict_y = [], []
    print 'start evaluate...'
    for inputs in eval_data:
        if dropout:
            inputs=inputs+[1.0]
        fd=dict(zip(model.inputs, inputs))
        predict_y=\
            sess.run(model.predictions, feed_dict = fd)
        total_y = total_y + [inputs[0]]
        total_predict_y = total_predict_y + [predict_y]
    total_y = np.concatenate(total_y,0)
    total_predict_y=np.concatenate(total_predict_y,0)
    # print total_y.shape, total_predict_y.shape
    # print classification_report(total_y, total_predict_y, target_names=target_names)
    p,r,f=precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print f,p,r
    return f,{"precision":p,"recall":r,"f1":f}

def evaluate_3(sess, model, eval_data, target_names=None, restore=False, dropout=True):
    # 获得Y_hat labels=[batch_size, seq_len], length=[batch_size]
    # 获得Y y_true
    # 摊平连接，evaluate即与一般的分类等价
    total_y,total_predict_y = [], []
    def process(labels, lengths):
        tags=[]
        for seq_tag, length in zip(labels.tolist(), lengths.tolist()):
            tags+=seq_tag[:length]
        return tags
    print 'start evaluate...'
    for inputs in eval_data:
        if dropout:
            inputs=inputs+[1.0]
        fd=dict(zip(model.inputs, inputs))
        predict_y, lengths=\
            sess.run([model.predictions, model.input_sequence_length], feed_dict = fd)

        total_y = total_y + process(inputs[0], lengths)
        total_predict_y = total_predict_y + process(predict_y, lengths)
    total_y = np.array(total_y)
    total_predict_y = np.array(total_predict_y)
    p,r,f=precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print f,p,r
    return f,{"precision":p,"recall":r,"f1":f}

