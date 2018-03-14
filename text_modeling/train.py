# encoding=utf-8
import sys
import os
import json
import argparse

import tensorflow as tf
import numpy as np

from utils.word2vec import load_w2v
from utils.base import get_vocab
from utils.base import get_func_args

from tf_utils.model import multi_filter_sizes_cnn 
from tf_utils import load_config
from tf_utils import check_dir
from tf_utils.profile import *
from shutil import copy
from data_utils import load_data

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from model import TextModel
from evaluate import evaluate_1, evaluate_2, evaluate_3
config=None

'''
assumpation:
* load_d train_data, dev_data
* model has init, loss, train_op, step_summaries, saver
'''

def train(sess, model, train_data, dev_data=None, summary_writers={}, tags=None):
    if config.objects=="tag":
        if config.tag_exclusive:
            evaluate=evaluate_1
        else:
            evaluate=evaluate_2
    else:
        if config.tag_exclusive:
            evaluate=evaluate_3
    
    profiler = model_analyzer.Profiler(graph=sess.graph)
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    show_params(profiler)

    sess.run(model.init)
    step=0
    best_score=0
    pt=5
    for epoch in range(config.epoch_num):
        for k,inputs in enumerate(train_data):
            print "get in ", step
            fd=dict(zip(model.inputs, inputs+[config.dropout_ratio]))
            if step%config.summary_steps!=0:
                # print sess.run(model.debugs, feed_dict=fd)
                loss,_=sess.run([model.loss, model.train_op], feed_dict=fd)
                '''
                # for explanation-cnn
                index=sess.run(model.pooled_index, feed_dict=fd)
                scores=sess.run(model.scores2, feed_dict=fd)
                locs=parse_text_cnn_index(config.filter_sizes, config.filter_nums, index)
                debug_info=[zip(locs[i],scores[i]) for i in range(config.batch_size)]
                print debug_info
                '''
            else:
                loss,_,summary=sess.run(\
                        [model.loss, model.train_op, model.step_summaries], feed_dict=fd,\
                        options=run_options,
                        run_metadata=run_metadata)
                summary_writers['train'].add_summary(summary, step)
                summary_writers['train'].add_run_metadata(run_metadata, "train"+str(step))
                profiler.add_step(step=step, run_meta=run_metadata)
                #统计内容为每个graph node的运行时间和占用内存
                if step/config.summary_steps==1:
                    save_json(profiler, step, '/tmp/profiler_{}.json'.format(step))
                    rank_ops_by_time(profiler)
                    rank_ops_by_memory(profiler)
                    profiler.advise(options=model_analyzer.ALL_ADVICE)
                print step, "write summary" 

            print "epoch={}\tstep={}\tglobal_step={}\tloss={}".format(epoch, k, step ,loss)
            step+=1
            # eval every batch
        if dev_data:
            _,train_data_metrics = evaluate(sess,model,train_data,tags)#_,{"f":0.8, "r":0.9}
            score,dev_data_metrics = evaluate(sess,model,dev_data,tags)
            def add_summary(writer, metric, step):
                for name,value in metric.iteritems():
                    summary = tf.Summary(value=[                         
                        tf.Summary.Value(tag=name, simple_value=value),   
                        ])
                    writer.add_summary(summary, global_step=step)
            add_summary(summary_writers['train'], train_data_metrics, step)
            add_summary(summary_writers['dev'], dev_data_metrics, step)
            # add_summary(summary_writers['test-2'], test_data_metricss[1], step) 
            if score>best_score:
                best_score=score
                model.train_saver.save(sess, config.model_path, global_step=step)
                model.train_saver.save(sess, config.model_path)
                pt=5
            else:
                pt-=1
            if pt<0:
                break
        else:
            model.train_saver.save(sess, config.model_path, global_step=step)
            
def main():
    check_dir(config.summary_dir, config.ask_for_del, config.restore)
    check_dir(config.model_dir, config.ask_for_del, config.restore)
    copy("config.py", config.summary_dir) 
    copy("config.py", config.model_dir) 
    
    data = load_data(config)
    def load_vec(vec):
        if not vec:return None
        if vec.vec_path:
            if not os.path.exists(vec.vec_path):
                raise Exception("File {} not exists".format(vec.vec_path))
            w2v = load_w2v(vec.vec_path, max_vocab_size=vec.max_vocab_size, norm=vec.norm)
            init_emb = np.array(w2v.values()+[[0,]*vec.vec_size,]*(len(data.words)-len(w2v)), dtype=np.float32)
        else:
            init_emb=None
            # init_emb=[len(data.words),len(data.sub_words)]
        return init_emb
    word_emb=load_vec(config.words_vec)
    char_emb=load_vec(config.chars_vec)
    with tf.Session(config=config.session_conf) as sess:
        # use tf.name_scope to manager variable_names
        model=TextModel(
            configp=config,
            vocab_size=len(data.words),
            num_classes=len(data.tags), 
            init_emb=word_emb,
            sub_init_emb=char_emb,
            reuse=False,# to use when several model share parameters
            debug=False,
            
            # debug model only work for cnn, tell how much score every ngram contribute to every label
            class_weights=data.class_weights,
            mode='train')
        if config.restore:
            try:
                model.train_saver.restore(sess, config.model_path)
                print "reload model"
            except Exception,e:
                print e
                print "reload model fail"
        summary_writers = {
            sub_path:tf.summary.FileWriter(os.path.join(config.summary_dir,sub_path), sess.graph, flush_secs=5)
                for sub_path in ['train','dev']}
        train(sess, model,
                data.train_data, data.dev_data,
                tags=None,
                summary_writers=summary_writers)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default=".")
    parser.add_argument("-r", "--restore", type=int, default=1)
    args = parser.parse_args()
    global config
    config=load_config(args.config_path)
    config.restore=args.restore
    print config.restore
    main()
