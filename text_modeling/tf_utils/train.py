# encoding=utf-8
import os,sys
from os import makedirs
from shutil import rmtree
import tensorflow as tf

def check_dir(dir_path, ask_for_del):
    if os.path.exists(dir_path):
        y=''
        if ask_for_del:
            y=raw_input('new empty {}? y/n:'.format(dir_path))
        if y.strip()=='y' or not ask_for_del:
            rmtree(dir_path)
        else:
            print('use a clean summary_dir')
            quit()
    makedirs(dir_path)

def standard_train(
        sess, model,
        train_data, dev_data, test_datas,
        train_op,
        summary_dir,model_dir,model_name,
        step_summary_op, epoch_summary_op, summary_steps=5, summary_epoch=1,
        init=True,
        epoch_num=20,
        eval_func=None, score_key='f1', test_func=None):# evaluate_func(sess, model, dev_data) return scalar
    check_dir(summary_dir,True)
    check_dir(model_dir,True)
    if init:
        sess.run(model.init)
    # build summary, saver
    step=0
    best_score=0
    summary_writers = {
        sub_path:tf.summary.FileWriter(os.path.join(summary_dir,sub_path), flush_secs=5)
            for sub_path in ['train','dev']+["test_{}".format(i) for i in range(len(test_datas))]}

    for epoch in range(epoch_num):
        for k,inputs in enumerate(train_data):
            fd=dict(zip(model.inputs, inputs))
            if step % summary_steps!=0:
                _=sess.run(train_op, feed_dict=fd)
            else:
                _,summary=\
                    sess.run([train_op, step_summary_op], feed_dict=fd)
                summary_writers['train'].add_summary(summary, step)
            # eval every batch
            if eval_func and k==0 and batch>=1:
                train_data_metrics = eval_func(sess,model,train_data)
                dev_data_metrics = eval_func(sess,model,dev_data)
                test_data_metricss = [eval_func(sess,model,test_data)
                    for test_data in test_datas]
                def add_summary(writer, metric, step):
                    for name,value in metric.iteritems():
                        summary = tf.Summary(value=[                         
                            tf.Summary.Value(tag=name, simple_value=value),   
                            ])
                        writer.add_summary(summary, global_step=step)
                add_summary(summary_writers['train'], train_data_metrics, step)
                add_summary(summary_writers['dev'], dev_data_metrics, step)
                for i in range(len(test_data_metricss)):
                    add_summary(summary_writers['test_{}'.format(i)], test_data_metricss[i], step)
                score=dev_data_metrics[score_key]
                if score>best_score:
                    best_score=score
                    model.saver.save(sess, os.path.join(model_dir,model_name), global_step=step)
            step+=1
