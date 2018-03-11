# encoding=utf-8
import sys
import os
from os import makedirs
from shutil import rmtree
import json

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score

from utils.word2vec import load_w2v
from tf_utils.model import multi_filter_sizes_cnn_debug, multi_filter_sizes_cnn
from tf_utils.model.rnn import  rnn_func
from tf_utils import load_config
from utils.base import get_vocab
from utils.base import get_func_args

config=None

'''
A general TextModel controled by parameter and config
'''
class TextModel(object):
    def __init__(self,
            num_classes,
            vocab_size=None,
            init_emb=None,
            sub_init_emb=None,
            emb_name=None,
            reuse=None,
            mode='train',
            name_scope=None,
            configp=None, 
            debug=False, 
            class_weights=None, *args, **kwargs):
        global config
        config=configp
        
        args_=get_func_args()
        for arg in args_:
            setattr(self, arg, args_[arg])
        for k,v in args:
            setattr(self, k, v)
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        
        with tf.name_scope(self.name_scope):
            self.build_inputs()
            self.build_embeddings()
            self.build_text_repr()
            self.build_outputs()
            self.build_others()

            
    def build_inputs(self):
        if config.objects=="tag":
            self.input_y = tf.placeholder(tf.int64, [None, self.num_classes], name="input_y")
        elif config.objects=="seq_tag":
            self.input_y = tf.placeholder(tf.int64, [None, config.seq_len], name="input_y")

        self.input_x = tf.placeholder(tf.int64, [None, config.seq_len], name="input_x")
        self.inputs=[ self.input_y, self.input_x ]
        if config.tok=="word_char":
            self.input_x_sub = tf.placeholder(tf.int64, [None, config.seq_len, config.char_len], name="input_x_sub")
            self.input_x_sub_length = tf.placeholder(tf.int64, [None, config.seq_len], name="input_l")
            self.inputs.append(self.input_x_sub)
            self.inputs.append(self.input_x_sub_length)
        if config.use_seq_length:
            self.input_sequence_length = tf.placeholder(tf.int64, [None], name="input_l")
            self.inputs.append(self.input_sequence_length)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.inputs.append(self.dropout_keep_prob)
        self.batch_size=tf.shape(self.input_x)[0]
        self.outputs = []
        self.debugs = []
        if config.text_model=='add+idf':
            self.idf = tf.Variable(self.init_idf, dtype=tf.float32, name='idf', trainable=False)

    def build_embeddings(self):
        # Embedding layer
        # tf not allowed init value of any tensor >2G
        def build_emb(init_emb, name):
            if init_emb is not None:
                init_emb=tf.constant(init_emb, dtype=tf.float32)
            else:
                pass
                # init_emb=np.array
            W = tf.get_variable(name, initializer=init_emb, trainable=True)
            return W
        self.word_W=build_emb(self.init_emb, "word_W")
        self.words_vec = tf.cast(tf.nn.embedding_lookup(self.word_W, self.input_x), tf.float32)
        self.vec_size=config.words_vec.vec_size
        if config.tok=="word_char":
            self.char_W=build_emb(self.sub_init_emb, "char_word_W")
            self.chars_vec = tf.cast(tf.nn.embedding_lookup(self.char_W, self.input_x_sub), tf.float32)
            self.dbg=chars_vec=tf.reshape(self.chars_vec,[config.batch_size*config.seq_len,config.char_len,-1])

            word_vec= multi_filter_sizes_cnn(tf.expand_dims(chars_vec,-1),config.char_len, config.chars_vec.vec_size, config.char_filter_sizes, config.char_filter_nums, name='char_cnn', reuse=False)
            cw_vec=tf.reshape(word_vec,[config.batch_size, config.seq_len, -1])
            self.words_vec=tf.concat([self.words_vec, cw_vec], -1)
            self.vec_size=config.words_vec.vec_size+sum(config.char_filter_nums)
        self.words_emb = self.words_vec
            
    def build_sent_cnn(self):
        words_emb = tf.expand_dims(self.words_emb, -1)
        if self.debug:
            self.sent_vec, self.pooled_index = multi_filter_sizes_cnn_debug(words_emb, config.seq_len, config.vec_size, config.filter_sizes, config.filter_nums, name='cnn', reuse=self.reuse)
        else:
            for i in range(config.cnn_layer_num):
                words_emb= multi_filter_sizes_cnn(words_emb, config.seq_len, self.vec_size, config.filter_sizes, config.filter_nums, name='cnn'+str(i), pooling=False, reuse=self.reuse, gated=config.gated, padding="same")
        self.words_emb=words_emb        
        #self.hidden_size=sum(config.filter_nums)
            
    def build_sent_add(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.seq_len), tf.float32)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0],1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_sent_add_idf(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.seq_len), tf.float32)
        self.idf_x = tf.nn.embedding_lookup(self.idf, self.input_x)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0]\
                    *tf.expand_dims(self.idf_x,-1),1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
    
    def build_hs_atta(self):
        pass
    
    def build_rnn(self, bi=False, attn_type=None):
        self.words_emb=rnn_func(self.words_emb, config.rnn_cell, config.cell_size, config.rnn_layer_num, sequence_length=self.input_sequence_length, attn_type=attn_type, bi=bi)
    
    def build_gated_cnn(self):
        word_emb=self.words_emb
        vec_size=self.vec_size
        for i in range(config.cnn_layers):
            words_emb = multi_filter_sizes_cnn(words_emb, config.seq_len, vec_size, config.filter_sizes, config.filter_nums, name='cnn'+str(i), reuse=False, pooling=False, padding="same") 
            vec_size=sum(config.filter_nums)

    def build_cnn_rnn(self):
        words_emb= multi_filter_sizes_cnn(self.words_emb, config.seq_len, self.vec_size, config.filter_sizes, config.filter_nums, name='cnn', reuse=False, pooling=False, padding="same")
        words_emb=rnn_func(words_emb, config.rnn_cell, config.cell_size, config.rnn_layer_num, sequence_length=self.input_sequence_length,attn_type=config.attn_type, bi=config.bi)
    
    def build_graph_cnn(self):
        '''
        several ways to do it:
        * make batch=1, generate sparse/dense W=[N*b,N*D] by Conv and graph link
          niubility: use tf.nn.embedding_lookup_sparse(Conv, sparse_L_L)
        * reshape to place indexs with same context_size, and reshape back. 
            could do with batch>1, but code some difficult
        implement sparse first.
        '''
        pass 

    def build_text_repr(self):
        if config.text_model=='add':# consider add from context
            self.build_sent_add()
        elif config.text_model=='add_idf':
            self.build_sent_add_idf()
        else:
            if config.text_model=='cnn':
                self.build_sent_cnn()
            elif config.text_model in ['rnn','birnn','rnn_attn','birnn_attn']:
                self.build_rnn(bi=config.bi, attn_type=config.attn_type)
            elif config.text_model=='cnn_rnn':
                self.build_cnn_rnn()
            elif config.text_model=='hs-rnn-attn':
                self.build_hs_rnn_attn()
            if config.objects=="tag":
                if config.text_model in ['cnn', 'gated-cnn','rnn-cnn']:
                    self.sent_vec=tf.reduce_max(self.words_emb,1)
                else:
                    self.sent_vec=self.words_emb[:,-1,:]
        if config.text_model in ['cnn', 'gated-cnn','rnn-cnn']:
            self.repr_size=sum(config.filter_nums)
        elif 'rnn' in config.text_model:
            self.repr_size=config.cell_size if not config.bi else 2*config.cell_size
             
        if config.objects=="tag":
            with tf.name_scope("dropout"):
                self.sent_vec = tf.nn.dropout(self.sent_vec, self.dropout_keep_prob)
        
    def build_exclusive_ouputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            self.scores = tf.layers.dense(inputs, num_classes, name="dense", reuse=self.reuse)
            if self.debug:
                inputs2=tf.stack([inputs,]*self.num_classes,-1)
                self.scores2 = tf.multiply(inputs2, tf.expand_dims(W,0))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.outputs.append(tf.nn.softmax(self.scores))
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            if config.use_label_weights:
                class_weights = self.class_weights#这里做了个糟糕的假设，假设input_y[i]只有一个是1，其余是0
                weights = tf.reduce_sum(class_weights * tf.cast(self.input_y,tf.float32),axis=1)
                losses = losses * weights
             
            self.l2_loss = tf.add_n([ tf.cast(tf.nn.l2_loss(v), tf.float32) for v in tf.trainable_variables() if 'bias' not in v.name ])
            self.loss = tf.reduce_mean(losses) + config.l2_lambda * self.l2_loss
        tf.summary.scalar("loss", self.loss)    
        self.outputs.append(self.loss)
        
        # Accuracy
        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.outputs.append(correct_predictions)
        # tf.summary.scalar("accuracy", self.accuracy)    

    def build_nonexclusive_outputs(self, inputs, num_classes):
        '''有几种方法来处理不互斥的tags
        '''
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                self.scores = tf.layers.dense(inputs, num_classes*2, name="dense", reuse=self.reuse)
            self.scores = tf.reshape(self.scores, [self.batch_size, num_classes, 2])
            self.predictions = tf.argmax(self.scores, -1, name="predictions")
            self.outputs.append(tf.nn.softmax(self.scores))
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):
            input_y = tf.one_hot(self.input_y,depth=2,axis=-1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=input_y)
            if config.use_label_weights:
                losses = losses*tf.expand_dims(self.class_weights,0)
            self.loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)
        self.outputs.append(self.loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.class_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), axis = 0, name="class_accuracy")
        # tf.summary.scalar("accuracy", self.accuracy)

    def build_noexclusive_sampled_outputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            #weights = tf.get_variable("weights",shape=[num_classes, sum(self.num_filters)],dtype=tf.float32,\
            #        initializer=tf.contrib.layers.xavier_initializer())
            #biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32,\
            #        initializer=tf.constant_initializer(0.2))
            weights = tf.get_variable("weights",shape=[num_classes, self.repr_len],dtype=tf.float32)
            biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32)
            tf.summary.histogram('weights',weights)
            tf.summary.histogram('biases',biases)

        # as class number is big, use sampled softmax instead dense layer+softmax
        with tf.name_scope("loss"):
            tags_prob = tf.pad(self.input_y_prob,[[0,0],[0,config.num_sampled]])
            out_logits, out_labels= _compute_sampled_logits( weights, biases, self.input_y, inputs,\
                    config.num_sampled, num_classes, num_true= config.max_tags )
            # TODO:check out_labels keep order with inpuy
            weighted_out_labels = out_labels * tags_prob*config.max_tags
            # self.out_labels = weighted_out_labels
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out_logits, labels=weighted_out_labels))
        
        with tf.name_scope("outputs"):
            logits = tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)
            self.output_values, self.ouput_indexs = tf.nn.top_k(logits, config.topn)

        with tf.name_scope("score"):
            self.score = self.loss/tf.cast(self.batch_size, tf.float32)
            #self.accuracy = tf.reduce_sum( self.top_prob )
   
        tf.summary.scalar('loss', self.loss)
    
    def build_noexclusive_sigmoid_outputs(self, inputs, num_classes):
        self.scores = tf.layers.dense(inputs, num_classes, name="dense", reuse=self.reuse)
        pass

    def build_seq_tag_ouptuts(self, inputs, num_classes):
        ntime_steps = tf.shape(inputs)[1]
        vec_len=tf.shape(inputs)[-1]
        inputs = tf.reshape(inputs, [-1, self.repr_size])
        self.scores = tf.layers.dense(inputs, num_classes, name="dense", reuse=self.reuse)
        self.scores = tf.reshape(self.scores, [-1, ntime_steps, num_classes])
        if config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.scores, tf.cast(self.input_y, tf.int32), self.input_sequence_length)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.predictions=tf.contrib.crf.crf_decode( self.scores, self.transition_params, self.input_sequence_length)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            mask = tf.sequence_mask(self.input_sequence_length, config.seq_len)
            self.debugs.append(self.input_sequence_length)
            self.debugs.append(mask)
            self.losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(self.losses)
            self.predictions=tf.argmax(self.scores,-1)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
    
    def build_seq_sampled_tag_outputs(self):
        pass
        
        # consider seq2seq, encoder-decoder
    
    def build_outputs(self):
        if config.objects=="tag":
            if config.tag_exclusive:
                self.build_exclusive_ouputs(self.sent_vec, self.num_classes)
            else:
                self.build_nonexclusive_outputs(self.sent_vec, self.num_classes)
        elif config.objects=="seq_tag":
            self.build_seq_tag_ouptuts(self.words_emb, self.num_classes)

    def build_others(self):
        if self.mode=='train':
            global_step = tf.Variable(0, trainable=False) 
            starter_learning_rate = config.start_learning_rate
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                    config.decay_steps, config.decay_rate, staircase=True)
            # Passing global_step to minimize() will increment it at each step.
            tf.summary.scalar("learning_rate",self.learning_rate)
            
            if config.learning_method=='adam':
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
            elif config.learning_method=='adam_decay':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif config.learning_method=='sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif config.learning_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif config.learning_method == 'pro':
                optimizer = tf.train.ProximalGradientDescentOptimizer(self.learning_rate)
            if 'rnn' in config.text_model: 
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                        config.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            else:
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)
                
        self.step_summaries = tf.summary.merge_all()   
        self.init = tf.global_variables_initializer()
        self.all_vars=list(set(
            (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)+
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="share"))))
        self.train_vars=[x for x in self.all_vars if x in tf.trainable_variables()]
        self.all_saver=tf.train.Saver(self.all_vars)
        self.train_saver = tf.train.Saver(self.train_vars)
        print 'ALL VAR:\n\t', '\n\t'.join(str(x) for x in self.all_saver._var_list)
        print 'TRAIN VAR:\n\t', '\n\t'.join(str(x) for x in self.train_saver._var_list)
        print 'INPUTS:\n\t', '\n\t'.join(str(x) for x in self.inputs)
        print 'OUTPUTS:\n\t', '\n\t'.join(str(x) for x in self.outputs)
