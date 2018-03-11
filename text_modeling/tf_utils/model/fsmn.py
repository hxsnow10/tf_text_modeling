import tensorflow as tf
import time
import numpy as np

class FSMN(object):
    '''
    the model of the fsmn and the ner.py will user it.
    '''
    def __init__(self, memory_size, input_size, output_size, dtype=tf.float32):
        self._memory_size = memory_size#5
        self._output_size = output_size#128
        self._input_size = input_size
        self._dtype = dtype
        self.build_graph()

    def build_graph(self):
        self._W1 = tf.get_variable("fsmnn_w1", [self._input_size, self._output_size], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._W2 = tf.get_variable("fsmnn_w2", [self._input_size, self._output_size], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._bias = tf.get_variable("fsmnn_bias", [self._output_size], initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        self._memory_weights = tf.get_variable("memory_weights", [self._memory_size], initializer=tf.constant_initializer(1.0, dtype=self._dtype))

    def __call__(self, input_data):
        batch_size = input_data.get_shape()[0].value
        num_steps = input_data.get_shape()[1].value

        aa = time.time()
        memory_matrix = []
        for step in range(num_steps):
            left_num = tf.maximum(0, step + 1 - self._memory_size)
            right_num = num_steps - step - 1
            mem = self._memory_weights[tf.minimum(step, self._memory_size)::-1]
            d_batch = tf.pad(mem, [[left_num, right_num]])
            memory_matrix.append([d_batch])
        memory_matrix = tf.concat(memory_matrix, 0)#200*200
        bb = time.time()
        cc = time.time()
        h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)#batch_size*200*128
        h = tf.matmul(input_data, [self._W1] * batch_size)#bs*200*128  X  bs*128*128  =  bs*200*128
        h += tf.matmul(h_hatt, [self._W2] * batch_size) + self._bias#bs*200*128
        dd = time.time()
        print "construct memory matrix:%f"%(bb - aa)
        print "get h:%f"%(dd - cc)
        return h

class BiFSMN(object):
    def __init__(self, memory_size_a, memory_size_c, input_size, output_size, dtype=tf.float32):
        self._memory_size_a = memory_size_a#5  look-ahead window size
        self._memory_size_c = memory_size_c#5  look-back window size
        self._output_size = output_size#128
        self._input_size = input_size
        self._dtype = dtype
        self.build_graph()
        print "using BiFSMN ..."

    def build_graph(self):
        self._W1 = tf.get_variable("fsmnn_w1", [self._input_size, self._output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._W2 = tf.get_variable("fsmnn_w2", [self._input_size, self._output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._bias = tf.get_variable("fsmnn_bias", [self._output_size],
                                     initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        self._memory_weights_a = tf.get_variable("memory_weights_a", [self._memory_size_a],
                                               initializer=tf.constant_initializer(1.0, dtype=self._dtype))
        self._memory_weights_c = tf.get_variable("memory_weights_c", [self._memory_size_c],
                                                 initializer=tf.constant_initializer(1.0, dtype=self._dtype))

    def __call__(self, input_data):
        batch_size = input_data.get_shape()[0].value
        num_steps = input_data.get_shape()[1].value

        aa = time.time()
        memory_matrix = []
        for step in range(num_steps):
            left_num = tf.maximum(0, step + 1 - self._memory_size_a)
            right_num = tf.maximum(0, num_steps - step - 1 - self._memory_size_c)
            mem_ahead = self._memory_weights_a[tf.minimum(step, self._memory_size_a)::-1]#look-ahead
            mem_back = self._memory_weights_c[:tf.minimum(num_steps-step-1, self._memory_size_c)]# look-back
            mem = tf.concat([mem_ahead, mem_back], 0)
            d_batch = tf.pad(mem, [[left_num, right_num]])
            memory_matrix.append([d_batch])
        memory_matrix = tf.concat(memory_matrix, 0)#200*200
        bb = time.time()
        cc = time.time()
        h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)#batch_size*200*128
        h = tf.matmul(input_data, [self._W1] * batch_size)#bs*200*128  X  bs*128*128  =  bs*200*128
        h += tf.matmul(h_hatt, [self._W2] * batch_size) + self._bias#bs*200*128
        dd = time.time()
        print "construct memory matrix:%f"%(bb - aa)
        print "get bifsmn h:%f"%(dd - cc)
        return h

# attention-based bifsmn
class AttbBiFSMN(object):
    def __init__(self, memory_size_a, memory_size_c, input_size, output_size, seq_len=200, dtype=tf.float32):
        self._memory_size_a = memory_size_a  # 5  look-ahead window size
        self._memory_size_c = memory_size_c  # 5  look-back window size
        self._memory_size = memory_size_a + memory_size_c
        self._output_size = output_size  # 128
        self._input_size = input_size #128
        self._seq_len = seq_len#200
        self._dtype = dtype
        self.build_graph()
        print "using attention-based BiFSMN ..."

    # def build_graph(self):
    #     self._W1 = tf.get_variable("fsmnn_w1", [self._input_size, self._output_size],
    #                                initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
    #     self._W2 = tf.get_variable("fsmnn_w2", [self._input_size, self._output_size],
    #                                initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
    #     self._bias = tf.get_variable("fsmnn_bias", [self._output_size],
    #                                  initializer=tf.constant_initializer(0.0, dtype=self._dtype))
    #     self._V = tf.get_variable("attention_V", [self._memory_size, self._seq_len],
    #                                  initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
    #     self._U = tf.get_variable("attention_U", [self._output_size, 1],
    #                               initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
    #     self._m = tf.get_variable("attention_m", [self._seq_len, 1],
    #                               initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
    #     # self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
    #     #                                          initializer=tf.constant_initializer(1.0, dtype=self._dtype))


    # def __call__(self, input_data):
    #     batch_size = input_data.get_shape()[0].value
    #     num_steps = input_data.get_shape()[1].value
    #     # embd_size = input_data.get_shape()[2].value
    #     memory_block = []
    #     aa = time.time()
    #     for i in range(batch_size):
    #         inp = input_data[i]
    #         self._memory_weights = tf.matmul(self._V, tf.nn.relu(tf.matmul(inp, self._U)+self._m))
    #         self._memory_weights = tf.reshape(self._memory_weights, [self._memory_size])
    #         self._memory_weights_a = self._memory_weights[:self._memory_size_a]
    #         self._memory_weights_c = self._memory_weights[-self._memory_size_c:]
    #
    #         memory_matrix = []
    #         for step in range(num_steps):
    #             left_num = tf.maximum(0, step + 1 - self._memory_size_a)
    #             right_num = tf.maximum(0, num_steps - step - 1 - self._memory_size_c)
    #             mem_ahead = self._memory_weights_a[tf.minimum(step, self._memory_size_a)::-1]  # look-ahead
    #             mem_back = self._memory_weights_c[:tf.minimum(num_steps - step - 1, self._memory_size_c)]  # look-back
    #             mem = tf.concat([mem_ahead, mem_back], 0)
    #             d_batch = tf.pad(mem, [[left_num, right_num]])
    #             memory_matrix.append([d_batch])
    #         memory_matrix = tf.concat(memory_matrix, 0)  # 200*200
    #         memory_block.append(memory_matrix)
    #     bb = time.time()
    #     cc = time.time()
    #     # h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)  # batch_size*200*128
    #     h_hatt = tf.matmul(memory_block, input_data)  # batch_size*200*128
    #     h = tf.matmul(input_data, [self._W1] * batch_size)  # bs*200*128  X  bs*128*128  =  bs*200*128
    #     h += tf.matmul(h_hatt, [self._W2] * batch_size) + self._bias  # bs*200*128
    #     dd = time.time()
    #     print "construct memory matrix:%f" % (bb - aa)
    #     print "get attention-based bifsmn h:%f" % (dd - cc)
    #     return h


    def build_graph(self):
        # self._W1 = tf.get_variable("fsmnn_w1", [self._input_size, self._output_size],
        #                            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        # self._W2 = tf.get_variable("fsmnn_w2", [self._input_size, self._output_size],
        #                            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        # self._bias = tf.get_variable("fsmnn_bias", [self._output_size],
        #                              initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        # self._V = tf.get_variable("attention_V", [1, self._memory_size],
        #                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        # self._U = tf.get_variable("attention_U", [self._output_size, 1],
        #                           initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        # self._m = tf.get_variable("attention_m", [1],
        #                           initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        # # self._memory_weights = tf.get_variable("memory_weights", [self._memory_size],
        # #                                          initializer=tf.constant_initializer(1.0, dtype=self._dtype))

        self._W1 = tf.Variable(self.xaiver_init(self._input_size, self._output_size))
        self._W2 = tf.Variable(self.xaiver_init(self._input_size, self._output_size))
        # self._bias = tf.get_variable("fsmnn_bias", [self._output_size],
        #                              initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        self._bias = tf.Variable(tf.constant(0.01, shape=[self._output_size]))
        # self._V = tf.Variable(self.xaiver_init(1, self._memory_size))#v4 -> v7
        self._V = tf.Variable(self.xaiver_init(self._memory_size, self._seq_len))#v5 -> v6
        self._U = tf.Variable(self.xaiver_init(self._output_size, 1))
        # self._m = tf.get_variable("attention_m", [1],initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))#v4 -> v7
        self._m = tf.Variable(self.xaiver_init(self._seq_len, 1))#scalar -> vector #v5 -> v6

    def xaiver_init(self, d1, d2, constant=1):
        low = -constant * np.sqrt(6.0 / (d1 + d2))
        high = constant * np.sqrt(6.0 / (d1 + d2))
        return tf.random_uniform((d1, d2), minval=low, maxval=high, dtype=tf.float32)


    def __call__(self, input_data):
        batch_size = input_data.get_shape()[0].value
        num_steps = input_data.get_shape()[1].value
        # embd_size = input_data.get_shape()[2].value
        memory_block = []
        aa = time.time()
        #v4 -> v7
        # self._memory_weights = tf.matmul(tf.tanh(tf.matmul(input_data, [self._U]*batch_size) + self._m), [self._V]*batch_size)
        #v5-> v6 ------start
        self._memory_weights = tf.matmul([self._V]*batch_size, tf.tanh(tf.matmul(input_data, [self._U]*batch_size) + self._m))
        self._memory_weights = tf.transpose(self._memory_weights, [0,2,1])
        #v5-> v6 ------end
        aaa = time.time()
        #TODO [::-1] for forward weights #v6
        for i in range(batch_size):

            # mem_w = self._memory_weights[i] #v4 -> v7
            mem_w0 = tf.reshape(self._memory_weights[i], [-1])#v5->v6
            mem_w = tf.stack([mem_w0]*self._seq_len)#v5->v6

            memory_matrix = tf.zeros([num_steps, num_steps+self._memory_size-1])
            for j in range(self._memory_size):
                #v6 -> v7
                if j < self._memory_size_a:
                    invj = self._memory_size_a-j-1
                else:
                    invj = j
                col_i = tf.diag(mem_w[:, invj])
                # col_i = tf.diag(mem_w[:,j])#v5
                left_num = j
                right_num = self._memory_size - j - 1
                d_batch = tf.pad(col_i, [[0,0],[left_num, right_num]])
                memory_matrix += d_batch
            memory_matrix = memory_matrix[:, self._memory_size_a-1:-self._memory_size_c]
            # memory_matrix = []
            # for step in range(num_steps):
            #     memory_weights_tmp = mem_w[step, :]
            #     self._memory_weights_a = memory_weights_tmp[:self._memory_size_a]
            #     self._memory_weights_c = memory_weights_tmp[-self._memory_size_c:]
            #     left_num = tf.maximum(0, step + 1 - self._memory_size_a)
            #     right_num = tf.maximum(0, num_steps - step - 1 - self._memory_size_c)
            #     mem_ahead = self._memory_weights_a[tf.minimum(step, self._memory_size_a)::-1]  # look-ahead
            #     mem_back = self._memory_weights_c[:tf.minimum(num_steps - step - 1, self._memory_size_c)]  # look-back
            #     mem = tf.concat([mem_ahead, mem_back], 0)
            #     d_batch = tf.pad(mem, [[left_num, right_num]])
            #     memory_matrix.append([d_batch])
            # memory_matrix = tf.concat(memory_matrix, 0)  # 200*200
            memory_block.append(memory_matrix)
        bb = time.time()
        cc = time.time()
        # h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)  # batch_size*200*128
        h_hatt = tf.matmul(memory_block, input_data)  # batch_size*200*128
        h = tf.matmul(input_data, [self._W1] * batch_size)  # bs*200*128  X  bs*128*128  =  bs*200*128
        h += tf.matmul(h_hatt, [self._W2] * batch_size) + self._bias  # bs*200*128
        dd = time.time()
        print "attention based process cost time: %s" % (aaa-aa)
        print "construct memory matrix:%f" % (bb - aa)
        print "get attention-based bifsmn h:%f" % (dd - cc)
        return h


    def gen_mem_mat(self, mem_w, num_steps, mem_size_a, mem_size_c):
        memory_matrix = []
        for step in range(num_steps):
            memory_weights_tmp = mem_w[step, :]
            self._memory_weights_a = memory_weights_tmp[:self._memory_size_a]
            self._memory_weights_c = memory_weights_tmp[-self._memory_size_c:]
            left_num = tf.maximum(0, step + 1 - self._memory_size_a)
            right_num = tf.maximum(0, num_steps - step - 1 - self._memory_size_c)
            mem_ahead = self._memory_weights_a[tf.minimum(step, self._memory_size_a)::-1]  # look-ahead
            mem_back = self._memory_weights_c[:tf.minimum(num_steps - step - 1, self._memory_size_c)]  # look-back
            mem = tf.concat([mem_ahead, mem_back], 0)
            d_batch = tf.pad(mem, [[left_num, right_num]])
            memory_matrix.append([d_batch])
        memory_matrix = tf.concat(memory_matrix, 0)
        return memory_matrix
