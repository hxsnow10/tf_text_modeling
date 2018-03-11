# encoding=utf-8
import tensorflow as tf

class TFModel():

    def __init__(self, sess, model_path, input_names, output_names):
        self.sess=sess
        saver = tf.train.import_meta_graph("{}.meta".format(model_path), clear_devices=True)
        self.init = tf.global_variables_initializer()
        sess.run(self.init)
        saver.restore(self.sess, model_path)
        self.inputs=[tf.get_default_graph().get_tensor_by_name(name) for name in input_names]
        self.outputs=[tf.get_default_graph().get_tensor_by_name(name) for name in output_names]
        print self.inputs
        print self.outputs
        print 'MODEL LOADED SCCESSFULLY'

    def predict(self, input_data):
        fd=dict(zip(self.inputs, input_data))
        rval=self.sess.run(self.outputs, feed_dict=fd)
        return rval
