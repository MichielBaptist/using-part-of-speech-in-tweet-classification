import numpy as np
import tensorflow as tf

from .rnn_model import RNN_model
from .rnn_model import RNN_model_pos

class basicLSTM(RNN_model):
    def __init__(self, log_dir="./log/LSTM_basic/"):
        super().__init__(log_dir)
    
    def build(self, predef_embed, hidden_size=128, learning_rate=0.001):
        with self.graph_.as_default():
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            opt_model = tf.train.AdamOptimizer(learning_rate)
            #train_mode = tf.placeholder(tf.bool, shape=(), name="train_mode")
            #print(train_mode.shape)

        super().build(predef_embed, cell, sigmoid, mean_cross_entr, opt_model)
    
class pos_tag_classifier(RNN_model_pos):
    def __init__(self, log_dir="./log/LSTM_basic/"):
        super().__init__(log_dir)
        
    def build(self, predef_embed, hidden_size=[128], n_layers=2, output_size=[128], 
              output_layers=3, learning_rate=0.001, keep_prob=[1.0, 1.0, 0.5], pos_emb_shape=(45,10)):
        with self.graph.as_default():
            #train_prob = tf.constant(keep_prob)
            #pred_prob = tf.constant([1, 1, 1])
        
            drop_out = tf.cond(self.model['train_mode'], 
                               true_fn= lambda: keep_prob, 
                               false_fn= lambda: [1.0, 1.0, 1.0])            
            
            cell = LSTM_layers(hidden_size, n_layers, drop_out)
            opt_model = tf.train.AdamOptimizer(learning_rate)
            output_fn = multi_layer(output_layers, output_size, [tf.sigmoid])

        super().build(predef_embed, pos_emb_shape, cell, output_fn, mean_cross_entr, opt_model)
    
class multi_layer_LSTM(RNN_model):
    def __init__(self, log_dir="./log/LSTM_basic/"):
        super().__init__(log_dir)
        
    def build(self, predef_embed, hidden_size=[128], n_layers=2, output_layers=3, 
              output_size=[128], learning_rate=0.001, keep_prob=[1.0, 1.0, 0.5]):
        with self.graph.as_default():
        
            drop_out = tf.cond(self.model['train_mode'], 
                               true_fn= lambda: keep_prob, 
                               false_fn= lambda: [1.0, 1.0, 1.0])            
            
            cell = LSTM_layers(hidden_size, n_layers, drop_out)
            opt_model = tf.train.AdamOptimizer(learning_rate)
            output_fn = multi_layer(output_layers, output_size, [tf.sigmoid])

        super().build(predef_embed, cell, output_fn, mean_cross_entr, opt_model)

def drop_out_cell(hidden_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                         input_keep_prob=keep_prob[0],
                                         state_keep_prob=keep_prob[1],
                                         output_keep_prob=keep_prob[2])
    return cell

def LSTM_layers(hidden_size, n_layers, keep_prob):
    m = len(hidden_size)
    layers = [drop_out_cell(hidden_size[i%m], keep_prob) for i in range(n_layers)]
    
    return tf.contrib.rnn.MultiRNNCell(layers)
    
    
def sigmoid(output):
    n, m = output.shape.as_list()
    w_out = tf.get_variable("w_out", shape=[m, 1], dtype=tf.float32, initializer=tf.random_normal_initializer )
    b_out = tf.get_variable("b_out", shape=[1], dtype=tf.float32, initializer=tf.random_normal_initializer )
    return tf.sigmoid(tf.matmul(output, w_out) + b_out)

def multi_layer(n_layers, layer_sizes, activation_fns, output_size=1, out_put_fn=tf.sigmoid):
    return lambda x: multi_layer_core(x, n_layers, layer_sizes, activation_fns, output_size, out_put_fn)

def multi_layer_core(inputs, n_layers, layer_sizes, activation_fns, output_size=1, out_put_fn=tf.sigmoid):
    prev_layer = inputs
    l, k = len(layer_sizes), len(activation_fns)
    for i in range(n_layers-1):
        prev_layer = tf.contrib.layers.fully_connected(
            prev_layer,
            layer_sizes[i%l],
            activation_fn=activation_fns[i%k])
    return tf.contrib.layers.fully_connected(
                prev_layer,
                num_outputs=output_size,
                activation_fn=out_put_fn)

def mean_cross_entr(labels, predictions):
    cross_entr = tf.losses.log_loss(labels = labels, predictions = predictions)
    return tf.reduce_mean(cross_entr)
    
    
