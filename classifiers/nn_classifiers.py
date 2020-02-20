import numpy as np
import tensorflow as tf

from .nn_model import NN_model

class deep_NN(NN_model):
    def __init__(self, log_dir="./log/standard_NN/"):
        super().__init__(log_dir)
        
    def build(self, input_size,
              hidden_sizes=[256, 256,128,64] + [1], 
              activation_fns=[tf.sigmoid]*4 + [tf.sigmoid], 
              dropout=[0.5]*4, 
              learning_rate=0.005):
        with self.graph.as_default():
            
            opt_model = tf.train.AdamOptimizer(learning_rate)
            output_fn = multi_layer(hidden_sizes, activation_fns, is_train=self.model['train_mode'], dropout=dropout)

        super().build(input_size, output_fn, mean_cross_entr, opt_model)
        


def mean_cross_entr(labels, predictions):
    cross_entr = tf.losses.log_loss(labels = labels, predictions = predictions)
    return tf.reduce_mean(cross_entr)


def multi_layer(layer_sizes, activation_fns, is_train=False, dropout=None):
    return lambda x: multi_layer_core(x, layer_sizes, activation_fns, is_train=is_train, dropout=dropout)

def multi_layer_core(inputs, layer_sizes, activation_fns, is_train=False, dropout=None):
    '''
        Args:
            inputs:
            layer_sizes: Array of sizes of the layers within the neural network. 
                Starts with first hidden layer size and ends with the output size
            activation_fns: Array of activation functions for all layers. Starts with 
                function between input and first layer and ends with function for output
            is_train: Decides whether or not the dropout will take effect. Use a conditional
                value that is true in training and false in validation/prediction
            dropout: Array of dropout percentages for each layer. None for no dropout.    
    '''
    assert len(layer_sizes) == len(activation_fns)
    assert (dropout is None) or (len(layer_sizes) - 1 == len(dropout))
    
    prev_layer = inputs
    n_layers = len(layer_sizes) - 1
    
    for i in range(n_layers):  
        prev_layer = tf.contrib.layers.fully_connected(
            prev_layer,
            layer_sizes[i],
            activation_fn=activation_fns[i])
        
        if (dropout is not None):            
            prev_layer = tf.contrib.layers.dropout(
                prev_layer,
                keep_prob=dropout[i],
                is_training=is_train)
        
    return tf.contrib.layers.fully_connected(
            prev_layer,
            layer_sizes[n_layers],
            activation_fn=activation_fns[n_layers])






