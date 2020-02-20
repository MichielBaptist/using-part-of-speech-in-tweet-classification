import numpy as np
import tensorflow as tf
from tqdm import tqdm

import math

from collections import namedtuple
from keras.preprocessing.sequence import pad_sequences

'''
Note: This code is based on the following article: 
https://medium.com/@Currie32/predicting-movie-review-sentiment-with-tensorflow-and-tensorboard-53bf16af0acf 
We used this as a startingpoint while learning tensorflow.
Many changes have been made mainly for generalization purposes.
There will however still be similarites with the code from the article.
'''


class RNN_model():
    '''A model/core which can be expanded on to run different Recurrent Neural Networks (RNN)'''
    def __init__(self, log_dir):
        self.graph = tf.Graph()
        self.sess_init = False
        self.log_dir = log_dir    
        self.model = {}
        with self.graph.as_default(), tf.name_scope('mode'):
            train_mode = tf.placeholder(tf.bool, shape=(), name="train_mode")
            self.model["train_mode"] = train_mode
            
    def build(self, predef_embed, cell, pred_model, cost_fn, opt_model):
        '''Build the Recurrent Neural Network
        
            Args:
                predef_embed: Predefined embeddings for data
                cell: RNN cell to be used in the network. (E.g. an LSTM cell or a multi-layered cell)
                pred_model: Model for going from RNN-output to prediction
                cost_fn: The function to be minimized (E.g. cross entropy)
                opt_model: Model for correcting weights. (E.g. Gradient descent)
        '''
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
                batch_size = tf.shape(inputs)[0]
                
            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            
            # Create the embeddings
            with tf.name_scope("embeddings"):
                embedding = tf.constant(predef_embed, name="Embeddings")
                embed = tf.nn.embedding_lookup(embedding, inputs)
                
            with tf.name_scope("RNN_init_state"):
                initial_state = cell.zero_state(batch_size, tf.float32)

            # Run the data through the RNN layers
            with tf.name_scope("RNN_forward"):
                outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
                output = outputs[:,-1, :]
            
            with tf.name_scope('predictions'):
                predictions = pred_model(output)
                tf.summary.histogram('predictions', predictions)

            with tf.name_scope('cost'):
                cost = cost_fn(labels, predictions)       
                tf.summary.scalar('cost', cost)

            # Train the model
            with tf.name_scope('train'):    
                optimizer = opt_model.minimize(cost)
                
            # Determine the accuracy
            with tf.name_scope("accuracy"):
                correct_pred = tf.equal(tf.round(predictions), labels)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

            # Merge all of the summaries
            merged = tf.summary.merge_all()    

            # Export the nodes 
            export_nodes = ['inputs', 'labels', 'final_state', 'accuracy',
                            'predictions', 'cost', 'optimizer', 'merged']
            local_dict = locals()
            for name in export_nodes:
                self.model[name] = local_dict[name]

    def train(self, X, y, epochs, batch_size, log_string, stop_at=3, measures_per_epoch=2):
        '''Train the RNN
            Args:
                X: Tuple of lists corresponding to train and validation data
                y: Tuple of lists corresponding to train and validation labels
                epochs: Number of training cycles through the whole training set
                log_string: Identifier when writing to log files
                measures_per_epoch: Amount of times per epoch to measure validation and
                                    training accuracy and loss. Only for logging purposes
            Returns:
                path to saved model
        '''
        
        open('{}/intermediate/{}'.format(self.log_dir, log_string), 'w').close()

        X_t, X_v = X
        y_t, y_v = y        
        if (not self.sess_init):
            with self.graph.as_default():
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())
                self.sess_init = True
        
        
        with self.graph.as_default() as G, self.session as sess:
            saver = tf.train.Saver()
            model = self.model
            
            # Used to determine when to stop the training early
            valid_loss_summary = []
            iteration = 0

            print("Training Model: {}".format(log_string))
            train_writer = tf.summary.FileWriter('{}/train/{}'.format(self.log_dir, log_string), sess.graph)
            valid_writer = tf.summary.FileWriter('{}/train/{}'.format(self.log_dir, log_string))

            for e in range(epochs):
                # Record progress with each epoch
                train_loss, val_loss = [], []
                train_acc, val_acc = [], []

                # Compute points to stop at for validation
                n_batches_total = math.ceil(len(X_t)/batch_size)
                batches_before_validation = math.floor(n_batches_total/measures_per_epoch)
                
                with tqdm(total=len(X_t)) as pbar:                    
                    v_loss_per_b, t_loss_per_b = [], []
                    v_acc_per_b, t_acc_per_b = [], []            
                    
                    for _, (X, y) in enumerate(get_batches(X_t, y_t, batch_size), 1):
                        feed = {model['inputs']: pad_sequences(X),
                                model['labels']: y,
                                model['train_mode']: True}

                        summary, loss, acc, state, _ = (
                            sess.run([model['merged'], 
                                      model['cost'], 
                                      model['accuracy'], 
                                      model['final_state'], 
                                      model['optimizer']], 
                                     feed_dict=feed))

                        # Record the loss and accuracy of each training batch
                        train_loss.append(loss)
                        train_acc.append(acc)
                        
                        # Record the loss and accuracy within the scope of the current measurement
                        t_loss_per_b.append(loss)
                        t_acc_per_b.append(acc)                        

                        # Record the progress of training
                        train_writer.add_summary(summary, iteration)

                        iteration += 1
                        pbar.update(len(X))
                        
                        if iteration % batches_before_validation == 0:
                            # Get validation loss and accuracy
                            avg_valid_loss, avg_valid_acc, summary = self.validate(X_v, y_v, batch_size)
                            
                            #Average over batches
                            t_loss_avg = np.mean(t_loss_per_b)
                            t_acc_avg = np.mean(t_acc_per_b)
                            
                            # Writing training and validation results to file
                            with open('{}/intermediate/{}'.format(self.log_dir, log_string), 'a') as int_file:
                                line = " ".join(map(str, [t_loss_avg, t_acc_avg, avg_valid_loss, avg_valid_acc]))
                                int_file.write(line)
                                int_file.write("\n")
                            
                            # Reset
                            v_loss_per_b, t_loss_per_b = [], []
                            v_acc_per_b, t_acc_per_b = [], []
                            

                # Average the training loss and accuracy of each epoch
                avg_train_loss = np.mean(train_loss)
                avg_train_acc = np.mean(train_acc) 
                        
                # Save validation loss for early stop condition
                valid_loss_summary.append(avg_valid_loss)

                # Record the validation data's progress
                valid_writer.add_summary(summary, iteration)

                # Print the progress of each epoch
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Train Loss: {:.3f}".format(avg_train_loss),
                      "Train Acc: {:.3f}".format(avg_train_acc),
                      "Valid Loss: {:.3f}".format(avg_valid_loss),
                      "Valid Acc: {:.3f}".format(avg_valid_acc))

                # Stop training if the validation loss does not decrease after <stop_at> epochs
                if avg_valid_loss > min(valid_loss_summary):
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop_at:
                        break   

                # Reset stop_early if the validation loss finds a new low
                # Save a checkpoint of the model
                else:
                    print("New Record!")
                    stop_early = 0
                    checkpoint ="{}/checkpoints/{}.ckpt".format(self.log_dir, log_string)
                    saver.save(sess, checkpoint)
            self.load(checkpoint)
            return checkpoint
        
    def validate(self, X_val, y_val, batch_size):
        '''Records accuracy and loss of give data'''
        sess = self.session
        model = self.model
        val_loss = []
        val_acc = []
        for X, y in get_batches(X_val, y_val, batch_size):
            feed = {model['inputs']: pad_sequences(X),
                    model['labels']: y,
                    model['train_mode']: False}

            summary, batch_loss, batch_acc = (
                sess.run([model['merged'],                         
                          model['cost'], 
                          model['accuracy']], 
                         feed_dict=feed))

            # Record the validation loss and accuracy of each epoch
            val_loss.append(batch_loss)
            val_acc.append(batch_acc)
        return np.mean(val_loss), np.mean(val_acc), summary

    def predict(self, X_test, batch_size):
        '''Get predictions from data'''
        assert(self.sess_init)
        
        all_preds = []
        model = self.model
        
        with self.session as sess:            
            for _, X in enumerate(get_test_batches(X_test, 
                                                   batch_size), 1):
                feed = {model['inputs']: pad_sequences(X),
                        model['train_mode']: False}
                predictions = sess.run(model['predictions'], feed_dict=feed)
                all_preds.extend(predictions)
        return np.array(all_preds)

    def load(self, path):
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            try:
                saver = tf.train.Saver()
                saver.restore(self.session, path)
                self.sess_init = True
            except Exception as e:
                print(e)
            finally:
                return self.sess_init
        
    def save(self, path):
        if self.sess_init:
            saver = tf.train.Saver()
            saver.save(self.session, path)
            
class RNN_model_pos():
    def __init__(self, log_dir):
        self.graph = tf.Graph()
        self.sess_init = False
        self.log_dir = log_dir    
        self.model = {}
        with self.graph.as_default(), tf.name_scope('mode'):
            train_mode = tf.placeholder(tf.bool, shape=(), name="train_mode")
            self.model["train_mode"] = train_mode
            
    def build(self, predef_embed, pos_emb_shape, cell, pred_model, cost_fn, opt_model):
        '''Build the Recurrent Neural Network        
            Args:
                predef_embed: Predefined embeddings for data
                pos_emb_shape: Wanted shape of embeddings for Part-Of-Speech data
                cell: RNN cell to be used in the network. (E.g. an LSTM cell)
                pred_model: Model for going from RNN-output to prediction
                cost_fn: The function to be minimized
                opt_model: Model for correcting weights. (E.g. Gradient descent)               
        '''
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
                batch_size = tf.shape(inputs)[0]
                
            with tf.name_scope('inputs_pos'):
                inputs_pos = tf.placeholder(tf.int32, [None, None], name='inputs_pos')

            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            
            # Create the embeddings
            with tf.name_scope("embeddings"):
                embedding = tf.constant(predef_embed, name="Embeddings")                
                embedding_pos = tf.get_variable("Embeddings_pos", pos_emb_shape)
                
                embed_pos = tf.nn.embedding_lookup(embedding_pos, inputs_pos)
                embed = tf.nn.embedding_lookup(embedding, inputs)
                
                joined_embedding = tf.concat([embed, embed_pos], 2)
                
            with tf.name_scope("RNN_init_state"):
                initial_state = cell.zero_state(batch_size, tf.float32)

            # Run the data through the RNN layers
            with tf.name_scope("RNN_forward"):
                outputs, final_state = tf.nn.dynamic_rnn(cell, joined_embedding, initial_state=initial_state)
                output = outputs[:,-1, :]
            
            with tf.name_scope('predictions'):
                predictions = pred_model(output)
                tf.summary.histogram('predictions', predictions)

            with tf.name_scope('cost'):
                cost = cost_fn(labels, predictions)       
                tf.summary.scalar('cost', cost)

            # Train the model
            with tf.name_scope('train'):    
                optimizer = opt_model.minimize(cost)
                
            # Determine the accuracy
            with tf.name_scope("accuracy"):
                correct_pred = tf.equal(tf.round(predictions), labels)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

            # Merge all of the summaries
            merged = tf.summary.merge_all()    

            # Export the nodes 
            export_nodes = ['inputs', 'inputs_pos', 'labels', 'final_state', 'accuracy',
                            'predictions', 'cost', 'optimizer', 'merged']
            local_dict = locals()
            for name in export_nodes:
                self.model[name] = local_dict[name]

    def train(self, X, y, epochs, batch_size, log_string, stop_at=3, measures_per_epoch = 2):
        '''Train the RNN
            Args:
                X: Tuple of lists corresponding to train and validation data
                y: Tuple of lists corresponding to train and validation labels
                epochs: Number of training cycles through the whole training set
                log_string: Identifier when writing to log files
                measures_per_epoch: Amount of times per epoch to measure validation and
                                    training accuracy and loss. Only for logging purposes
            Returns:
                path to saved model
        '''
        X_t, X_v = X
        y_t, y_v = y        
        if (not self.sess_init):
            with self.graph.as_default():
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())
                self.sess_init = True
        
        
        with self.graph.as_default() as G, self.session as sess:
            saver = tf.train.Saver()
            model = self.model
            
            # Used to determine when to stop the training early
            valid_loss_summary = []
            iteration = 0

            print("Training Model: {}".format(log_string))
            train_writer = tf.summary.FileWriter('{}/train/{}'.format(self.log_dir, log_string), sess.graph)
            valid_writer = tf.summary.FileWriter('{}/train/{}'.format(self.log_dir, log_string))
            
            for e in range(epochs):
                # Record progress with each epoch
                train_loss = []
                train_acc = []
                val_acc = []
                val_loss = []
                
                """TEMP STUFF TO MEASURE VALIDATION LOSS"""
                n_batches_total = math.ceil(len(X_t)/batch_size)
                batches_before_validation = math.ceil(n_batches_total/measures_per_epoch)
                print(n_batches_total, batches_before_validation)
                """TEMP STUFF TO MEASURE VALIDATION LOSS"""

                with tqdm(total=len(X_t)) as pbar:
                    """TEMP STUFF TO MEASURE VALIDATION LOSS"""
                    v_loss_per_b = []
                    v_acc_per_b = []
                    t_loss_per_b = []
                    t_acc_per_b = []
                    """TEMP STUFF TO MEASURE VALIDATION LOSS"""
                    
                    for _, (X, y) in enumerate(get_batches(X_t, y_t, batch_size), 1):
                        feed = {model['inputs']: pad_sequences(X[:,0]),      #
                                model['inputs_pos']: pad_sequences(X[:,1]),
                                model['labels']: y,
                                model['train_mode']: True}

                        summary, loss, acc, state, _ = (
                            sess.run([model['merged'], 
                                      model['cost'], 
                                      model['accuracy'], 
                                      model['final_state'], 
                                      model['optimizer']], 
                                     feed_dict=feed))

                        # Record the loss and accuracy of each training batch
                        train_loss.append(loss)
                        train_acc.append(acc)

                        """TEMP STUFF TO MEASURE VALIDATION LOSS"""
                        t_loss_per_b.append(loss)
                        t_acc_per_b.append(acc)
                        """TEMP STUFF TO MEASURE VALIDATION LOSS"""
                        
                        # Record the progress of training
                        train_writer.add_summary(summary, iteration)

                        iteration += 1
                        pbar.update(len(X))                                
                            
                        if iteration % batches_before_validation == 0:
                            # Get validation loss and accuracy
                            avg_valid_loss, avg_valid_acc, summary = self.validate(X_v, y_v, batch_size)
                            
                            #Average over batches
                            t_loss_avg = np.mean(t_loss_per_b)
                            t_acc_avg = np.mean(t_acc_per_b)
                            
                            # Writing training and validation results to file
                            with open('{}/intermediate/{}'.format(self.log_dir, log_string), 'a') as int_file:
                                line = " ".join(map(str, [t_loss_avg, t_acc_avg, avg_valid_loss, avg_valid_acc]))
                                int_file.write(line)
                                int_file.write("\n")
                            
                            # Reset
                            v_loss_per_b, t_loss_per_b = [], []
                            v_acc_per_b, t_acc_per_b = [], []


                # Average the training loss and accuracy of each epoch
                avg_train_loss = np.mean(train_loss)
                avg_train_acc = np.mean(train_acc)
             
                valid_loss_summary.append(avg_valid_loss)

                # Record the validation data's progress
                valid_writer.add_summary(summary, iteration)

                # Print the progress of each epoch
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Train Loss: {:.3f}".format(avg_train_loss),
                      "Train Acc: {:.3f}".format(avg_train_acc),
                      "Valid Loss: {:.3f}".format(avg_valid_loss),
                      "Valid Acc: {:.3f}".format(avg_valid_acc))

                # Stop training if the validation loss does not decrease after 3 epochs
                if avg_valid_loss > min(valid_loss_summary):
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop_at:
                        break   

                # Reset stop_early if the validation loss finds a new low
                # Save a checkpoint of the model
                else:
                    print("New Record!")
                    stop_early = 0
                    checkpoint ="{}/checkpoints/{}.ckpt".format(self.log_dir, log_string)
                    saver.save(sess, checkpoint)
            self.load(checkpoint)
            return checkpoint        
            
    def validate(self, X_val, y_val, batch_size):
        '''Records accuracy and loss of given data'''
        sess = self.session
        model = self.model
        val_loss = []
        val_acc = []
        for X, y in get_batches(X_val, y_val, batch_size):
            feed = {model['inputs']: pad_sequences(X[:,0]),
                    model['inputs_pos']: pad_sequences(X[:,1]),
                    model['labels']: y,
                    model['train_mode']: False}

            summary, batch_loss, batch_acc = (
                sess.run([model['merged'],                         
                          model['cost'], 
                          model['accuracy']], 
                         feed_dict=feed))

            # Record the validation loss and accuracy of each epoch
            val_loss.append(batch_loss)
            val_acc.append(batch_acc)
        return np.mean(val_loss), np.mean(val_acc), summary

    def predict(self, X_test, batch_size):
        '''Get predictions from data'''
        assert(self.sess_init)
        
        all_preds = []
        model = self.model
        
        with self.session as sess:            
            for _, X in enumerate(get_test_batches(X_test, 
                                                   batch_size), 1):
                feed = {model['inputs']: pad_sequences(X[:,0]),
                        model['inputs_pos']: pad_sequences(X[:,1]),
                        model['train_mode']: False}
                predictions = sess.run(model['predictions'], feed_dict=feed)
                all_preds.extend(predictions)
        return np.array(all_preds)

    def load(self, path):
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            try:
                saver = tf.train.Saver()
                saver.restore(self.session, path)
                self.sess_init = True
            except Exception as e:
                print(e)
            finally:
                return self.sess_init
        
    def save(self, path):
        if self.sess_init:
            saver = tf.train.Saver()
            saver.save(self.session, path)


def unison_shuffle(A, B):
    assert(len(A) == len(B))
    p = np.random.permutation(len(A))
    return A[p], B[p]

def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    x, y = unison_shuffle(x, y)
    
    for ii in range(0, len(x), batch_size):      
        x_batch = x[ii:ii+batch_size]
        
        y_batch = y[ii:ii+batch_size]
        yield x_batch, y_batch[:, None]
        
def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]
