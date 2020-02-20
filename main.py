from pprint import pprint
import pickle
import re
import nltk
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import argparser
import utils.loader as loader
from classifiers.naive_bayes_wrap import naive_bayes_wrap
from classifiers.rnn_classifiers import multi_layer_LSTM
from classifiers.rnn_classifiers import pos_tag_classifier
from classifiers.nn_classifiers import deep_NN
from classifiers.average_embedding import average_embedding_classifier
from utils import utils

models = {'naive_bayes': naive_bayes_wrap,
          'pos_tag_classifier': pos_tag_classifier,          
          'multi_layer_LSTM': multi_layer_LSTM,
          'deep_NN': deep_NN,
          'average_embedding': average_embedding_classifier}

is_tf_model = {'naive_bayes': False,
               'pos_tag_classifier': True,          
               'multi_layer_LSTM': True,
               'deep_NN': True,
               'average_embedding': False}



def main():

  np.random.seed(42)
  
  # Get the given arguments and initialize the utils and loader
  args = argparser.parse_args()

  # Initialization of variables
  model_extra_kwargs = {} 
  prediction_kwargs = {}
  word_to_int = None

  if args.use_predefined_embedding:
    #Load embeddings
    embedding_path = "{}/{}".format(args.embed_dir, args.embedding_name)
    word_to_int, embeddings = utils.format_embeddings(args.embed_dir, args.embedding_name)
    model_extra_kwargs['predef_embed'] = embeddings
    

  if not args.load_processed_data:
    # Load and preprocess tweets
    tweet_sets = loader.load_all_tweet_sets(args)    
    tweet_sets = utils.preprocess_data_collection(tweet_sets, word_to_int, embeddings,
                                  type=args.preprocess_type, add_pad=False)
  else:
    tweet_sets = loader.load_all_tweet_sets_fully_processed(args)

  if args.save_processed_data:
    # Save processed data
    loader.save_all_tweet_sets_fully_processed(args, tweet_sets)

  # Split tweets
  pos_tweets, neg_tweets, test_tweets = tweet_sets  

  pos_labels = loader.get_labels(tweet_sets[0],  1)
  neg_labels = loader.get_labels(tweet_sets[1],  0)
  
  print(pos_tweets.shape)
  print(neg_tweets.shape)

  # Combine positive and negative tweets into one big set!
  all_tweets, all_labels = loader.join_and_shuffle(
                                   [pos_tweets, neg_tweets]
                                  ,[pos_labels, neg_labels])
  all_tweets = np.array(all_tweets)
  all_labels = np.array(all_labels)

  # Split data into training and validation sets
  train_tweets_X, test_tweets_X, train_tweets_Y, test_tweets_Y = train_test_split(all_tweets, all_labels, train_size = args.test_train_split)
  print("using {0} tweets for training".format(len(train_tweets_X)))
  print("using {0} tweets for testing".format(len(test_tweets_X)))

  all_tweets = None
  tweet_sets = None
  gc.collect()

  if is_tf_model[args.classifier_type]:
    # Initialize and build Tensorflow model  
    classifier = models[args.classifier_type]()
    classifier.build(**model_extra_kwargs, **args.classifier_kwargs)

    prediction_kwargs['batch_size'] = 256

    if args.load_model_tf:    
      classifier.load('{}/checkpoints/{}.ckpt'.format(classifier.log_dir, args.model_name))
  
    if args.train_model:
      print("--Fitting the model--")
      data = (train_tweets_X, test_tweets_X)
      labels = (train_tweets_Y, test_tweets_Y)

      classifier.train(data, labels, log_string=args.model_name, **args.train_args)

  else:
    if args.classifier_type == 'naive_bayes':
      classifier = models['naive_bayes'](**args.classifier_kwargs)
      classifier.fit(train_tweets_X, train_tweets_Y)
    elif args.classifier_type == 'average_embedding':
      classifier = models['average_embedding'](**args.classifier_kwargs)
      classifier.fit(train_tweets_X, train_tweets_Y)

  if args.do_validation:
    # Test accuracy on Validation data
    pred_val = classifier.predict(test_tweets_X, **prediction_kwargs)
    pred_val = np.round(pred_val)
    
    utils.print_classification_statistics(pred_val, test_tweets_Y)
    if is_tf_model[args.classifier_type]:
      classifier.load('{}/checkpoints/{}.ckpt'.format(classifier.log_dir, args.model_name))

  if args.do_prediction:
    print("--Predicting with the model--")
    pred_test = classifier.predict(test_tweets, **prediction_kwargs)
  
    loader.save_prediction(pred_test, args.prediction_dir, args.model_name)



    

main()
