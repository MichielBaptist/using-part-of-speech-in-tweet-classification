import argparse


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--vocab_file', action='store', default='data/vocab.pkl', help='Location of the vocabulary file')
  parser.add_argument('--data_dir',        action='store', default='./data',               help='Location of the data directory, here the tweets and embeddings are kept')
  parser.add_argument('--neg_tweets_file', action='store', default='train_neg.txt', help='Name of the negative training tweets')
  parser.add_argument('--pos_tweets_file', action='store', default='train_pos.txt', help='Name of the positive training tweets')
  parser.add_argument('--val_tweets_file', action='store', default='test_data.txt', help='Name of the test tweets')
  parser.add_argument('--test_train_split',action='store', default=0.9,                  help='The amount of data should be used for training and testing')
  parser.add_argument('--use_full_dataset', action='store_true')
  parser.add_argument('--no_test', action='store_true')

  parser.add_argument('--embed_dir', default='data/embeddings', action='store')
  parser.add_argument('--use_predefined_embedding', default=True)
  parser.add_argument('--embedding_name', default='twitter_200d.txt')
  parser.add_argument('--calculate_embedding', nargs='?', const=None, default='calculate_glove_solution')
  parser.add_argument('--processed_data_dir', action='store', default='data/fully_processed')
  parser.add_argument('--load_processed_data', action='store', default=False)
  parser.add_argument('--save_processed_data', action='store', default=False)
 
  #options: 'embed', 'POS', 'sentence', 'none', 'average_embedding'
  parser.add_argument('--preprocess_type', action='store', default='average_embedding')

  #options: 'naive_bayes', 'pos_tag_classifier', 'multi_layer_LSTM', 'deep_NN'
  parser.add_argument('--classifier_type', action='store', default='average_embedding')
  parser.add_argument('--classifier_kwargs', action='store', default={})

  parser.add_argument('--model_name', action='store', default='25d_1x128LSTM_2x128out_50drop__Test')
  parser.add_argument('--load_model_tf', action='store', default=False)
  parser.add_argument('--train_model', action='store', default=False)
  parser.add_argument('--train_args', action='store', default={'epochs': 20, 'batch_size': 512, 'stop_at': 3, 'measures_per_epoch': 2})
  parser.add_argument('--do_validation', action='store', default=True)
  parser.add_argument('--do_prediction', action='store', default=True)
  parser.add_argument('--prediction_dir', action='store', default='./predictions/')

  return parser.parse_args()
