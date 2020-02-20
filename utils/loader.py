import numpy as np
import pickle
import time

from pathlib import Path
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import csv

def load_all_tweet_sets_fully_processed(args):
    
    test_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.val_tweets_file, args.preprocess_type)
    pos_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.pos_tweets_file,  args.preprocess_type)
    neg_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.neg_tweets_file,  args.preprocess_type)
    
    return np.load(pos_file_path), np.load(neg_file_path), np.load(test_file_path)
    
def save_all_tweet_sets_fully_processed(args, data):
    
    test_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.val_tweets_file, args.preprocess_type)
    pos_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.pos_tweets_file,  args.preprocess_type)
    neg_file_path = "{0}/{1}_{2}.npy".format(args.processed_data_dir, args.neg_tweets_file,  args.preprocess_type)
    
    np.save(pos_file_path, data[0])
    np.save(neg_file_path, data[1])
    np.save(test_file_path, data[2])
    
def load_all_tweet_sets(args):
      
  #Load all tweet files
  test_file_path = "{0}/{1}".format(args.data_dir, args.val_tweets_file)
  pos_file_path = "{0}/{1}".format(args.data_dir, args.pos_tweets_file)
  neg_file_path = "{0}/{1}".format(args.data_dir, args.neg_tweets_file)
  
  tweets_per_file = load_tweets(pos_file_path, neg_file_path, test_file_path)
  pos_tweets = tweets_per_file[0]
  neg_tweets = tweets_per_file[1]
  test_tweets= tweets_per_file[2]
  
  return pos_tweets, neg_tweets, test_tweets
  
def generate_indices(tweets):
    return np.arange(1, len(tweets)+1)
    
def load_vocabulary(vocab_file):
    return pickle.load(open(vocab_file, 'rb'))

def get_labels(tweets, label):
    return [label]*len(tweets)

def join_and_shuffle(tweet_collections, tweet_labels):
    if len(tweet_collections) != len(tweet_labels):
        raise ValueError("Given tweet collection and labels are not the same size! Check join_and_shuffle(.) call")
        
    for i in range(len(tweet_collections)):
        if len(tweet_collections[i]) != len(tweet_labels[i]):
            raise ValueError('One of the collections has mismatched label amount')
    
    print("--Starting to shuffle--")
    all_tw = []
    all_la = []
    for i, list in enumerate(tweet_collections):
        all_tw.extend(list)
        all_la.extend(tweet_labels[i])
    
    shuffled_tw, shuffled_la = shuffle(all_tw, all_la)
    print("--Done shuffling--\n")
    return shuffled_tw, shuffled_la

# Help function to clean some text if you want
def clean_lines(lines, fns=[lambda x:x]):
    print("cleaning lines")
    lines_c = []
    for i, line in enumerate(lines):
        line_c = line
        for fn in fns:
            line_c = fn(line_c)
        lines_c.append(line_c)
    print("Done cleaning {0} lines".format(len(lines_c)))
    return lines_c

#Loads a collection of files of tweets in raw form (lines)
def load_tweets(*tweet_files):
    start = time.time()
    print("--Starting to load tweets--")
    tweets_per_file = []
    for tweet_file in tweet_files:
        tweets = open(tweet_file, encoding="utf8")
        tweets_per_file.append(tweets.readlines())
        
    end = time.time()
    print("--End loading tweets-- {0} \n".format(end-start))
    return tweets_per_file
    
def preprocess(vocab, pos_data, neg_data, test_data):
    return vocab, pos_data, neg_data, test_data

def load_glove(file_path):
    # Measure time and word statistics
    start = time.time()
    words_loaded = 0    
    print ("--Loading embedding: {0} ---".format(file_path))
    
    # Open and loop lines, 1 line = 1 mapping of word to vector
    f = open(file_path,'r', encoding='utf8')
    model = {}
    int_to_emb = []
    word_to_int = {}
    for i, line in enumerate(f):
        
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        
        if word in word_to_int:
            print("Word {0} has been found multiple times".format(word))
        
        #word_to_embed
        model[word] = embedding
        #word to int i.e. position in matrix
        word_to_int[word] = i
        #int to emb i.e. matrix
        int_to_emb.append(embedding)
        
        words_loaded += 1
    
    end = time.time()
    print("--Done loading embedding-- {0}".format(end-start))
    print("----> {} words were loaded\n".format(words_loaded))
    #print("----> {} dimension of the embeddings\n".format(np.array(int_to_emb).shape[1]))
    
    return model, word_to_int, int_to_emb
    
def load_embedding(file_path, format=None):
    if format=='glove':
        return load_glove(file_path)
    else:
        return None

def load_data(d_dir, pos_name, neg_name):
    d_pos = np.load(d_dir + pos_name)
    d_neg = np.load(d_dir + neg_name)
    
    n, = d_pos.shape
    m, = d_neg.shape   
    
    l_pos = np.ones(n)
    l_neg = np.zeros(m)
    
    return np.append(d_pos, d_neg), np.append(l_pos, l_neg)

def save_embeddings(e_dict, e_data, e_dir, e_name):
    np.save(e_dir + e_name + '_dict.npy', e_dict) 
    np.save(e_dir + e_name + '_data.npy', e_data)
    
def load_embeddings(e_dir, e_name):
    e_dict = np.load(e_dir + e_name + '_dict.npy').item()
    e_data = np.load(e_dir + e_name + '_data.npy')    
    return e_dict, e_data.astype(np.float32)


def save_prediction(predictions, f_dir, f_name):
    ''' Saves prediction to a suitable format for submitions to Kaggle
            Args:
                predictions: nd-array of any shape with values in [0, 1], (float or int)
                f_dir: Name of directory to place prediction
                f_name: Name of prediction file
            Note:
                Predictions are made in {-1, 1}, therefore values of 0 (after rounding) 
                will be set to negative. 
    '''
    preds = np.round(predictions, decimals=0).reshape(-1).astype(int)
    preds[preds == 0] = -1

    with open(f_dir + f_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for row in enumerate(preds, 1):
            writer.writerow(row)
