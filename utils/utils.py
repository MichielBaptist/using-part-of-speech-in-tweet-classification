import numpy as np
import nltk
import time
import tokenizers.custom_twitter_tokenizer

import pickle
import csv

import utils.loader as loader

from embedders.google_sentence_emb import GoogleSentenceEmbedder

    
def preprocess_data_collection(data_collection, word_to_int, embeddings,type="embed", add_pad=True):
    """
        gets a collection of data objects and preprocesses all of them. Data objects
        are a list of strings, representing the tweets.
    """
    print("Starting to preprocess data sets...")
        
    if type == "embed":
        #One by One possible
        return [preprocess_data_standard(data, word_to_int, add_pad) for data in data_collection]
    elif type == "POS":
        #Needs all data at once
        train = data_collection[0:2] # For now assum this is the case
        test = data_collection[2:]   # For now assume this is the case
        train, test = preprocess_data_pos(train, test, word_to_int, add_pad)
        return train + test
    elif type == "sentence":
        return preprocess_data_sentence(data_collection)
    elif type == "average_embedding":
        return [preprocess_data_average_embedding(data, word_to_int, embeddings) for data in data_collection]
    elif type == "none":
        return data_collection
    else:
        return None

def preprocess_data_average_embedding(data, word_to_int, int_to_emb):
    # data:     [n_samples] List of tweets in string format
    
    # Start by tokenizeing the data
    all_tokenz = tokenizers.custom_twitter_tokenizer.tokenize_text(data)
    
    # Embed the tokenz to ints
    embedded_tweets = embed_to_ints(all_tokenz, word_to_int)
    
    # Average the word embeddings
    return np.array([average_embedding(embedding, int_to_emb) for embedding in embedded_tweets])
    
def average_embedding(embedding, int_to_emb):
    # Get embedding for each index
    embedding_vectors = np.array([int_to_emb[emb] for emb in embedding])
    # Average across words/tokens    
    averaged = np.mean(embedding_vectors, axis = 0) 
    
    return averaged.tolist()    

def preprocess_data_sentence(collections):
    embedder = GoogleSentenceEmbedder()
    return [embedder.embed_sentences(collection) for collection in collections]
    
def preprocess_data_standard(data, word_to_int, add_pad = True):
    #data:          [n_samples] List of tweets in string format
    #word_to_int:   {voc:int}   dictionary of word to int
    
    # Start by tokenizeing the data
    all_tokenz = tokenizers.custom_twitter_tokenizer.tokenize_text(data)
    
    # Embed the tokenz to ints
    embedded_tweets = embed_to_ints(all_tokenz, word_to_int)

    
    return embedded_tweets
    
def preprocess_data_pos(train, test, word_to_int, add_pad = True):
    #train:      collection of tweet lists [n_collections, n_samples] of strings
    #test:       collection of tweet lists [n_collections, n_samples] os strings
    #word_to_int:   {voc:int}   dictionary of word to int
    
    # Tokenize data and tag train data
    tokens_per_collection_train = [tokenizers.custom_twitter_tokenizer.tokenize_text(coll) for coll in train]
    tokens_per_collection_test  = [tokenizers.custom_twitter_tokenizer.tokenize_text(coll) for coll in test]
    tags_per_collection_train   = [pos_tag_text(collection) for collection in tokens_per_collection_train]
    tags_per_collection_test    = [pos_tag_text(collection) for collection in tokens_per_collection_test]
    
    # Pool all training samples together and create the tag to int mapping
    tags_pooled_train = [sample for collection in tags_per_collection_train for sample in collection]
    _, tag_to_int = gen_tag_embedding(tags_pooled_train)
    
    # Embed words and tags of all collections
    wembedding_per_collection_train = [embed_to_ints(token_collection,word_to_int) for token_collection in tokens_per_collection_train]
    wembedding_per_collection_test  = [embed_to_ints(token_collection,word_to_int) for token_collection in tokens_per_collection_test]
    tembedding_per_collection_train  = [embed_to_ints(tag_collection, tag_to_int) for tag_collection in tags_per_collection_train]
    tembedding_per_collection_test   = [embed_to_ints(tag_collection, tag_to_int) for tag_collection in tags_per_collection_test]
    
    # emb: [n samples, None]
    # tag: [n samples, None]
    collection_train = []
    for (emb, tag) in list(zip(wembedding_per_collection_train, tembedding_per_collection_train)):
        samples = []
        for (e, t) in list(zip(emb, tag)):            
            samples.append( np.stack( (e, t) ).tolist() )
        print("train_sample: ", np.array(samples).shape )
        collection_train.append( np.array(samples) )

    collection_test = []
    for (emb, tag) in list(zip(wembedding_per_collection_test, tembedding_per_collection_test)):
        samples = []
        for (e, t) in list(zip(emb, tag)):            
            samples.append( np.stack( (e, t) ).tolist() )
        print("test_sample: ", np.array(samples).shape )
        collection_test.append( np.array(samples) )
        
    print("train: ", np.array(collection_train).shape )
    print("test: ", np.array(collection_test).shape )
    return collection_train, collection_test
    
def check_consistency(tag_list, word_list):
    """
        A help method to check wether two lists of lists are the same
            -length 
            -Have the same sublist lengths
    """
    inc = False

    if len(tag_list) != len(word_list):
        print("Inconsistent sample size")
        inc = True
        
    for (t_l, w_l) in list(zip(tag_list, word_list)):
        if len(t_l) != len(w_l):
            print("Inconsistent sample found!")
            inc = True

    return inc
    
def gen_tag_embedding(tags, embed_size = 15):
    # tags list of tag lists [n_samples, None]
    
    print("--Starting to make tag embedding --")
    
    all_tags = [tag for list in tags for tag in list]
    unsorted_uniques, unsorted_counts = np.unique(all_tags, return_counts = True)
    
    
    # Sort tokens by frequency
    sorted_unique_tokens = list(zip(unsorted_uniques, unsorted_counts))
    sorted_unique_tokens.sort(key=lambda t: t[1], reverse=True)
    
    tag_to_int = {}
    int_to_emb = []
        
    tag_to_int['<pad>'] = 0
    int_to_emb.append(np.zeros(embed_size).tolist())
    
    for i, (tag, count) in enumerate(sorted_unique_tokens):
        tag_to_int[tag] = i +1
        int_to_emb.append(random_embed(embed_size))
    
    tag_to_int['<unk>'] = len(tag_to_int) + 1
    int_to_emb.append(np.zeros(embed_size).tolist())
    
    #print some info
    print("--Done making tag embedding--")
    print("---->counted {} unique tags".format(len(unsorted_uniques)))
    
    return int_to_emb, tag_to_int
    
def random_embed(size):
    return np.random.rand(size).tolist()

def pos_tag_text(tokenized_text_list):
    # Tokenized_text_list: a list of lists containing tokenz [n_samples, None] 
    start = time.time()
    print("-- Starting to pos tag --")
    
    it = 0
    pos_tagged = []
    for tokenz in tokenized_text_list:
        pos_tagged.append(pos_tag_tokenz(tokenz))
        it += 1
        if it % 1000 == 0:
            print(it)
        
    print("-- Done pos tagging -- {}".format(start - time.time()))
    
    return pos_tagged
    
def pos_tag_tokenz(tokenz):
    # tokenz: List of tokens to tag
    
    tagged = nltk.pos_tag(tokenz)
    tagged_array = [tag for (_, tag) in tagged]
    
    return tagged_array

def print_classification_statistics(y_pred, y_true):
    tp, fn, tn, fp = calculate_classification_stats(y_pred, y_true)
    recall = tp/(tp + fn)
    precision = tp/(tp+fp)
    accuracy = (tn + tp)/(tn + tp+fn+fp)
    tn_rate = tn /(tn + fn)
    print(" TP:{0}\n TN:{1} \n FP:{2} \n FN:{3}".format(tp,tn,fp,fn))
    print("--Precision: {0} \n--Recall: {1} \n--Accuracy:{2} \n--True negative rate:{3}".format(recall, precision, accuracy, tn_rate))
    
def calculate_classification_stats(prediction, actual, f_lab = 0, t_lab = 1):
    if len(prediction) != len(actual):
        raise ValueError('Prediction must have thhe same length as actual results!')
    
    res = [find_case(x, f_lab, t_lab) for x in zip(prediction, actual)]
    
    counts = {'tp':0, 'fn':0, 'tn':0, 'fp':0}
    for x in res:
        counts[x] +=1
    
    return counts['tp'],counts['fn'],counts['tn'],counts['fp']
    
def find_case(tuple, f_lab, t_lab):
    pr, ac = tuple
    if ac == f_lab and pr == ac:
        return 'tn'
    elif ac == f_lab and pr != ac:
        return 'fp'
    elif ac == t_lab and pr == ac:
        return 'tp'
    elif ac == t_lab and pr != ac:
        return 'fn'
        
def write_to_log(str):
    with open("log/log.txt", "a") as file:
        file.write(str)
        
def reset_log():
    open("log/log.txt", 'w').close()
    
def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def inverse_map(mp):
    return {v: k for k, v in mp.items()}
    
def convert_to_int(word, word_to_int):
    global total, unconverted
    total += 1
    try:
        return word_to_int[word]
    except:
        if is_number(word):
            return word_to_int['<number>']
        else:        
            unconverted += 1
            return word_to_int['<unk>']

total = 0
unconverted = 0

def embed_to_ints(X, word_to_int):
    global total, unconverted
    start = time.time()
    print("--starting to embed--")
    embedded = []
    for sentence in X:
        embedded_s = [convert_to_int(w, word_to_int) for w in sentence]
        embedded.append(embedded_s)
    end = time.time()
    print("--done embedding-- {0}".format(end-start))
    print("---->Words not embedded: {0}%\n".format(unconverted/total))
    return np.array(embedded)

def format_embeddings(e_dir, e_name, add_pad=True):
    ''' Formats embeddings as a matrix where each row corresponds to a word embedding
        and a dictionary containing word to matrix-index mappings
            Args:
                e_dir: Directory of the embedding
                e_name: File name of the embedding
                add_pad: Bool, adds a row of zeros at the beginning of the embeddings
                         and adds key-value pair (<pad>, 0) to the mapping if true
            Returns:
                Mapping from word to embedding,
                Embeddings in a suitable format                
            Note:
                File has to be of format:
                    <word1> <e1_1> <e1_2> ...
                    <word2> <e2_1> <e2_2> ...
                And words have to have the same embedding length    
    '''
    start = time.time()    
    e_path = './{}/{}'.format(e_dir, e_name)
    
    with open(e_path, encoding='utf8') as f:
        first_line = f.readline()
    first_line = first_line.split(" ")
    v_size = len(first_line)
    
    np_str = np.dtype(str)
    cols = range(1, v_size)
    
    print("--Loading: {} --".format(e_name))
    words = np.loadtxt(e_path, dtype = np_str, usecols = 0, delimiter = " ", encoding = 'utf-8')
    e_data = np.loadtxt(e_path, usecols = cols, delimiter = " ", encoding = 'utf-8')
    print("--File loaded. Creating dictionary--")
    
    e_dict = {}
    start = 0
    if add_pad:
        n, m = e_data.shape
        first_row = np.zeros((1, m))

        e_data = np.concatenate([first_row, e_data])
        e_dict['<pad>'] = 0
        start += 1    
    
    for i, w in enumerate(words, start):
        e_dict[w] = i
    print("--Dictionary completed--")
    end = time.time()
    print('--Done formating embeddings-- {0}\n'.format(end-start))
    return e_dict, e_data.astype(np.float32)