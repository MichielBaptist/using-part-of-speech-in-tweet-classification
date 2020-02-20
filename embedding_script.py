import utils.loader as loader
import tokenizers.custom_twitter_tokenizer as tokenizer
import utils.utils as ut
import numpy as np

root = ""
data_dir = root + "data/"

embedding_dir ="embeddings/"
tokenized_save_dir = "tokenized_data/"
embedding_save_dir = "embedded_data/"

tweets_fname = "train_pos.txt"

embedding_fname = "glove.twitter.27B.25d.txt"
embedding_name = "glove"
tokenize_name = "standard"

tweets_file_path = data_dir + tweets_fname
embedding_file_path = data_dir + embedding_dir + embedding_fname
tokenized_file_path = data_dir + tokenized_save_dir + tweets_fname + "_" + tokenize_name
embedded_file_path = data_dir + embedding_save_dir + tweets_fname + "_" + embedding_name
    
def process_data():

    word_to_emb, word_to_int, int_to_word = loader.load_embedding(embedding_file_path, 'glove')
    tweets = loader.load_tweets(tweets_file_path)[0]

    #Tokenize and save
    tokenz = tokenizer.tokenize_text(tweets)
    np.save(tokenized_file_path, tokenz)

    #Embed and save
    embedz = ut.embed_to_ints(tokenz, word_to_int)
    np.save(embedded_file_path, embedz)
    
def check():
    tokenz = np.load(tokenized_file_path + ".npy")
    embedz = np.load(embedded_file_path + ".npy")
    
    print(tokenz[0])
    print(embedz[0])
    
#process_data()
check()