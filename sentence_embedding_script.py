import numpy as np
import utils.utils as utils
import utils.loader as loader
import math 
from embedders.google_sentence_emb import GoogleSentenceEmbedder

save_dir = "data/fully_processed"
load_dir = "data"
int_dir = "int_"
file_name = "test_data.txt"
start_at = 0
b_s = 100000

tweets_path = "{}/{}".format(load_dir, file_name)
int_path = "{}/{}_".format(int_dir, "part")
save_file = "{}/{}_sentence".format(save_dir, file_name)

embedder = GoogleSentenceEmbedder()
pos = loader.load_tweets(tweets_path)[0]

n_b = math.ceil(len(pos)/b_s)


print(n_b)
for i in range(start_at, n_b):
    print("from {} to {}".format(i*b_s,(i+1)*b_s))
    b = pos[i*b_s: (i+1)*b_s]
    b_p = embedder.embed_sentences(b)
    print("saving")
    np.save(int_path + "{}".format(i), b_p)
    
full = []
for i in range(n_b):
    part_path = int_dir + "/part_{}.npy".format(i)
    partial = np.load(part_path)
    print("Loaded part {}".format(i))
    full.extend(partial.tolist())
    
np.save(save_file, full)




