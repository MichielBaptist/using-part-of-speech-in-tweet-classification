import nltk
import re
import time
from utils import utils as ut


#Temporary before clean-up
def tokenize_text(X):
    start = time.time()
    print("--Starting to tokenize--")
    # Which tokens should be preserved? e.g. <url> should stay <url>
    preserve_map = {'<user>': "_USER_",
                    '<url>': "_URL_"}
    inversed_map = ut.inverse_map(preserve_map)
    
    # This tokenizer will map "<url>" --> ["<", "url", ">"]
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    
    for i in range(len(X)):
        tokenized_sentence = X[i]
        tokenized_sentence = replace_tokens(tokenized_sentence, preserve_map)
        tokenized_sentence = tokenizer.tokenize(tokenized_sentence)
        tokenized_sentence = [replace_tokens(token, inversed_map) for token in tokenized_sentence]
                
        X[i] = tokenized_sentence
    end = time.time()
    
    print("--End of tokenizeing-- {0} \n".format(end-start))
    
    return X
    
def replace_tokens(string, preserve_map):
    for preserve_token in preserve_map:
        string = re.sub(preserve_token, preserve_map[preserve_token], string)
    return string
