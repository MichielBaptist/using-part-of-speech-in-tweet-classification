import tensorflow_hub as hub
import tensorflow as tf
import math
import numpy as np
import time


class GoogleSentenceEmbedder:
    def __init__(self):
        print("--Starting to download/load sentence embedding model!--")
        session = tf.Session()
        session.run([tf.tables_initializer(), tf.global_variables_initializer()])
            
        adr = module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
        self.sentence_embedder = hub.Module(adr)
        
        print("--Done downloading/loading sentence embedding model!--")
        
    def embed_sentences(self, sentences):
        start = time.time()
        print("--Starting to embed sentences--") 
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            
            full_embeddings = []
            for batch in self.get_batches(sentences):                
                embeddings = session.run(self.sentence_embedder(batch))
                full_embeddings.extend(embeddings)
                print("done")
                
        print("--Done embedding sentences--{}".format(time.time() - start)) 
            
        return full_embeddings
        
    def get_batches(self, sentences, batch_sz = 20000):
        n_batchs = math.ceil(len(sentences)/batch_sz)
        for i in range(n_batchs):
            yield sentences[i*batch_sz:(i+1)*batch_sz]