# About
This project was in collaboration with [Richard Wigren](https://github.com/ricwi102), [Gorm Rødder](https://github.com/gefylen) and myself. The project was completed for the Computational Intelligence Lab course at ETH. In this project we explore deep learning models in the task of tweet classification. We employ a naive bayes baseline and experimented with different neural network based architectures. Notably, we incorporate part of speech in one of our models by essentially embedding part of speech information in a similar vein as word embeddings. For a brief overview of the work, consult the mini-paper included in the repo.

All the data (source and processed) is quite large (4 GB) and stored in Google drive folders.

# Setup guide:
1. Unzip the code.
2.  Install the dependenceis by running: 
```pip install -r requirements.txt ```
3. Run the setup script to install NLTK requirements:
```python setup.py```
4. Download the pre-trained models from the [Google Drive](https://drive.google.com/open?id=1oyyeUzKcx6i_e8edBRHIlVgDh3h5W6cF).
5. Unzip the pre-trained models in the ```log\``` folder.
6. Download the data zip from the [Google Drive](https://drive.google.com/open?id=1aYCgOq8cmfQl4_f05IlxzuOCEupqzoCo).
7. Extract the data zip in the ```data\``` folder.
8. Optional, but recommended: Download the pre-processed data from the Google Drive: [embedded data](https://drive.google.com/open?id=1_PlZ8Dcv8Rt76NPDjj1hVQ04JpVsFNnD), [POS data](https://drive.google.com/open?id=1PmtiU8fsCjOMgnIP1_HOIWb_VcN2q1V-) and [sentence embedding](https://drive.google.com/open?id=1hHOdNhnZn3EvgnsjaIGVe3HJJexnhKU4).
9. Optional, but recommended: Unzip the pre-processed data in the ```data\fully_processed\``` folder.

# Important:
1. Make sure to have a ```predictions\``` folder, this will be included standard.
2. If you want to save pre-processed data make sure to have a ```data\fully_processed``` folder. This folder will be included standard. If you wish to save some computation time be sure to do step 8 and 9.
# How to run
The model is run by calling main.py. All necessary arguments are in argparser.py and are listed below. For running our pre-trained models see the section 'How to run our pretrained models'
## Arguments
### data_dir
#### type: string
Directory of training and test data
### neg_tweets_file
#### type: string
Name of the file containing tweets with a negative tag
### pos_tweets_file
#### type: string
Name of the file containing tweets with a positive tag
### val_tweets_file
#### type: string
Name of the file containing tweets whose sentiment are to be determined
### test_train_split
#### type: float
A value between 0 and 1 that determines the percentage of training data used as training. The remaining will be used for validation of the model.
### embed_dir
#### type: string
Directory of embedding files
### use_predefined_embedding
#### type: bool
* True: loads predefined embeddigns
* False: doesn't load predefined embeddings
Used for models that use predefined word embeddings.
### embedding_name
#### type: string
File name of the embedding
### processed_data_dir
#### type: string
Directory of data that has been processed and saved.
### load_processed_data
#### type: bool
* True: load processed data. 
* False: process raw data
### save_processed_data
#### type: bool
* True: Saves processed data in processed_data_dir
* False: Skips this step
### preprocess_type
#### type: string
#### options: ['embed', 'POS', 'sentence', 'none']
Chooses the prepocessing to be done.
### classifier_type
#### type: string
#### options: ['naive_bayes', 'pos_tag_classifier', 'multi_layer_LSTM', 'deep_NN']
Which model/classifier to use for training, validation and/or prediction
### classifier_kwargs
#### type: Dict
Any additional arguments the classifier might need
### model_name
#### type: string
Chosen name of the model. Used for logging, saving and loading.
### load_model_tf
#### type: bool
* True: Loads a model of the same name if it exists. (e.g. load a trained model for making predictions)
* False: Skips this step
### train_model
#### type: bool
* True: Uses training data to train a model in its specified way
* False: Skips this step
### train_args
#### type: Dict
Any additional arguments that might be necessary for the <train> function
### do_validation
#### type: bool
* True: Tests accuracy on the validation-split of the data
* False: Skips this step
### do_prediction
#### type: bool
* True: Uses the model to make a prediction of the test data and saves it
* False: Skips this step
### prediction_dir
#### type: string
Name of the directory where predictions are saved.
  

## How to run our pretrained models
### POS model: Pos-tag + GloVe classifier:
The model concatinates a 200 dimension GloVe embeddings ([pretrained] (https://nlp.stanford.edu/projects/glove/) and a 50 dimension part of speech tagger as input to an LSTM with hidden size 128. The output of the last LSTM cell is then run through a two hidden layer neural network both of size 128 and all activation functions are sigmoids.  (LSTM -> 128 -> 128 -> 1)

It is run by calling main.py with the arguments set as shown below.
#### Arguments
neg_tweets_file: ```'train_neg_full.txt'``` \
pos_tweets_file: ```'train_pos_full.txt'``` \
val_tweets_file: ```'test_data.txt'``` 

use_predefined_embedding': ```True``` \
embedding_name: ```'twitter_200d.txt'``` 

load_processed_data: ```True``` \
preprocess_type: ```'POS'```

classifier_type: ```'pos_tag_classifier'``` \
classifier_kwargs: ```{'hidden_size': [128], 'n_layers': 1, 'output_size': [128], 
                    'output_layers': 2, 'learning_rate': 0.001, 
                    'keep_prob': [1.0, 1.0, 0.5], 'pos_emb_shape': (45, 50)}``` 
                    
model_name: ```'200d_50pos_1x128LSTM_2x128out_50drop'``` \
load_model_tf: ```True``` \
train_model: ```False```


do_validation: ```True``` \
do_prediction: ```True```

### Vanilla LSTM model: GloVe classifier:
The model uses pre-defined GloVe embeddings, [pretrained](https://nlp.stanford.edu/projects/glove/), as input to an LSTM. he output of the last LSTM cell is then run through a two hidden layer neural network both of size 128 and all activation functions are sigmoids.  (LSTM -> 128 -> 128 -> 1)

It is run by calling main.py with the arguments set as shown below. Exchange <embed_size> in the args model_name and embedding_name with the desired embedding size. Options are 25, 50, 100, 200
#### Arguments

neg_tweets_file: ```'train_neg_full.txt'``` \
pos_tweets_file: ```'train_pos_full.txt'``` \
val_tweets_file: ```'test_data.txt'``` 

use_predefined_embedding': ```True``` \
embedding_name: ```'twitter_<embed-size>d.txt'``` 

preprocess_type: ```'embed'``` \
load_processed_data: ```True```

classifier_type: ```'multi_layer_LSTM'``` \
classifier_kwargs: ```{'hidden_size': [128], 'n_layers': 1, 'output_size': [128], 
                    'output_layers': 2, 'learning_rate': 0.001, 
                    'keep_prob': [1.0, 1.0, 0.5]}``` 
                    
model_name: ```'<embed-size>d_1x128LSTM_2x128out_50drop'``` \
load_model_tf: ```True``` \
train_model:  ```False```


do_validation: ```True``` \
do_prediction: ```True```

## Running Baselines

### Deep Neural Net: Sentence embeddings
The model uses the Google universal sentence encoder (https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/2) as input to a Neural Network. 

It is run by calling main.py with the arguments set as shown below. It is strongly recommended to load the processed data for preservation of time. 
#### Arguments

neg_tweets_file: ```'train_neg_full.txt'``` \
pos_tweets_file: ```'train_pos_full.txt'``` \
val_tweets_file: ```'test_data.txt'``` 

use_predefined_embedding': ```False``` 

preprocess_type: ```'sentence'``` \
load_processed_data: ```True```

classifier_type: ```'deep_NN'``` \
classifier_kwargs: ```{'input_size': 512,
              'hidden_sizes': [256, 256,128,64] + [1], 
              'activation_fns': [tf.sigmoid]*4 + [tf.sigmoid], 
              'dropout': [0.5]*4}``` 
                    
model_name: ```'512sent_2x256_1x128_1x64_50drop'``` \
load_model_tf: ```True``` \
train_model: ```False```

do_validation: ```True``` \
do_prediction: ```True```

### Naive Bayes
#### Arguments

neg_tweets_file: ```'train_neg_full.txt'``` \
pos_tweets_file: ```'train_pos_full.txt'``` \
val_tweets_file: ```'test_data.txt'``` 

use_predefined_embedding': ```False``` 

preprocess_type: ```'none'``` \
load_processed_data: ```False``` \
save_processed_data = ```False```

classifier_type: ```'naive_bayes'``` \
classifier_kwargs: ```{}```
                    
model_name: ```'naive_bayes'``` 

do_validation: ```True``` \
do_prediction: ```True```

### Average Embedding
#### Arguments

neg_tweets_file: ```'train_neg_full.txt'``` \
pos_tweets_file: ```'train_pos_full.txt'``` \
val_tweets_file: ```'test_data.txt'``` 

use_predefined_embedding': ```True``` 

preprocess_type: ```'average_embedding'``` \
load_processed_data: ```False``` \
save_processed_data = ```False```

classifier_type: ```'average_embedding'``` \
classifier_kwargs: ```{}```
                    
model_name: ```'average_embedding'``` 

do_validation: ```True``` \
do_prediction: ```True```


## Training
Training is done by defining a model (for examples se our pretrained models) and setting ```train_model = True```

### Requirements
1. The directory ```./log/<model_type>/intermediate/  (e.g ./log/LSTM/intermediate/```) has to exist and enables the possiblilty to run validation multiple times per epoch for logging purposes.



