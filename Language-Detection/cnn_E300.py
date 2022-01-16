import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout,GRU,Conv1D,GlobalMaxPooling1D
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("label_per_sentence_Hansard.csv")

df = df.drop(['label','id','number'],axis=1)
df.columns = ['TEXT','label']
df['TEXT'] = df['TEXT'].astype(str)



def vectorize_notes(col, MAX_NB_WORDS, verbose = True):
    """Takes a note column and encodes it into a series of integer
        Also returns the dictionnary mapping the word to the integer"""
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(col)
    data = tokenizer.texts_to_sequences(col)
    note_length =  [len(x) for x in data]
    vocab = tokenizer.word_index
    MAX_VOCAB = len(vocab)
    if verbose:
        print('Vocabulary size: %s' % MAX_VOCAB)
        print('Average note length: %s' % np.mean(note_length))
        print('Max note length: %s' % np.max(note_length))
    return data, vocab, MAX_VOCAB, tokenizer

def pad_notes(data, MAX_SEQ_LENGTH):
    data = pad_sequences(data, maxlen = MAX_SEQ_LENGTH)
    return data, data.shape[1]


def embedding_matrix(f_name, dictionary, EMBEDDING_DIM, verbose = True, sigma = None):
    """Takes a pre-trained embedding and adapts it to the dictionary at hand
        Words not found will be all-zeros in the matrix"""

    # Dictionary of words from the pre trained embedding
    pretrained_dict = {}
    try:
        with open(f_name, 'r') as f:
            for line in f:
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:])
                    pretrained_dict[word] = coefs
                except ValueError:
                    continue
    except:
        print("error")


    if sigma:
        pretrained_matrix = sigma * np.random.rand(len(dictionary) + 1, EMBEDDING_DIM)
    else:
        pretrained_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    
    # Substitution of default values by pretrained values when applicable
    for word, i in dictionary.items():
        try:
            vector = pretrained_dict.get(word)
            if vector is not None:
                pretrained_matrix[i] = vector
        except ValueError:
            continue

    if verbose:
        print('Vocabulary in notes:', len(dictionary))
        print('Vocabulary in original embedding:', len(pretrained_dict))
        inter = list( set(dictionary.keys()) & set(pretrained_dict.keys()) )
        print('Vocabulary intersection:', len(inter))

    return pretrained_matrix, pretrained_dict





def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size), random_state=random_state)
    return X_train, X_test, y_train, y_test


#preprocess notes
MAX_VOCAB = None 
MAX_SEQ_LENGTH = 250 
text = df.TEXT
data_vectorized, dictionary, MAX_VOCAB, tokenizer = vectorize_notes(text, MAX_VOCAB, verbose = True)
data, MAX_SEQ_LENGTH = pad_notes(data_vectorized, MAX_SEQ_LENGTH)

print("Final Vocabulary: %s" % MAX_VOCAB)
print("Final Max Sequence Length: %s" % MAX_SEQ_LENGTH)


EMBEDDING_DIM = 300 
EMBEDDING_MATRIX= []


EMBEDDING_LOC = 'cc.en.300.txt' 
EMBEDDING_MATRIX, embedding_dict = embedding_matrix(EMBEDDING_LOC, dictionary, EMBEDDING_DIM, verbose = True, sigma=True)


X = data
Y = pd.get_dummies(df['label']).values
print('Shape of label tensor:', Y.shape)


#split sets
X_train, X_test, y_train, y_test = train_val_test_split(
    X, Y, val_size=0.2, test_size=0.1, random_state=51)
print("Train: ", X_train.shape, y_train.shape)
print("Test: ", X_test.shape, y_test.shape)


del  data, X,  Y

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Sequential()
embedding=Embedding(MAX_VOCAB + 1, EMBEDDING_DIM,
          weights=[EMBEDDING_MATRIX], input_length=MAX_SEQ_LENGTH, trainable=False)

model.add(embedding)
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history=model.fit(X_train,y_train, callbacks=callback, 
                  batch_size=64, epochs=50, 
                  validation_split=0.1, 
                  verbose=2)
# class_weight=class_weights)

# creates a HDF5 file 'my_model.h5'
model.save('cnn-E300.h5')

import pickle 
# Save Tokenizer i.e. Vocabulary
with open('tokenizercnn_E300.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



from sklearn.metrics import classification_report

y_out = model.predict(X_test,batch_size=64)
y_pred = np.where(y_out > 0.5, 1, 0)


print(classification_report(y_test, y_pred))
print('accuracy %s', accuracy_score(y_pred, y_test))

