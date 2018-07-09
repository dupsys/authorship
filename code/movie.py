# Library definition
import re
import sys
import pandas as pd
import numpy as np
import itertools
import h5py
import keras
import keras.callbacks
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from AdamW import AdamW
from SGDW import SGDW
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, merge, concatenate, 
from keras.layers import Embedding, Conv1D, MaxPooling1D,AveragePooling1D, GlobalMaxPooling1D,
from keras.layers import TimeDistributed, Bidirectional, LSTM, Lambda
from keras.layers.normalization import BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.optimizers import SGD
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau
seed = 7
import numpy
numpy.random.seed(seed)

# Parameter definition
batch_size = 32
num_epochs = 120

maxlen = 512
max_sentences = 15

max_seq_length = 15

filter_length = [5,7,7]
nb_filter = [64,128,256]
pool_length = 2

lstm_h = 92
nb_classes = 60
w_l2 = 1e-4

# Data clearning
def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub("", str(s))

def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)    

def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 71

# Load data
path_dir = '/data/imbd62/'
train = pd.read_csv(path_dir + 'train.csv', index_col=0)
print("Training Sample: ", train.shape[0])

txt = ''
docs = []
sentences = []
labels = []
doc_length = 0

for cont, sentiment in zip(train.content, train.userId):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
    sentences = [sent.lower() for sent in sentences]
    docs.append(sentences)
    labels.append(sentiment)
    doc_length = doc_length + 1
    if (doc_length % 2500 == 0):
        print('zipping training set complete:', doc_length)

#  Vocabulary
num_sent = []
for doc in docs:
    num_sent.append(len(doc))
    for s in doc:
        txt += s
chars = set(txt)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# Doing One hot encoding for training sample and targets -- Training
'''
We bound the maximum length of the sentence to be 512 characters wheile the maximum number of
sentences in a document is bounded @ 15. We reverse the order of charctr putting the first character at the
end of 512 vector.
Example: we index X_train as (document, sentence, char)
'''
X_train = np.ones((len(docs), max_sentences, maxlen), dtype=np.int64) * -1
y_train = np.array(labels)

for i, doc in enumerate(docs):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X_train[i, j, (maxlen - 1 - t)] = char_indices[char]
                
print('Sample Training X_train:{}'.format(X_train[1200,2]))

print('encoding class values as integers')
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to one hot encoding
y_train = np_utils.to_categorical(encoded_Y)
print('training label:', y_train.shape)

# Testing Data
test = pd.read_csv(path_dir + 'test.csv', index_col=0)
print("Testing Sample: ", test.shape[0])

'''
Here we split the testing document into sentences
'''
txt = ''
docs1 = []
sentences1 = []
labels1 = []
doc_length = 0
for cont, sentiment in zip(test.content, test.userId):
    sentences1 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
    sentences1 = [sent.lower() for sent in sentences1]
    docs1.append(sentences1)
    labels1.append(sentiment)
    doc_length = doc_length + 1
    if (doc_length % 2500 == 0):
        print('zipping training set complete:', doc_length)
print('Success!')

print('Doing One hot encoding for testing sample and targets:')
X_test = np.ones((len(docs1), max_sentences, maxlen), dtype=np.int64) * -1
y_test = np.array(labels1)

for i, doc in enumerate(docs1):
    for j, sentence in enumerate(doc):
        if j < max_sentences:
            for t, char in enumerate(sentence[-maxlen:]):
                X_test[i, j, (maxlen-1-t)] = char_indices[char]
                
print('Sample Testing Data:{}'.format(X_test[120,2]))

print('encoding class values as integers')
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
# convert integers to one hot encoding
y_test = np_utils.to_categorical(encoded_Y)
print('testing label:', y_test.shape)


# Encoding Sentence with Convolutional Neural Network
def char_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       kernel_regularizer=regularizers.l2(w_l2),
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        block = BatchNormalization()(block)
        block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    block = GlobalMaxPooling1D()(block)
    block = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(w_l2))(block)
    return block

sequence_input = Input(shape=(maxlen,), dtype='int64')
embedded = Lambda(binarize, output_shape=binarize_outshape)(sequence_input)

block2 = char_block(embedded, (100, 200, 200), filter_length=(5, 3, 3), subsample=(1, 1, 1), pool_length=(2, 2, 2))
block3 = char_block(embedded, (200, 300, 300), filter_length=(7, 3, 3), subsample=(1, 1, 1), pool_length=(2, 2, 2))
sent_encode = concatenate([block2, block3], axis=-1)
sent_encode = Dropout(0.4)(sent_encode)
encoder = Model(inputs=sequence_input, outputs=sent_encode)
encoder.summary()


# Customerise matrix formulation for precision, recall and F1-score
def precision(y_true, y_pred):
    """Precision metric.
    Computes the precision, a metric for multi-label classification of
    how many samples were predicted correctly.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Computes the precision, a metric for multi-label classification of
    how many target (e.g., actual label) were predicted correctly.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric"""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric."""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision + recall + K.epsilon()))


# Document Encoder
'''
We feeds the sentence encoder output to a document encoder that is composed of a bidirectional lstm 
with fully connected layers at the top.

'''
def build_model_adam():
    sequence = Input(shape=(max_sentences, maxlen), dtype='int64')
    encoded = TimeDistributed(encoder)(sequence)
    forwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(encoded)
    backwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)
    merged = concatenate([forwards, backwards], axis=-1)
    
    output = Dropout(0.5)(merged)
    output = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(w_l2))(output)
    output = Dropout(0.5)(output)
    predictions = Dense(nb_classes, activation='softmax')(output)
    
    model_adam = Model(inputs=sequence, output=predictions)
    
    model_adam.compile(optimizer=AdamW(weight_decay=1.1),
                   loss='categorical_crossentropy', metrics=['accuracy', f1, recall, precision])
    model_adam.summary()
    return model_adam

def build_model_sgd():
    sequence = Input(shape=(max_sentences, maxlen), dtype='int64')
    encoded = TimeDistributed(encoder)(lstm_input)
    
    forwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(encoded)
    backwards = LSTM(lstm_h, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu', go_backwards=True)(encoded)
    
    merged = concatenate([forwards, backwards], axis=-1)
    
    output = Dropout(0.5)(merged)
    output = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(w_l2))(output)
    output = Dropout(0.5)(output)
    predictions = Dense(nb_classes, activation='softmax')(output)
    
    model_adam = Model(inputs=sequence, output=predictions)
    
    model_sgdw.compile(optimizer=SGDW(weight_decay=0.001, momentum=0.9),
                   loss='categorical_crossentropy', metrics=['accuracy', f1, recall, precision])
    model_sgdw.summary()
    return model_sgdw


# # Tensorboard with callback.
tensor_board = TensorBoard(log_dir='Outputfile', histogram_freq=0, write_graph=True, write_images=True)

# # K-fold Validation using Adam Optimization
k = 5
all_val_loss_histories = []
all_val_acc_histories = []
all_recall = []
all_precision = []
all_f1 = []
all_scores = []
num_validation_sample = len(X_train)//k

for i in range(k):
    print("-----------------------------")
    print('Processing fold #', i)
    print("-----------------------------")
    val_data = X_train[i * num_validation_sample: (i + 1) * num_validation_sample]
    val_lab  = y_train[i * num_validation_sample: (i + 1) * num_validation_sample]
    
    parttial_train_X_data = np.concatenate(
                [X_train[:i * num_validation_sample],
                X_train[(i + 1) * num_validation_sample:]], axis=0)
    
    parttial_train_X_label = np.concatenate(
                [y_train[:i * num_validation_sample],
                y_train[(i + 1) * num_validation_sample:]], axis=0)
    
    model_adam = build_model_adam()
    hist_adam = model_adam.fit(parttial_train_X_data, parttial_train_X_label, 
                               validation_data=(val_data,val_lab),
                               batch_size=batch_size, epochs=num_epochs, 
                               verbose=1, callbacks=[tensor_board])

val_acc__history = hist_adam.history['val_acc']
all_val_acc_histories.append(val_acc_history)
averagee_all_val_acc = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(num_epochs)]

# Visualisation
plt.plot(range(1, len(averagee_all_val_acc) + 1), averagee_all_val_acc)
plt.xlabel('Epochs')
plt.ylabel('Val_acc')
plt.show()
