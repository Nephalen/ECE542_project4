'''
Reference:
1. https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py

'''

import numpy as np
import collections
import os
import keras.backend as K
import tensorflow as tf

from keras.layers import Dense, LSTM, Embedding, Dropout, CuDNNLSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model

def decode(data):
    decoded = np.zeros(data.shape[0]*data.shape[1])
    ct = 0
    for d in np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2])):
        decoded[ct] = np.argmax(d)
        ct += 1
    return decoded

def perplexity(y_true, y_pred):
    entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    result = K.pow(2., entropy)
    return result

def read_words(path):
    with open(path, "r") as f:
        return f.read().replace("\n", "<eos>").split()
    
def build_vocab(path):
    data = read_words(path)
    
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))

    return word_to_id, id_to_word

def file_to_word_ids(path, word_to_id):
    data = read_words(path)
    return np.array([word_to_id[word] for word in data if word in word_to_id])

def evaluate_model(model_path, test_data, vocab_size, time_step = 30):
    old_model = load_model(model_path, custom_objects = {'perplexity': perplexity})
    
    model = Sequential()
    model.add(Embedding(vocab_size, 120, batch_input_shape = (1, time_step)))
    model.add(CuDNNLSTM(256, return_sequences = True, stateful = True))
    #model.add(CuDNNLSTM(256, batch_input_shape = (1, time_step, 1), return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation = "softmax"))
    
    model.set_weights(old_model.get_weights())
    
    adam_opt = Adam(lr = 3e-4)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy', perplexity])
    
    valid_x, valid_y = test_data[0:-1], test_data[1:]
    trun_index = valid_x.shape[0]//time_step*time_step
    valid_x = valid_x[0:trun_index]
    valid_y = valid_y[0:trun_index]
    valid_x = valid_x.reshape(valid_x.shape[0]//time_step, time_step)
    valid_y = valid_y.reshape(valid_y.shape[0]//time_step, time_step, 1)
    
    result = model.evaluate(valid_x, valid_y, batch_size = 1)
    print(result)

def fit_model(train_data, valid_data, vocab_size, train_portion, time_step = 30, batch_size = 30):
    train_data = train_data[0:int(train_data.shape[0]*train_portion)]
    train_x, train_y = train_data[0:-1], train_data[1:]
    
    trunc_step = time_step*batch_size
    
    trun_index = train_x.shape[0]//trunc_step*trunc_step
    train_x = train_x[0:trun_index]
    train_y = train_y[0:trun_index]
    
    #split subsequence and reshape
    train_x = np.reshape(train_x, (1, len(train_x)))
    train_y = np.reshape(train_y, (1, len(train_y)))
    
    train_x = np.split(train_x, batch_size, axis = 1)
    train_y = np.split(train_y, batch_size, axis = 1)
    
    for i in range(batch_size):
        train_x[i] = np.split(train_x[i], train_x[i].shape[1]//time_step, axis = 1)
        train_y[i] = np.split(train_y[i], train_y[i].shape[1]//time_step, axis = 1)
    
    train_x = np.concatenate(train_x, axis = 1)
    train_y = np.concatenate(train_y, axis = 1)
    
    train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1], train_x.shape[2])
    #train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1], train_x.shape[2], 1)
    #train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1], train_y.shape[2], 1)
    train_y = train_y.reshape(train_y.shape[0]*train_y.shape[1], train_y.shape[2])
    train_y = np.array([train_y[i][len(train_y[i])-1] for i in range(train_y.shape[0])])
    train_y = train_y.reshape(len(train_y), 1)
    
    #validation data prepare
    valid_x, valid_y = valid_data[0:-1], valid_data[1:]
    trun_index = valid_x.shape[0]//trunc_step*trunc_step
    valid_x = valid_x[0:trun_index]
    valid_y = valid_y[0:trun_index]
    
    valid_x = np.reshape(valid_x, (1, len(valid_x)))
    valid_y = np.reshape(valid_y, (1, len(valid_y)))
    
    valid_x = np.split(valid_x, batch_size, axis = 1)
    valid_y = np.split(valid_y, batch_size, axis = 1)
    
    for i in range(batch_size):
        valid_x[i] = np.split(valid_x[i], valid_x[i].shape[1]//time_step, axis = 1)
        valid_y[i] = np.split(valid_y[i], valid_y[i].shape[1]//time_step, axis = 1)
    
    valid_x = np.concatenate(valid_x, axis = 1)
    valid_y = np.concatenate(valid_y, axis = 1)
    
    valid_x = valid_x.reshape(valid_x.shape[0]*valid_x.shape[1], valid_x.shape[2])
    #valid_x = valid_x.reshape(valid_x.shape[0]*valid_x.shape[1], valid_x.shape[2], 1)
    #valid_y = valid_y.reshape(valid_y.shape[0]*valid_y.shape[1], valid_y.shape[2], 1)
    valid_y = valid_y.reshape(valid_y.shape[0]*valid_y.shape[1], valid_y.shape[2])
    valid_y = np.array([valid_y[i][len(valid_y[i])-1] for i in range(valid_y.shape[0])])
    valid_y = valid_y.reshape(len(valid_y), 1)
    
    print(train_x.shape)
    print(valid_x.shape)
    print(train_y.shape)
    print(valid_y.shape)
    
    model = Sequential()
    model.add(Embedding(vocab_size, 120, batch_input_shape = (batch_size, train_x.shape[1])))
    model.add(CuDNNLSTM(256, stateful = True))
    #model.add(CuDNNLSTM(256, batch_input_shape = (batch_size, train_x.shape[1], 1), return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation = "softmax"))
    
    csv_logger = CSVLogger("ptb_words.csv", append=True, separator=',')
    checkpoint = ModelCheckpoint('ptb_words_best.h5', monitor = 'loss', verbose=1, save_best_only = True, mode='min')
    adam_opt = Adam(lr = 0.001)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy', perplexity])
    
    model.summary()
    
    for i in range(100):
        model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=batch_size, epochs=1, verbose=1, shuffle=False, callbacks=[csv_logger, checkpoint])
        model.reset_states()
        
    #model.save('ptb_words_last.h5')

def text_generation(model_path, test_data, id_to_word, vocab_size, time_step = 30):
    old_model = load_model(model_path, custom_objects = {'perplexity': perplexity})
    
    model = Sequential()
    model.add(Embedding(vocab_size, 120, batch_input_shape = (1, time_step)))
    model.add(CuDNNLSTM(256, stateful = True))
    #model.add(CuDNNLSTM(256, batch_input_shape = (1, time_step, 1), return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation = "softmax"))
    
    model.set_weights(old_model.get_weights())
    
    adam_opt = Adam(lr = 0.001)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy', perplexity])
    
    model.reset_states()
    
    trun_index = test_data.shape[0]//time_step*time_step
    test_data = test_data[0:trun_index]
    test_data = test_data.reshape(test_data.shape[0]//time_step, time_step)
    
    seed_index = np.random.randint(0, len(test_data)-1)
    result = test_data[seed_index]
    pattern = test_data[seed_index]
    
    f = open("ptb_word_generated.txt", 'w')
    result_string = [id_to_word[i] for i in result]
    result_string = ' '.join(result_string)
    f.write(result_string)
    f.write('\n')
    
    for i in range(200):
        x = np.reshape(pattern, (1, len(pattern)))
        #x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x)
        #print(prediction[0][prediction.shape[1]-1])
        #index = np.argmax(prediction[0][prediction.shape[1]-1])
        index = np.argmax(prediction)
        result = np.append(result, index)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]
        model.reset_states()
    
    result_string = [id_to_word[i] for i in result]
    result_string = ' '.join(result_string)
    
    f.write(result_string)
    f.close()
    
if __name__ == "__main__":
    #print(os.getcwd())
    
    data_path = "data\simple-examples\data"
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    word_to_id, id_to_word = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    
    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)
    
    #fit_model(train_data, valid_data, len(word_to_id), 1)
    evaluate_model("ptb_words.h5", valid_data, len(word_to_id))
    #text_generation("ptb_words_best.h5", test_data, id_to_word, len(word_to_id))
    
    
    
    