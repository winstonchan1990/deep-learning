import numpy as np
import random
import tensorflow as tf
import datetime
import time
import sys
from tqdm import *

class timer:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + ' start ......')
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + ' finish ......')
            print('Total time taken : {0}s \n\n'.format(end_time - self.begin_time))


def load_text(filepath,encoding):
    start = time.time()
    text = open(filepath,encoding=encoding).read()
    end = time.time()
    print('Total time taken : {}s'.format(end-start))
    print('Text length : {} characters'.format(len(text)))
    print('First 1000 characters :\n')
    print(text[:1000])
    print()
    print('Last 1000 characters :\n')
    print(text[-1000:])
    return text

def generate_char_indices(text):
    chars = sorted(list(set(text)))
    char2id = dict((c,i) for i,c in enumerate(chars))
    id2char = dict((i,c) for i,c in enumerate(chars))
    num_chars = len(chars)
    return char2id, id2char, num_chars

def encode_text(s,char2id):
    return list(map(lambda a: char2id[a], s))

def decode_text(c,id2char):
    return ''.join(map(lambda a: id2char[a], c))

def sample_from_probabilities(prob, topn=0):
    p = np.squeeze(prob)
    assert p.shape[0] >= topn, 'topn is greater than the length of prob'
    if topn>0:
        p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(p.shape[0],1,p=p)[0]

def batch_sequencer(encoded_text,batch_size,seq_size,num_epochs):
    data = np.array(encoded_text)
    data_len = data.shape[0]
    num_batches = (data_len-1) // (batch_size*seq_size)
    assert num_batches > 0 , 'Insufficient data for given batch size'

    truncated_data_len = num_batches * batch_size * seq_size

    X_all = np.reshape(data[0:truncated_data_len], [batch_size, num_batches * seq_size])
    Y_all = np.reshape(data[1:truncated_data_len + 1], [batch_size, num_batches * seq_size])

    # to ensure that text sequence flows from batch to batch within epoch,
    # and from epoch to epoch
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            x = X_all[:, batch * seq_size:(batch + 1) * seq_size]
            y = Y_all[:, batch * seq_size:(batch + 1) * seq_size]
            # shift x and y upwards by 1 row to 'continue' the text sequence in next epoch
            x = np.roll(x, -epoch, axis=0)  
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch
