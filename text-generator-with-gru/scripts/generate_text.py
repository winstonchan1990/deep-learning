import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  
import os
import time
import datetime
import math
import numpy as np
import argparse
import json
import glob
from utils import *

## Settings [make sure they are the same as the training settings]

parser = argparse.ArgumentParser()
globalArgs = parser.add_argument_group('Global options')
globalArgs.add_argument('--internalSize',type=int,default=512,help='internal size of each GRU cell [must be same as training settings]')
globalArgs.add_argument('--numLayers',type=int,default=3,help='number of GRU layers in the network [must be same as training settings]')
globalArgs.add_argument('--modelName',type=str,default='',help='model name')
globalArgs.add_argument('--generatedTextLength',type=int,default=10000,help='number of characters in generated text')
globalArgs.add_argument('--outfileGeneratedText',type=str,default='',help='filepath to output generated text')
ARGS = parser.parse_args()

INTERNALSIZE = ARGS.internalSize
NLAYERS = ARGS.numLayers
modelName = ARGS.modelName
generatedTextLength = ARGS.generatedTextLength
outfileGeneratedText = ARGS.outfileGeneratedText

print('\n[Settings]\n')
print('Internal size : {}'.format(INTERNALSIZE))
print('Number of layers : {}'.format(NLAYERS))
print('Length of generated text : {}'.format(generatedTextLength))
print('\n[End Settings]\n')

## Check valid model name
assert len(modelName)>0,'Please input name of model'
assert os.path.exists('models/{}'.format(modelName)), 'Please enter a valid model name'

## chars directory
dir_chars = 'models/{}/chars/'.format(modelName)
char2idFile = 'models/{}/chars/char2id.json'.format(modelName)
with timer('Reading character indices from {}'.format(char2idFile)):
    with open(char2idFile,'r') as f_char2id:
        char2id = json.load(f_char2id)
        id2char = {j:i for i,j in char2id.items()}
        num_chars = len(char2id.keys())

## ASCII char index
ascii_char_idx = [i for i in range(0,127,1)]

## Text Generator
ncnt = 0
with tf.Session() as sess:

    dir_checkpoints = 'models/{}/checkpoints/'.format(modelName)
    metagraphFile = glob.glob(dir_checkpoints+'*.meta')
    assert len(metagraphFile)>0,'MetaGraphDef file is missing from checkpoints sub-directory'
    assert len(metagraphFile)<2,'more than 1 MetaGraphDef files found'
    metagraphFile = metagraphFile[0]

    latest_checkpoint = tf.train.latest_checkpoint('models/{}/checkpoints'.format(modelName))

    with timer('Importing MetaGraphDef file {}'.format(metagraphFile)):
        new_saver = tf.train.import_meta_graph(metagraphFile)

    with timer('Restoring latest checkpoint {}'.format(latest_checkpoint)):
        new_saver.restore(sess, latest_checkpoint)

    with timer('Generating initial input to feed into model'):
        init_char_seq = input('Enter some text to initiate the text generator: ')
        init_char_idx = [char2id[char] for char in init_char_seq]
        x = np.array([init_char_idx])
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)

    print('\n\n[Text generation start]\n\n')

    generated_string = ''
    for idx in init_char_idx:
        generated_string += id2char[idx]

    print(generated_string,end='')

    for i in range(generatedTextLength):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
        generated_char_idx = [sample_from_probabilities(yo_, topn=10) for yo_ in yo]
        y = np.array([[generated_char_idx[-1]]])  # [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1 after initial text input
        
        generated_char_seq = [id2char[idx] for idx in generated_char_idx]
        generated_char_seq = ''.join(generated_char_seq)
        generated_string += generated_char_seq
        
        for char in generated_char_seq:
            if ord(char) in ascii_char_idx : # can be printed to console
                print(char, end='')
            if char == '\n':
                ncnt = 0 # new line
            else:
                ncnt += 1
            if ncnt == 100:
                print('') # new line
                ncnt = 0

    print('\n\n[Text generation end]\n\n')

    if outfileGeneratedText:
        with timer('Writing generated text to {}'.format(outfileGeneratedText)):
            with open(outfileGeneratedText,encoding='utf-8') as fout:
                fout.write(generated_string)



