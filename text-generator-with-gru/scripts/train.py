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
import pickle
import glob
from utils import *

## Settings

timestamp = str(math.trunc(time.time()))

parser = argparse.ArgumentParser()
trainingArgs = parser.add_argument_group('Training options')
trainingArgs.add_argument('--seqLen',type=int,default=30,help='character length of each training sample')
trainingArgs.add_argument('--batchSize',type=int,default=100,help='number of training samples in one batch')
trainingArgs.add_argument('--internalSize',type=int,default=512,help='internal size of each GRU cell')
trainingArgs.add_argument('--numLayers',type=int,default=3,help='number of GRU layers in the network')
trainingArgs.add_argument('--learningRate',type=float,default=0.001,help='learning rate')
trainingArgs.add_argument('--dropoutKeep',type=float,default=1.0,help='1-dropout rate')
trainingArgs.add_argument('--numEpochs',type=int,default=100,help='number of epochs')
globalArgs = parser.add_argument_group('Global options')
globalArgs.add_argument('--modelName',type=str,default='model_{}'.format(timestamp),help='unique model name')
globalArgs.add_argument('--textFilePath',type=str,default='',help='file path to input text data file')
globalArgs.add_argument('--removeNonASCII',action='store_true',default=False,help='remove non-ASCII characters')
globalArgs.add_argument('--resumeTraining',action='store_true',default=False,help='resume training from a previously saved model')
ARGS = parser.parse_args()

SEQLEN = ARGS.seqLen
BATCHSIZE = ARGS.batchSize
INTERNALSIZE = ARGS.internalSize
NLAYERS = ARGS.numLayers
learning_rate = ARGS.learningRate
dropout_pkeep = ARGS.dropoutKeep
numEpochs = ARGS.numEpochs
modelName = ARGS.modelName
textfilepath = ARGS.textFilePath 

assert len(textfilepath)>0,'Please input file path of input text data file'
assert os.path.exists(textfilepath),'Please specify valid file path of input text data file'
assert len(modelName)>0,'Please input name of model'

DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

print('\n[Settings]\n')
print('Sequence length : {}'.format(SEQLEN))
print('Batch size : {}'.format(BATCHSIZE))
print('Internal size : {}'.format(INTERNALSIZE))
print('Number of layers : {}'.format(NLAYERS))
print('Learning rate : {}'.format(learning_rate))
print('Dropout Keep rate : {}'.format(dropout_pkeep))
print('Number of epochs : {}'.format(numEpochs))
print('Model name : {}'.format(modelName))
print('Input data file : {}'.format(textfilepath))
print('\n[End Settings]\n')

if not os.path.exists('models'):
    os.mkdir('models')
    
## Create sub-directory to store model output
if not os.path.exists('models/{}'.format(modelName)):
    os.mkdir('models/{}'.format(modelName))
    os.mkdir('models/{}/chars'.format(modelName))
    os.mkdir('models/{}/train'.format(modelName))
    os.mkdir('models/{}/checkpoints'.format(modelName))
    os.mkdir('models/{}/log'.format(modelName))

## Load raw text
with timer('Loading raw text file'):
    assert os.path.exists(textfilepath), 'Input text file does not exist.'
    text = load_text(textfilepath,encoding='utf-8')

## Remove non-ASCII characters (optional)
if ARGS.removeNonASCII:
    with timer('Removing non-ASCII characters'):
        ascii_char_idx = [i for i in range(0,127,1)]
        text = ''.join([t for t in text if ord(t) in ascii_char_idx])

## Generate character indices mapping and encode text
with timer('Generating char indices'):
    char2id, id2char, num_chars = generate_char_indices(text)
    encoded_text = encode_text(text,char2id)
    print('Number of characters : {}'.format(num_chars))
    print()
    print('Characters:\n')
    for i,c in id2char.items():
        print(i,c.encode('utf-8'))
    print()
    print('First 1000 entries of encoded text : {}'.format(encoded_text[:1000]))

## Save character indices in json files
with timer('Saving char indices'):
    with open('models/{}/chars/char2id.json'.format(modelName),'w') as f_char2id:
        with open('models/{}/chars/id2char.json'.format(modelName),'w') as f_id2char:
            json.dump(char2id,f_char2id)
            json.dump(id2char,f_id2char)

## Define epoch size
epoch_size = len(text) // (BATCHSIZE * SEQLEN)
print('Number of batches per epoch: {}'.format(epoch_size))

###############################
## Build computational graph ##
###############################

print('Building computational graph')
graph = tf.Graph()
with graph.as_default():

    # Parameters
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize') # batch size

    # Inputs
    X = tf.placeholder(tf.uint8, [None, None], name='X')    
    Xo = tf.one_hot(X, num_chars, 1.0, 0.0, name='Xo')

    # Outputs (expected)
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, num_chars, 1.0, 0.0, name='Yo_')      

    # Hidden input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin') 

    # NLAYERS of GRU cells
    def build_gru_cell():
        onecell = rnn.GRUCell(INTERNALSIZE)
        dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
        return(dropcell)

    multicell = rnn.MultiRNNCell([build_gru_cell() for _ in range(NLAYERS)], state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    
    Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
    H = tf.identity(H, name='H') # to assign a name to H

    # Softmax layer
    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, num_chars)     # [ BATCHSIZE x SEQLEN, num_chars ]
    
    Yflat_ = tf.reshape(Yo_, [-1, num_chars])     # [ BATCHSIZE x SEQLEN, num_chars ]
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])                                       # [ BATCHSIZE, SEQLEN ]

    Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, num_chars ]
    Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name='Y')  # [ BATCHSIZE, SEQLEN ]
    
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])


###############
## Run graph ##
###############

print('Model training')
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    ### Init for Tensorboard
    summary_writer = tf.summary.FileWriter('models/{}/log/training'.format(modelName))

    ### If restoring from prev checkpoint
    if ARGS.resumeTraining:
        # Check if checkpoint directory exists
        dir_checkpoints = 'models/{}/checkpoints/'.format(modelName)
        assert os.path.exists(dir_checkpoints), 'Checkpoint directory not found'
      
        with timer('Restoring latest checkpoint'):
            latest_checkpoint = tf.train.latest_checkpoint('models/{}/checkpoints'.format(modelName))
            print(latest_checkpoint)
            assert len(latest_checkpoint)>0,'No checkpoint file found'
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, latest_checkpoint)
            for var in tf.global_variables():
                print(var)

        with timer('Restore initial input state as previous output state'):
            istate = pickle.load(open('models/{}/train/ostate.pkl'.format(modelName),'rb'))
            print('istate : {}'.format(istate)) 

        with timer('Restore global step value'):
            step = pickle.load(open('models/{}/train/step.pkl'.format(modelName),'rb'))
            print('step : {}'.format(step))

    ### If training from scratch
    else:
        # Init for saving model
        saver = tf.train.Saver(max_to_keep=1)

        # Initialize model
        istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
        tf.global_variables_initializer().run()
        step = 0

    ### Batch iterator
    batch_iterator = batch_sequencer(encoded_text, BATCHSIZE, SEQLEN, num_epochs=numEpochs)

    ### Fast forwarding batch_sequencer to latest train step
    iter_skip = step // (BATCHSIZE * SEQLEN)
    with timer('Fast forwarding batch_sequencer to latest train step'):
        for i in range(iter_skip):
            next(batch_iterator)
    
    ### Start training at last saved step
    for x, y_, epoch in batch_iterator:

        # train on one minibatch
        feed_dict = {
            X: x, 
            Y_: y_, 
            Hin: istate, 
            lr: learning_rate, 
            pkeep: dropout_pkeep, 
            batchsize: BATCHSIZE
        }

        _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

        # save training data for Tensorboard
        summary_writer.add_summary(smm, step)

        # display updates every 50 batches
        if step % _50_BATCHES == 0:
            feed_dict = {
                X: x, 
                Y_: y_, 
                Hin: istate, 
                pkeep: 1.0, # no dropout for validation
                batchsize: BATCHSIZE
            }  

            y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)

            batch_size = x.shape[0]  # batch_size in number of sequences
            sequence_len = x.shape[1]  # sequence_len in number of characters
            
            start_index_in_epoch = step % (epoch_size * batch_size * sequence_len)
            batch_index = start_index_in_epoch // (batch_size * sequence_len)
            
            batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
            stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, bl, acc)
            print()
            print("TRAINING STATS: {}".format(stats))

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            print('[Generating random text from learned state]')
            print()
            rand_char_idx = np.random.choice(num_chars)

            ry = np.array([[rand_char_idx]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            display_string = ''
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                display_string += id2char[rc]
                ry = np.array([[rc]])
            
            print(display_string.encode('utf-8'))
            print()
            print('[Finished generating random text]')
            print()

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            with timer('Saving train variables'):
                pickle.dump(ostate,open('models/{}/train/ostate.pkl'.format(modelName),'wb'))
                pickle.dump(step,open('models/{}/train/step.pkl'.format(modelName),'wb'))

            with timer('Saving checkpoint'):
                saver.save(
                    sess, 
                    'models/{}/checkpoints/train-step'.format(modelName), 
                    global_step=step
                )