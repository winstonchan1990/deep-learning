import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  
import os
import time
import datetime
import math
import numpy as np
import argparse
from utils import *

## Settings

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
globalArgs.add_argument('--textFilePath',type=str,default='data/wiki.test.raw',help='file path to input text data file')
globalArgs.add_argument('--removeNonASCII',action='store_true',default=False,help='remove non-ASCII characters')
ARGS = parser.parse_args()

SEQLEN = ARGS.seqLen
BATCHSIZE = ARGS.batchSize
INTERNALSIZE = ARGS.internalSize
NLAYERS = ARGS.numLayers
learning_rate = ARGS.learningRate
dropout_pkeep = ARGS.dropoutKeep
numEpochs = ARGS.numEpochs
textfilepath = ARGS.textFilePath 

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
print('Input data file : {}'.format(textfilepath))
print('\n[End Settings]\n')


## Load raw text
with timer('Loading raw text file'):
    assert os.path.exists(textfilepath), 'Input text file does not exist.'
    text = load_text(textfilepath,encoding='utf-8')

## Remove non-ASCII characters (optional)
if ARGS.removeNonASCII:
    with timer('Removing non-ASCII characters'):
        usable_char_idx = [i for i in range(0,127,1)]
        text = ''.join([t for t in text if ord(t) in usable_char_idx])

## Generate character indices mapping and encode text
with timer('Generating char indices'):
    char2id, id2char, num_chars = generate_char_indices(text)
    encoded_text = encode_text(text,char2id)
    print('Number of characters : {}'.format(num_chars))
    print('Characters:\n')
    for i,c in id2char.items():
        print(i,c.encode('utf-8'))
    print('First 1000 entries of encoded text : {}'.format(encoded_text[:1000]))

## Define epoch size
epoch_size = len(text) // (BATCHSIZE * SEQLEN)
print('Number of batches per epoch: {}'.format(epoch_size))

## Build computational graph
print('Building computational graph')
graph = tf.Graph()
with graph.as_default():
    # Parameters
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize') # batch size

    # Inputs
    X = tf.placeholder(tf.uint8, [None, None], name='X')    
    Xo = tf.one_hot(X, num_chars, 1.0, 0.0)

    # Outputs (expected)
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    Yo_ = tf.one_hot(Y_, num_chars, 1.0, 0.0)      

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




#initialize a session with a graph
print('Model training')
with tf.Session(graph=graph) as sess:

    # Init for Tensorboard
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

    # Init for saving model
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1)


    # Initialize model
    istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
    tf.global_variables_initializer().run()
    step = 0
    

    # training loop
    for x, y_, epoch in batch_sequencer(encoded_text, BATCHSIZE, SEQLEN, num_epochs=numEpochs):

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

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            with timer('Saving checkpoint'):
                saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN


