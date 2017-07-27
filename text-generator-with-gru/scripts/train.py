import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  
import os
import time
import datetime
import math
import numpy as np
from utils import *

## Settings

SEQLEN = 30
BATCHSIZE = 100
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 1.0    # no dropout
textfilepath = 'data/wiki.test.raw'

DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

## Load raw text
with timer('Loading raw text file'):
    text = load_text(textfilepath,encoding='utf-8')

## Generate character indices mapping and encode text
with timer('Generating char indices'):
    char2id, id2char, num_chars = generate_char_indices(text)
    encoded_text = encode_text(text,char2id)
    print('Number of characters : {}'.format(num_chars))
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

    # onecell = rnn.GRUCell(INTERNALSIZE)
    # dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    # multicell = rnn.MultiRNNCell([dropcell]*NLAYERS, state_is_tuple=False)

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
    for x, y_, epoch in batch_sequencer(encoded_text, BATCHSIZE, SEQLEN, num_epochs=1000):

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
            print('Generating random text from learned state : ')
            print()
            rand_char_idx = np.random.choice(num_chars)

            ry = np.array([[rand_char_idx]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(id2char[rc].encode('utf-8'), end='')
                ry = np.array([[rc]])
            
            print('Finished generating random text')
            print()

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            with timer('Saving checkpoint'):
                saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN


