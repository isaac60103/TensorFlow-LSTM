#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import tensorflow as tf
import os
import argparse
import re

vocab = None
vocab_size = None
idx_to_vocab = None
vocab_to_idx = None
data = None

def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
    Raises:
    ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield ptb_iterator(data, batch_size, num_steps)

def resetGraph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def trainNetwork(g, num_epochs, num_steps = 100, batch_size = 64, verbose = True, save=False):
    tf.set_random_seed(1234)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

def buildGraph( state_size, num_classes, batch_size, num_steps, num_layers, learning_rate):

    resetGraph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )


def saveConfig(path, num_time_steps, layers, state_size, vocab_size):

    f = open(path+'/network_config', 'w')

    f.write("number of time steps: ")
    f.write(str(num_time_steps)+"\n")
    f.write("number of layers: ")
    f.write(str(layers)+"\n")
    f.write("state size: ")
    f.write(str(state_size)+"\n")
    f.write("number of vocab size: ")
    f.write(str(vocab_size)+"\n")

    f.close()

    return



def generateCharacters(g, checkpoint, num_chars, prompt, pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))

def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def readTargetFile(training_file):
    
    with open(training_file,'r') as f:
        raw_data =  f.read()
        raw_data = unicode(raw_data, 'utf8')

    global vocab
    global vocab_size 
    global idx_to_vocab
    global vocab_to_idx
    global data

    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    data = [vocab_to_idx[c] for c in raw_data]

    print("Total data length:", len(raw_data))

    del raw_data
    return

def getConfig(path):

    f = open(path+'/network_config', 'r')
    num_time_steps = re.findall(r'\d+',f.readline())[0]
    num_layers = re.findall(r'\d+',f.readline())[0]
    state_size = re.findall(r'\d+',f.readline())[0]
    vocab_size = re.findall(r'\d+',f.readline())[0]

    f.close()
    return int(num_layers), int(state_size), int(vocab_size)
    

def main():

    parser = argparse.ArgumentParser(
    description="",
    )
    parser.add_argument("-m", "--mode", type=str, default="train",
    help="train mode or generate mode")
    parser.add_argument("-t", "--num_time_steps", type=int, default=100,
    help="number of time steps")
    parser.add_argument("-l", "--layers", type=int, default=3,
    help="number of layers")
    parser.add_argument("--num_epochs", type=int, default=20,
    help="number of epochs")
    parser.add_argument("--learning_rate", type=int, default=0.001,
    help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
    help="batch size")
    parser.add_argument("--state_size", type=int, default=256,
    help="state_size")
    parser.add_argument("-p", "--pick_top_chars", type=int, default=5,
    help="pick number of top probability of chars")
    parser.add_argument("-c", "--num_chars", type=int, default=500,
    help="number of characters to generate")
    parser.add_argument("-i", "--input", type=str,
    help="training file")
    parser.add_argument("-o", "--output",type=str,
    help="output result")

    args = parser.parse_args()

    global vocab_size 

    if args.mode == 'train':
        print('Training ......')
        createDir('saves')
        readTargetFile(args.input)
        saveConfig("saves", args.num_time_steps, args.layers, args.state_size, vocab_size)

        g = buildGraph(
            num_steps=args.num_time_steps, 
            num_layers=args.layers, 
            state_size=args.state_size, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate, 
            num_classes= vocab_size)

        losses = trainNetwork(g, args.num_epochs, args.num_time_steps, args.batch_size, verbose = True, save="saves/lstm_result")
        print('Training Finish!')

        g = buildGraph(
            num_steps=1, 
            num_layers=args.layers, 
            state_size=args.state_size, 
            batch_size=1, 
            learning_rate=args.learning_rate, 
            num_classes=vocab_size)

        generateCharacters(g, "saves/lstm_result", args.num_chars, prompt=u'楊', pick_top_chars=args.pick_top_chars)

    else:
        print('Generating ......')

        num_layers, state_size, vocab_size = getConfig("saves")
        readTargetFile(args.input)
        g = buildGraph(
            num_steps=1, 
            num_layers=num_layers, 
            state_size=state_size, 
            batch_size=1, 
            learning_rate=0.001, 
            num_classes= vocab_size)
        generateCharacters(g, "saves/lstm_result", args.num_chars, prompt=u'楊', pick_top_chars=args.pick_top_chars)

if __name__ == "__main__":
    main()






