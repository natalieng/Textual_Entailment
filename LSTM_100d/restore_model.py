import os
import time

from collections import Counter
from utils.general_utils import get_minibatches
from q2_initialization import xavier_weight_init
from LSTMCell import LSTMCell, LSTMCell2
from project2_cleaned import *

import numpy as np
import tensorflow as tf

test_file = 'snli_1.0_test.txt'

train_set, dev_set, test_set, tok2id, labels, n_tokens = init()
embeddings_matrix = loadEmbeddings(tok2id)
input_placeholder_s1 = tf.placeholder(tf.int32, shape=[None, max_length])
input_placeholder_s2 = tf.placeholder(tf.int32, shape=[None, max_length])
labels_placeholder = tf.placeholder(tf.int32, shape=[None, n_classes])
mask_placeholder_s1 = tf.placeholder(tf.bool, shape=[None, max_length])
mask_placeholder_s2 = tf.placeholder(tf.bool, shape=[None, max_length])
dropout_placeholder = tf.placeholder(tf.float32, shape=[])

pred, loss, train_op = buildGraph(input_placeholder_s1, input_placeholder_s2, labels_placeholder, mask_placeholder_s1, mask_placeholder_s2, dropout_placeholder, embeddings_matrix)


if reduced:
    dev_set = dev_set[:reduced_size/2]
dev_examples = vectorize(dev_set, tok2id, labels)
dev_input = embed(dev_examples)

with tf.Session() as sess:
    
    new_saver = tf.train.Saver()#tf.train.import_meta_graph('./models/cryball_RNN-1000.meta')
    new_saver.restore(sess, './models/cryball_RNN-1000')

    predictions = np.argmax(sess.run(pred, 
                feed_dict={input_placeholder_s1: [ex[0] for ex in dev_input], 
                input_placeholder_s2: [ex[1] for ex in dev_input],
                mask_placeholder_s1: [ex[3] for ex in dev_input],
                mask_placeholder_s2: [ex[4] for ex in dev_input], 
                dropout_placeholder: 0}), axis=1)
    num = 0.
    dev_accuracy = 0.
    for i, prediction in enumerate(predictions):
        num += 1
        #print dev_examples[i]['label'], prediction
        if dev_examples[i]['label']== prediction:
            dev_accuracy += 1
    dev_accuracy /= num
    print "- dev accuracy: {:.2f}".format(dev_accuracy * 100.0)
