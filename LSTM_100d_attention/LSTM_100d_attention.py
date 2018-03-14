import os
import time

from collections import Counter
from utils.general_utils import get_minibatches
from q2_initialization import xavier_weight_init
from tensorflow.contrib.layers import l1_l2_regularizer, apply_regularization
from LSTMCell import LSTMCell, LSTMCell2

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#DEFINE GLOBAL VARIABLES
data_path = './data/snli_1.0'
train_file = 'snli_1.0_train.txt'
dev_file = 'snli_1.0_dev.txt'
test_file = 'snli_1.0_test.txt'
embedding_file = './data/glove.6B.100d.txt'
#embedding_file = './data/en-cw.txt'
reduced = False
reduced_size = 4000
UNK = '<UNK>'
n_features = 2
n_classes = 3
dropout = 0.25  # (p_drop in the handout)
hidden_size = 200
embed_size = 100 # CHANGE ME
batch_size = 1024
n_epochs = 30
max_length = 20
max_grad_norm = 5.
lr = 0.001
l1_reg = 0.00001
l2_reg = 0.00001

#READ EXAMPLES INTO A DICTIONARY
def read_nli(in_file):
	examples = []
	with open(in_file) as f:
		for line in f.readlines():
			sp = line.strip().split('\t')
			examples.append({'label': sp[0], 'sentence1': sp[5].split(), 'sentence2': sp[6].split()})
	return examples

#ASSIGN EVERY TOKEN IN TRAINING SET A UNIQUE ID
def build_dict(keys, n_max=None, offset=0):
	count = Counter()
	for key in keys:
		count[key] += 1
	ls = count.most_common() if n_max is None else count.most_common(n_max)
	return {w[0]: index + offset for (index, w) in enumerate(ls)}

#TURNS SENTENCES INTO LISTS OF IDS
def vectorize(examples, tok2id, labels):
	vec_examples = []
	for ex in examples:
		sentence1 = [tok2id[w] if w in tok2id else UNK for w in ex['sentence1']]
		sentence2 = [tok2id[w] if w in tok2id else UNK for w in ex['sentence2']]
		label = labels[ex['label']] if ex['label'] in labels else UNK
		if ex['label'] in labels:
			vec_examples.append({'sentence1': sentence1, 'sentence2': sentence2, 'label': label})
	return vec_examples

def embed(examples):
	embedding = []
	for ex in examples:
		pad_length1 = max(max_length - len(ex['sentence1']), 0)
		pad_length2 = max(max_length - len(ex['sentence2']), 0)
		sentence1 = (ex['sentence1'] + [0] * pad_length1)[:max_length]
		sentence2 = (ex['sentence2'] + [0] * pad_length2)[:max_length]
		sentence1mask = ([True for i in range(len(ex['sentence1']))] + [False for i in range(pad_length1)])[:max_length]
		sentence2mask = ([True for i in range(len(ex['sentence2']))] + [False for i in range(pad_length2)])[:max_length]
		label = np.zeros(3)
		label[ex['label']] = 1
		embedding.append((sentence1, sentence2, label, sentence1mask, sentence2mask))
		#embedding.append((np.concatenate((sentence1vec, sentence2vec)), label))
	#return np.asarray(embedding)
	return embedding


def init():
	global UNK
	#PRINT INITIALIZING
	print(80 * "=")
	print("INITIALIZING")
	print(80 * "=")

	#LOAD THE DATA
	print("Loading data...")
	start = time.time()
	train_set = read_nli(os.path.join(data_path, train_file))
	dev_set = read_nli(os.path.join(data_path, dev_file))
	test_set = read_nli(os.path.join(data_path, test_file))

	if reduced:
		train_set = train_set[:reduced_size]
		dev_set = dev_set[:int(reduced_size/2)]
		test_set = test_set[:int(reduced_size/2)]
	print("took {:.2f} seconds".format(time.time() - start))

	#BUILD TOK2ID FOR TRAINING SET
	tok2id = build_dict([w.lower() for ex in train_set for w in ex['sentence1'] + ex['sentence2']])
	UNK = len(tok2id)
	tok2id[UNK] = len(tok2id)
	n_tokens = len(tok2id)
	labels = {'entailment': 0, 'contradiction': 1, 'neutral': 2, UNK: 3}
	return train_set, dev_set, test_set, tok2id, labels, n_tokens

def buildGraph(input_placeholder_s1, input_placeholder_s2, labels_placeholder, mask_placeholder_s1, mask_placeholder_s2, dropout_placeholder, embeddings_matrix):
	params = tf.Variable(embeddings_matrix)
	tensor_s1 = tf.nn.embedding_lookup(params, input_placeholder_s1)
	tensor_s2 = tf.nn.embedding_lookup(params, input_placeholder_s2)
	embeddings_s1 = tf.reshape(tensor_s1, [-1, max_length, embed_size])
	embeddings_s2 = tf.reshape(tensor_s2, [-1, max_length, embed_size])
	#print embeddings_s1.shape
	#print tf.boolean_mask(embeddings_s1, mask_placeholder_s1, axis=1).shape
	#embeddings = tf.concat([tf.reduce_mean(tf.boolean_mask(embeddings_s1, mask_placeholder_s1), axis=1), tf.reduce_mean(tf.boolean_mask(embeddings_s2, mask_placeholder_s2), axis=1)], 0)
	#print embeddings.shape

	dropout_rate = dropout_placeholder

	preds = [] 
	cell1 = LSTMCell(embed_size, hidden_size)
	cell2 = LSTMCell2(embed_size, hidden_size)

	c = tf.zeros([tf.shape(embeddings_s1)[0], hidden_size])
	h = tf.zeros([tf.shape(embeddings_s2)[0], hidden_size])
	initial_state = tf.contrib.rnn.LSTMStateTuple(c, h)
	l1 = tf.reduce_sum(tf.cast(mask_placeholder_s1, tf.int32), axis = 1)
	outputs1, state1 = tf.nn.dynamic_rnn(cell1, embeddings_s1, dtype=tf.float32, initial_state=initial_state, sequence_length=l1)
	h = tf.zeros([tf.shape(embeddings_s2)[0], hidden_size])
	initial_state = tf.contrib.rnn.LSTMStateTuple(state1.c, h)
	l2 = tf.reduce_sum(tf.cast(mask_placeholder_s2, tf.int32), axis = 1)
	outputs2, state2 = tf.nn.dynamic_rnn(cell2, embeddings_s2, dtype=tf.float32, initial_state=initial_state, sequence_length=l2)

	func = xavier_weight_init()

	# Implementation of attention on the final hidden layer
	Y = tf.transpose(outputs1, perm=[0, 2, 1])
	W_y = tf.Variable(func([hidden_size, hidden_size]))
	W_h = tf.Variable(func([hidden_size, hidden_size]))
	e_l  = tf.constant(1.0, shape=[1, max_length])
	WY = tf.tensordot(W_y,  Y, axes=[[0], [1]])
	WY = tf.transpose(WY, perm=[1, 0, 2])
	h_n = tf.reshape(state2.h, shape=[-1, hidden_size, 1])
	Whe = tf.tensordot(h_n, e_l, axes=[[2], [0]])
	Whe = tf.tensordot(W_h, Whe, axes = [[0], [1]])
	Whe = tf.transpose(Whe, perm = [1, 0, 2])
	M = tf.tanh(WY + Whe)
	w_alpha = tf.Variable(func([1, hidden_size]))
	alpha = tf.nn.softmax(tf.tensordot(w_alpha, M, axes=[[1], [1]]))
	alpha = tf.transpose(alpha, perm=[1, 2, 0])
	alpha = tf.reshape(alpha, shape=[-1, max_length, 1])
	#alpha_entries = tf.unstack(alpha, axis = 0, num=[tf.shape(embeddings_s1)[0]])
	#Y_entries = tf.unstack(Y, axis=0, num=[tf.shape(embeddings_s1)[0]])
	#r = tf.stack([tf.matmul(Y_entries[i], alpha_entries[i]) for i in len(alpha.shape[0])], axis=0)

	#print Y.shape, alpha.shape
	#r = tf.tensordot(Y, alpha, axes=[[2], [1]])
	#r = tf.reduce_mean(r, axis=2)
	#r = r[:, :, 0, :]
	#r = tf.diag_part(r)
	r = tf.matmul(Y, alpha)
	r = tf.reshape(r, shape=[-1, hidden_size])
	#r = Y * alpha
	#print r.shape
	#r = tf.matmul(Y, tf.transpose(alpha, perm=[0, 2, 1]))
	
	U = tf.Variable(func([hidden_size , n_classes]))
	b1 = tf.Variable(tf.zeros([1, n_classes]))
	W_p = tf.Variable(func([hidden_size, hidden_size]))
	W_x = tf.Variable(func([hidden_size, hidden_size]))
	#print r.shape, state2.h.shape
	hstar = tf.tanh(tf.matmul(r, W_p) + tf.matmul(state2.h, W_x))
	#hstar = tf.tanh(tf.matmul(state2.h, W_x))
	h_drop = tf.nn.dropout(hstar, keep_prob = 1-dropout_rate)
	pred =  tf.matmul(h_drop, U) + b1
	#pred = tf.add(tf.matmul(h_drop, U), b1, name="pred")

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_placeholder, logits = pred)
	loss = tf.reduce_mean(loss)
	regularizer = l1_l2_regularizer(l1_reg, l2_reg)
	reg_loss = apply_regularization(regularizer, tf.trainable_variables())
	loss += reg_loss
	#y = labels_placeholder
	#loss = tf.nn.l2_loss(y-preds)
	#loss = tf.reduce_mean(loss)

	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	#train_op = optimizer.minimize(loss)

	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
	gradients = optimizer.compute_gradients(loss)
	grads = [x[0] for x in gradients]
	grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)
	gradients = [(grads[i], gradients[i][1]) for i in range(len(grads))]
	train_op = optimizer.apply_gradients(gradients)
	return pred, loss,  train_op

def loadEmbeddings(tok2id):
	#LOAD IN EMBEDDINGS
	print("Loading pretrained embeddings...")
	start = time.time()
	word_vectors = {}
	n_tokens = len(tok2id)
	for line in open(embedding_file).readlines():
		sp = line.strip().split()
		word_vectors[sp[0]] = [float(x) for x in sp[1:]]
	embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (n_tokens, embed_size)), dtype='float32')

	#CREATE EMBEDDINGS MATRIX
	for token in tok2id:
		i = tok2id[token]
		if token in word_vectors:
			embeddings_matrix[i] = word_vectors[token]
	print("took {:.2f} seconds".format(time.time() - start))
	return embeddings_matrix

if __name__ == "__main__":
	train_set, dev_set, test_set, tok2id, labels, n_tokens = init()
	embeddings_matrix = loadEmbeddings(tok2id)

	#ENSURE WEIGHTS DIRECTORY EXISTS
	if not os.path.exists('./data/weights/'):
			os.makedirs('./data/weights/')

	train_examples = vectorize(train_set, tok2id, labels)
	dev_examples = vectorize(dev_set, tok2id, labels)
	test_examples = vectorize(test_set, tok2id, labels)

	train_input = embed(train_examples)
	dev_input = embed(dev_examples)
	test_input = embed(test_examples)

	input_placeholder_s1 = tf.placeholder(tf.int32, shape=[None, max_length])
	input_placeholder_s2 = tf.placeholder(tf.int32, shape=[None, max_length])
	labels_placeholder = tf.placeholder(tf.int32, shape=[None, n_classes])
	mask_placeholder_s1 = tf.placeholder(tf.bool, shape=[None, max_length])
	mask_placeholder_s2 = tf.placeholder(tf.bool, shape=[None, max_length])
	dropout_placeholder = tf.placeholder(tf.float32, shape=[])

	pred, loss, train_op = buildGraph(input_placeholder_s1, input_placeholder_s2, labels_placeholder, mask_placeholder_s1, mask_placeholder_s2, dropout_placeholder, embeddings_matrix)

	best_dev = 0

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		#new_saver = tf.train.Saver()
		#new_saver.restore(sess, './models/regularized_RNN_100d-1-29')
		saver = tf.train.Saver()
		for epoch in range(n_epochs):
			print("Epoch: ", epoch)
			#n_minibatches = 1 + len(train_embeddings) / batch_size
			#prog = tf.keras.utils.Progbar(target=n_minibatches)


			for i, (s1, s2, l, m1, m2) in enumerate(get_minibatches([[ex[0] for ex in train_input], 
					[ex[1] for ex in train_input], [ex[2] for ex in train_input], [ex[3] for ex in train_input], [ex[4] for ex in train_input]], batch_size)):
				feed_dict = {input_placeholder_s1: s1, 
					input_placeholder_s2: s2, 
					mask_placeholder_s1: m1, 
					mask_placeholder_s2: m2, 
					labels_placeholder: l, 
					dropout_placeholder: dropout}
				sess.run(train_op, feed_dict=feed_dict)
				_, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
				#print "Gradient: ", global_norm.eval()
				#_, batch_loss = sess.run([train_op(loss), loss], feed_dict=feed_dict)
				#print pred, labels_placeholder, loss, losses
				#prog.update(i + 1, [("train loss", batch_loss)], force=i + 1 == n_minibatches)
				print(i, " Batch loss: ", batch_loss)

			print("Evaluating on train set")
			#feed = self.create_feed_dict(inputs_batch)
		#	predictions = np.argmax(sess.run(pred, 
		#		feed_dict={input_placeholder_s1: [ex[0] for ex in train_input], 
		#		input_placeholder_s2: [ex[1] for ex in train_input],
		#		mask_placeholder_s1: [ex[3] for ex in train_input],
		#		mask_placeholder_s2: [ex[4] for ex in train_input], 
		#		dropout_placeholder: 0}), axis=1)
			accuracy_batch_size = 20000
			num = 0.
			train_accuracy = 0.
			for k in range(max(1, int(len(train_input)/accuracy_batch_size))):
				examples = train_input[k*accuracy_batch_size:(k+1)*accuracy_batch_size]
				d = {input_placeholder_s1: [example[0] for example in examples], 
					input_placeholder_s2: [example[1] for example in examples],
					mask_placeholder_s1: [example[3] for example in examples],
					mask_placeholder_s2: [example[4] for example in examples], dropout_placeholder:0}
				predictions = np.argmax(sess.run(pred, feed_dict=d), axis=1)
				for j, prediction in enumerate(predictions):
					if train_examples[k*accuracy_batch_size + j]['label']==prediction:
						train_accuracy +=1
					num += 1
			#for i, prediction in enumerate(predictions):
		#		num += 1
				#print dev_examples[i]['label'], prediction
		#		if train_examples[i]['label']== prediction:
		#			train_accuracy += 1
			train_accuracy /= num
			print("- train accuracy: {:.2f}".format(train_accuracy * 100.0)) 

			print("Evaluating on dev set")
			#feed = self.create_feed_dict(inputs_batch)
			#predictions = np.argmax(sess.run(pred, 
			#	feed_dict={input_placeholder_s1: [ex[0] for ex in dev_input], 
		#		input_placeholder_s2: [ex[1] for ex in dev_input],
			#	mask_placeholder_s1: [ex[3] for ex in dev_input],
			#	mask_placeholder_s2: [ex[4] for ex in dev_input], 
			#	dropout_placeholder: 0}), axis=1)
			num = 0.
			dev_accuracy = 0.
			for k in range(max(1, int(len(dev_examples)/accuracy_batch_size))):
				examples = dev_input[k*accuracy_batch_size:(k+1)*accuracy_batch_size]
				d = {input_placeholder_s1: [example[0] for example in examples], 
					input_placeholder_s2: [example[1] for example in examples],
					mask_placeholder_s1: [example[3] for example in examples],
					mask_placeholder_s2: [example[4] for example in examples], dropout_placeholder:0}
				predictions = np.argmax(sess.run(pred, feed_dict=d), axis=1)
				for j, prediction in enumerate(predictions):
					if dev_examples[k*accuracy_batch_size + j]['label']==prediction:
						dev_accuracy +=1
					num += 1
			#for i, prediction in enumerate(predictions):
			#	num += 1
				#print dev_examples[i]['label'], prediction
			#	if dev_examples[i]['label']== prediction:
			#		dev_accuracy += 1
			dev_accuracy /= num
			print("- dev accuracy: {:.2f}".format(dev_accuracy * 100.0))
			if dev_accuracy > best_dev:
				best_dev = dev_accuracy
				print("New best dev accuracy achieved! Saving model!")
				saver.save(sess, './models/attention_RNN_100d', global_step=epoch)
