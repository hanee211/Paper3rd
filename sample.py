from __future__ import print_function
import tensorflow as tf
import os
from model import Model
import word_embedding as wem
import sentence_encoding as sn

def sample():
	
	params = dict()
	seq_length = 13
	
	rnn_size = 3
	params['rnn_size'] = rnn_size	
	params['seq_length'] = seq_length
	word2em, word2id = wem.get_embeddingLookup()
	total_word_cnt = len(word2em)
	params['total_word_cnt'] = total_word_cnt
	batch_size = 1
	params['batch_size'] = batch_size
	total_size = len(sn.get_sentences())
	params['total_size'] = total_size

	model = Model(params, training=False)
	
	print("before session")
	with tf.Session() as sess:
		print("1")
		tf.global_variables_initializer().run()
		saver = tf.train.Saver(tf.global_variables())
		print("2")
		model_ckpt_file = './status/model.ckpt'
		print("3")
		if os.path.isfile(model_ckpt_file):
			print("Setting done.")
			saver.restore(sess, model_ckpt_file)
			
			feed_predict = dict()
			for i in range(seq_length):
				feed_predict[model.decoder_input[i]] = word2em["GO"]
				
			feed_predict[model.decorder_initial_state] = model.C[1]
			print("Predict result....")
			print(sess.run(model.predict, feed_dict=feed_predict))
		print("4")

if __name__ == '__main__':
	sample()	