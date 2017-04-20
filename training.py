from __future__ import print_function
import tensorflow as tf
import word_embedding as wem
import sentence_encoding as sn
import time
import os
import numpy as np
from model import Model

def train():
	print("start training!!")
	
	params = dict()
	seq_length = 13
	epochs = 100
	
	rnn_size = 3
	params['rnn_size'] = rnn_size
	params['seq_length'] = seq_length
	#워드 임베딩을 가져오고
	word2em, word2id = wem.get_embeddingLookup()
	total_word_cnt = len(word2em)
	params['total_word_cnt'] = total_word_cnt
	params['total_size'] = len(sn.get_sentences())
	#이 워드 임베딩으로 문장을 인코딩 까지 해줌. 
	em_encoded_sentences = sn.get_encoded_sentences(word2em, seq_length)
	id_encoded_sentences = sn.get_encoded_sentences(word2id, seq_length)
	em_encoded_sentences_for_decoder = sn.get_encoded_sentences_for_decoder(word2em, seq_length)

	total_size = len(em_encoded_sentences)
	batch_size = 1
	params['batch_size'] = batch_size
	
	model = Model(params)
	
	with tf.Session() as sess:
		summaries = tf.summary.merge_all()
		#writer = tf.summary.FileWriter('./graph', time.strftime("%Y-%m-%d-%H-%M-%S"))
		#writer.add_graph(sess.graph)
		
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		
		model_ckpt_file = './status/model.ckpt'
		if os.path.isfile(model_ckpt_file):
			print("check!!")
			saver.restore(sess, model_ckpt_file)
			
		for e in range(epochs):
			# get bunch of data
			start = 0
			for i in range(int(total_size/batch_size) + 1):
				end = start + batch_size
				
				if i == total_size/batch_size:
					if start > total_size - 1:
						break
					else:
						end = total_size - 1
					
					
				#1. encoder_final_states
				em_en_tran = np.transpose(em_encoded_sentences[start:end], (1,0,2))
				em_de_tran = np.transpose(em_encoded_sentences_for_decoder[start:end], (1,0,2))
				id_en_tran = np.transpose(id_encoded_sentences[start:end])
				
				feed = {model.encoder_input:em_encoded_sentences[start:end]}
				
				encoder_states = sess.run(model.encoder_final_states, feed_dict=feed)
				
				feed_state = dict()
				feed_state[model.decorder_initial_state] = encoder_states
				
				for r in range(end-start):
					model.C[start + r] = encoder_states[r]
				
				feed_decoder_input = dict()
				feed_decoder_target = dict()
				
				
				for i in range(seq_length):

					feed_decoder_input[model.decoder_input[i]] = em_de_tran[i]
					feed_decoder_target[model.decoder_target[i]] = id_en_tran[i]
				
				feed_decoder = dict()
				feed_decoder = feed_decoder_input
				feed_decoder.update(feed_decoder_target)
				feed_decoder.update(feed_state)
				_cost, _ = sess.run([model.cost, model.train], feed_dict=feed_decoder)

				

				start = end
			
			if e % 50 == 0:
				saver.save(sess, model_ckpt_file)
				print("mode saved to ", model_ckpt_file)
			
		
if __name__ == '__main__':
	train()