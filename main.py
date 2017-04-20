import tensorflow as tf
import word_embedding as wem
import sentence_encoding as sn
import numpy


class Model():
	def __init__(self, args, training=True):
		print("start Model")
		'''
		hidden_size = 3
		word2vec_dim = 3
		batch_size = 1
		# same at both encoder and decoder
		seq_length = 12

		#워드 임베딩을 가져오고
		word2em, word2id = wem.get_embeddingLookup()
		total_word_cnt = len(word2em)


		#이 워드 임베딩으로 문장을 인코딩 까지 해줌. 
		em_encoded_sentences = sn.get_encoded_sentences(word2em, seq_length)
		id_encoded_sentences = sn.get_encoded_sentences(word2id, seq_length)
		id_encoded_sentences_for_decoder = sn.get_encoded_sentences_for_decoder(word2id, seq_length)

		print(em_encoded_sentences[0])
		print(id_encoded_sentences[0])
		print(id_encoded_sentences_for_decoder[0])

		#_______________________________________________________________________________
		#   Encoder Part
		#_______________________________________________________________________________
		encoder_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size, input_size=word2vec_dim)
		decoder_cell = tf.contrib.rnn.GRUCell(num_units=hidden_size, input_size=word2vec_dim)

		encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, word2vec_dim])
		initial_state = encoder_cell.zero_state(batch_size, tf.float32)
		encoder_outputs, encoder_final_states = tf.nn.dynamic_rnn(encoder_cell, encoder_input, initial_state=initial_state, dtype=tf.float32)

		#_______________________________________________________________________________
		#   Decoder Part
		#_______________________________________________________________________________


		decoder_input = [tf.placeholder(dtype=tf.float32, shape=[None, word2vec_dim]) for i in range(seq_length)]
		#decoder_input = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(seq_length)]
		decoder_target = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(seq_length)]
		decorder_initial_state = tf.placeholder(dtype=tf.float32, shape=[None, hidden_size])
		infer = False

		def loop(prev, _):
			return prev

		decoder_outputs, decoder_states = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_input, decorder_initial_state, decoder_cell, loop_function=loop if infer else None, scope=None)


		W = tf.Variable(tf.random_normal([hidden_size, total_word_cnt], stddev=0.35))
		B = tf.Variable(tf.zeros([total_word_cnt]))

		decoder_outputs_to_symbol = [(tf.matmul(_decode_output, W) + B) for _decode_output in decoder_outputs]

		loss_weights = [ tf.ones_like(y, dtype=tf.float32) for y in range(seq_length) ]

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		sess.run(encoder_final_states, feed_dict={encoder_input:[em_encoded_sentences[0]]})
		'''
		'''
		loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs_to_symbol, decoder_target, loss_weights)

		train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
		predict = tf.argmax(decoder_symbols, axis=2)
		'''
