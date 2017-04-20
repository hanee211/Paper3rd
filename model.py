import tensorflow as tf
import word_embedding as wem
import sentence_encoding as sn
import numpy


class Model():
	def __init__(self, params, training=True):
		print("Model Initialize")
		
		if not training:
			print("Test mode")
			batch_size = 1
			seq_length = 1
			output_keep_prob = 1
			input_keep_prob = 1
		else:
			print("Training mode")
			batch_size = params['batch_size']
			output_keep_prob = 1
			input_keep_prob = 1
			
		rnn_size = params['rnn_size']
		
		# same at both encoder and decoder
		num_layers = 1
		total_word_cnt = params['total_word_cnt']
		seq_length = params['seq_length']
		total_size = params['total_size']
		self.C = [tf.Variable(tf.zeros([rnn_size])) for _ in range(total_size)]
		#_______________________________________________________________________________
		#   Encoder Part
		#_______________________________________________________________________________
		cells = []
		
		for _ in range(num_layers):
			_cell = tf.contrib.rnn.GRUCell(num_units=rnn_size, input_size=rnn_size)
		
			if training and (output_keep_prob < 1.0 or input_keep_prob < 1.0):
				_cell = rnn.DropoutWrapper(_cell, input_keep_prob=args.input_keep_prob, output_keep_prob=args.output_keep_prob)
		
			cells.append(_cell)
		
		#encoder_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		encoder_cell = tf.contrib.rnn.GRUCell(num_units=rnn_size, input_size=rnn_size)
		decoder_cell = tf.contrib.rnn.GRUCell(num_units=rnn_size, input_size=rnn_size)

		self.encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, rnn_size])
		initial_state = encoder_cell.zero_state(batch_size, tf.float32)
		encoder_outputs, self.encoder_final_states = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, initial_state=initial_state, dtype=tf.float32)

		#_______________________________________________________________________________
		#   Decoder Part
		#_______________________________________________________________________________


		self.decoder_input = [tf.placeholder(dtype=tf.float32, shape=[None, rnn_size]) for i in range(seq_length)]
		#decoder_input = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(seq_length)]
		self.decoder_target = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(seq_length)]
		self.decorder_initial_state = tf.placeholder(dtype=tf.float32, shape=[None, rnn_size])
		
		
		def loop(prev, _):
			return prev

		decoder_outputs, decoder_states = tf.contrib.legacy_seq2seq.rnn_decoder(self.decoder_input, self.decorder_initial_state, decoder_cell, loop_function=loop if not training else None, scope=None)

		
		W = tf.Variable(tf.random_normal([rnn_size, total_word_cnt], stddev=0.35))
		B = tf.Variable(tf.zeros([total_word_cnt]))

		decoder_outputs_to_symbol = [(tf.matmul(_decode_output, W) + B) for _decode_output in decoder_outputs]
		
		loss_weights = [ tf.ones_like([_b for _b in range(batch_size)], dtype=tf.float32) for _ in range(seq_length) ]
		 
		loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_outputs_to_symbol, self.decoder_target, loss_weights)
		self.cost = tf.reduce_sum(loss) / batch_size / seq_length
		
		
		self.train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
		self.predict = tf.argmax(decoder_outputs_to_symbol, axis=2)
		

if __name__ == '__main__':
	model = Model()