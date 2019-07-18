import tensorflow as tf
import tensorflow_hub as hub

from utils import initializer
from attention import attention_output
from transformer import multihead_attention, feedforward


class EntityAttentionLSTM:
    def __init__(self, sequence_length, num_classes,
                 vocab_size, embedding_size, pos_vocab_size, pos_embedding_size,
                 hidden_size, num_heads, attention_size,
                 use_elmo=False, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.input_text = tf.placeholder(tf.string, shape=[None, ], name='input_text')
        self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # print("e1 shape:", self.input_e1.get_shape()) #(?,)

        if use_elmo:
            # Contextual Embedding Layer
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("/home/wangyan/relation_extraction/ELMo", trainable=True)
                self.embedded_chars = elmo_model(self.input_text, signature="default", as_dict=True)["elmo"]
                print(self.embedded_chars.get_shape())
        else:
            # Word Embedding Layer
            with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
                self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
                self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_x)
                # print("shape:",self.embedded_chars.get_shape()) #(?, 90, 300)

        # Position Embedding Layer
        # with tf.device('/cpu:0'), tf.variable_scope("position-embeddings"):
        #     self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
        #     print("embedding_char:", self.embedded_chars.get_shape()[1])
        #     self.p1 = tf.nn.embedding_lookup(self.W_pos, self.input_p1)[:, :tf.shape(self.embedded_chars)[1]]
        #     self.p2 = tf.nn.embedding_lookup(self.W_pos, self.input_p2)[:, :tf.shape(self.embedded_chars)[1]]
        #     # print("p shape:", self.p1.get_shape()) # (?, ? 50)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars,  self.emb_dropout_keep_prob)

        dim_model = 300
        dim_ff = 1200
        num_stack = 1
        num_heads = 12
        # self.embedded_chars = self.text_embedded_chars
        ##transformer
        for i in range(num_stack):
            with tf.variable_scope("block-{}".format(i)):
                # Multi-head Attention (self attention)
                with tf.variable_scope("multihead-attention"):
                    self.mh = multihead_attention(query=self.embedded_chars, key=self.embedded_chars,
                                                  value=self.embedded_chars,
                                                  dim_model=dim_model, num_head=num_heads)
                    # Residual & Layer Normalization
                    self.mh = tf.contrib.layers.layer_norm(self.embedded_chars + self.mh)

                # Position-wise Feed Forward
                with tf.variable_scope("position-wise-feed-forward"):
                    self.ff = feedforward(self.mh, dim_model, dim_ff)
                    # Residual & Layer Normalization
                    self.enc = tf.contrib.layers.layer_norm(self.mh + self.ff)
        # self.enc_change = tf.layers.conv1d(inputs=self.enc, filters=300, kernel_size=1, activation=tf.nn.relu)
        # self.self_attn = tf.layers.dense(self.enc, units=300,
        #                                  activation=tf.nn.relu, use_bias=True,
        #                                  trainable=True, kernel_initializer=initializer())  # [800 90 300]
        print("change:", self.enc.get_shape())  # (?, 90, 300)
        # self.self_atten_change = tf.expand_dims(self.enc, -1)  # [800 90 300 1]dim_model = 1024
        # print("attention shape:", self.self_attn.get_shape) #(?, 90 ,300)
        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            print("sequence_len:",self._length(self.input_x).get_shape())
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.enc,
                                                                  sequence_length=self._length(self.input_x),
                                                                  dtype=tf.float32)

            self.rnn_outputs = tf.concat(self.rnn_outputs, axis=-1)
            # print("rnn_output_shape:", self.rnn_outputs.get_shape()) #(? 90 600)

        # Attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas = attention_output(self.rnn_outputs)

        # print("attn:", self.attn.get_shape()) #(? 600)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            print("logit shape:", self.attn.get_shape()) #(? ,600)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            print("predit shape:", self.predictions.get_shape()) #(?,)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
