import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from transformer import multihead_attention,feedforward

class TextCNN:
    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, pos_vocab_size, pos_embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0, use_elmo=False):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_x_text = tf.placeholder(tf.string, shape=[None,], name='input_x_text')
        self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal


        # Embedding layer

        if use_elmo:
            # Contextual Embedding Layer
            with tf.variable_scope("elmo-embeddings"):
                elmo_model = hub.Module("/home/wangyan/relation_extraction/ELMo", trainable=True)
                self.text_embedded_chars = elmo_model(self.input_x_text, signature="default", as_dict=True)["elmo"]
                # print(list(self.input_x_text)[:2])
                print("embedding_shape:", self.text_embedded_chars.get_shape())
                # sequence_length =self.text_embedded_chars.shape[1].value
        else:
            with tf.device('/cpu:0'), tf.variable_scope("text-embedding"):
                self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -0.25, 0.25), name="W_text")
                self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text) #[800 90 300]

        with tf.variable_scope('dropout-embeddings'):
            # self.embedded_chars_expanded = tf.concat([self.text_embedded_chars, self.pos_embedded_chars], 2)
            self.embedded_chars = tf.nn.dropout(self.text_embedded_chars, self.dropout_keep_prob)
        # self.embedded_chars = tf.layers.dense(self.embedded_chars, units=300,
        #                                  activation=tf.nn.relu, use_bias=True,
        #                                  trainable=True, kernel_initializer=initializer())
        dim_model = 100
        dim_ff = 400
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
        self.self_atten_change = tf.expand_dims(self.enc, -1)  # [800 90 300 1]dim_model = 1024

        # with tf.device('/cpu:0'), tf.variable_scope("position-embedding"):
        #     self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
        #     self.p1_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p1)
        #     self.p2_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p2)
        #     self.p1_embedded_chars_expanded = tf.expand_dims(self.p1_embedded_chars, -1) #[800 90 50 1]
        #     self.p2_embedded_chars_expanded = tf.expand_dims(self.p2_embedded_chars, -1)

        # self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
        #                                           self.p1_embedded_chars_expanded,
        #                                           self.p2_embedded_chars_expanded], 2) #[800 90 400 1]
        _embedding_size = 100
        self.text_shape=tf.shape(self.text_embedded_chars)
        self.text_expand_shape=tf.shape(self.self_atten_change)
        # self.pos_expand_shape=tf.shape(self.p1_embedded_chars_expanded)
        # self.embedd_shape=tf.shape(self.embedded_chars_expanded)
        self.embedding_size_shape=tf.shape(_embedding_size)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(self.self_atten_change, num_filters, [filter_size, _embedding_size],
                                        kernel_initializer=initializer(), activation=tf.nn.relu, padding="SAME",
                                        strides=(1, _embedding_size),
                                        name="conv")  # num_filter=128,filter_size=2,3,4,5
                print(conv.get_shape())  # (?,89,1, 128);(?88,1,128)(?87,1,128)(?86 1 128)
                # R = tf.squeeze(conv,axis=-2)
                # print(R.get_shape())

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(conv, ksize=[1, sequence_length, 1, 1],
                                        strides=[1, sequence_length, 1, 1], padding='SAME', name="pool")
                print(pooled.get_shape())  # (?, 1, 1, 128)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print(np.array(pooled_outputs).shape)#(4,)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print(self.h_pool.get_shape()) #(?,1,1,512)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(self.h_pool_flat.get_shape()) #(?,512)

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
