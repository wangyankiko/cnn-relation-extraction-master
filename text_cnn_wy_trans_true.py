import tensorflow as tf
import numpy as np
import os, sys

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from attention import multihead_attention
from attention import fnn
from transformer import transformer_model


class TextCNN:
    def __init__(self, sequence_length, num_classes,
                 text_embedding_size, filter_sizes, num_heads, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.text_embedded_chars = tf.placeholder(tf.float32, shape=[None, sequence_length, 768],
                                                  name='text_embedded_chars')
        # self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        # self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')  # [20 19]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Embedding layer
        # with tf.device('/device:GPU:0'), tf.variable_scope("text-embedding"):
        #     # self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -0.25, 0.25), name="W_text")
        #     # self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text) #[800 90 300]
        #     # self.text_embedded_chars = server_bert.get_sentence_embedding(self.input_text) #[800 90 768]
        #     # self.text_embedded_chars_trans = transformer.transformerencoder(self.text_embedded_chars)
        #     self.text_embedded_chars_change = tf.layers.dense(self.text_embedded_chars, units=300,activation=tf.nn.relu,use_bias=True, trainable=True) #[800 90 300]
        #     print("change:",self.text_embedded_chars_change.get_shape())# (?, 90, 300)
        #     self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars_change, -1) #[800 90 300 1]
        #     print(self.text_embedded_chars_expanded.get_shape())

        # with tf.device('/cpu:0'), tf.variable_scope("position-embedding"):
        #     self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
        #     self.p1_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p1)
        #     self.p2_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p2)
        #     self.p1_embedded_chars_expanded = tf.expand_dims(self.p1_embedded_chars, -1) #[800 90 50 1]
        #     self.p2_embedded_chars_expanded = tf.expand_dims(self.p2_embedded_chars, -1)

        # self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
        #                                           self.p1_embedded_chars_expanded,
        #                                           self.p2_embedded_chars_expanded], 2) #[800 90 400 1]
        _embedding_size = 300
        self.text_shape = tf.shape(self.text_embedded_chars)
        # print("shape：",self.text_embedded_chars.get_shape())
        # self.text_expand_shape=tf.shape(self.text_embedded_chars_expanded)
        # self.pos_expand_shape=tf.shape(self.p1_embedded_chars_expanded)
        # self.embedd_shape=tf.shape(self.text_embedded_chars_change)
        # self.embedding_size_shape=tf.shape(_embedding_size)

        # self.text_embedded_chars_dense = tf.layers.dense(self.text_embedded_chars, units=768,kernel_initializer=initializer())

        #Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.text_embedded_chars, self.emb_dropout_keep_prob)

        # #transformer
        # with tf.variable_scope("transformer"):
        #     self.output_layer = transformer_model(self.text_embedded_chars) #[?, 100, 768]

        # self-attention
        with tf.variable_scope("self-attention"):
            self.self_attn_output, self.self_alphas = multihead_attention(self.embedded_chars, self.embedded_chars,
                                                                          num_units=768, num_heads=num_heads)
            # print("attention shape:", self.self_attn.get_shape) #(?, 90 ,300)


        with tf.variable_scope("feed_forward"):
            self.layer_output = fnn(self.self_attn_output)

        self.self_attn = tf.layers.dense(self.layer_output, units=300,
                                         kernel_initializer=initializer())  # [800 90 300]
        print("change:", self.layer_output.get_shape())  # (?, 90, 300)
        self.self_atten_change = tf.expand_dims(self.self_attn, -1)  # [800 90 300 1]
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(self.self_atten_change, num_filters, [filter_size, _embedding_size],
                                        kernel_initializer=initializer(), activation=tf.nn.relu,
                                        name="conv")  # num_filter=128,filter_size=2,3,4,5
                print(conv.get_shape())  # (?,89,1, 128);(?88,1,128)(?87,1,128)(?86 1 128)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(conv, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                print(pooled.get_shape())  # (?, 1, 1, 128)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # print(pooled_outputs.get_shape())
        print(np.array(pooled_outputs).shape)  # （4，）
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print(self.h_pool.get_shape()) #(?,1,1,512)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print(self.h_pool_flat.get_shape())#(?,512)

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            print(self.logits.get_shape())  # (?,19)
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