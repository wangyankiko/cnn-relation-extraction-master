import tensorflow as tf
import numpy as np
import os,sys
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from attention import multihead_attention,fnn

class TextCNN:
    def __init__(self, sequence_length, num_classes,pos_vocab_size,pos_embedding_size,
                  text_embedding_size,filter_sizes, num_heads, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.text_embedded_chars = tf.placeholder(tf.float32, shape=[None, 2304], name='text_embedded_chars')
        # self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        # self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.input_pos = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y') #[20 19]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        _embedding_size = text_embedding_size
        self.text_shape=tf.shape(self.text_embedded_chars)


        self.self_attn = tf.layers.dense(self.text_embedded_chars, units=300,
                                          activation=tf.nn.relu, use_bias=True,
                                          trainable=True, kernel_initializer=initializer())  # [800 90 300]
        print("change:", self.self_attn.get_shape())  # (?, 90, 300)

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.self_attn, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            print(self.logits.get_shape()) #(?,19)
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