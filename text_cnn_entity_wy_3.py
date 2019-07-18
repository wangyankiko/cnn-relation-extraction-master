import tensorflow as tf
import numpy as np
import os,sys
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from configure import FLAGS
from transformer import multihead_attention, feedforward,attention_2,attention_3,attention_4,attention_5,attention_6,attention_7,attention_8
from attention import attention_entity

class TextCNN:
    def __init__(self, sequence_length, num_classes,pos_vocab_size,pos_embedding_size,
                  text_embedding_size,filter_sizes, num_heads, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.text_embedded_chars = tf.placeholder(tf.float32, shape=[None, sequence_length, 768], name='text_embedded_chars')
        # self.input_p1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p1')
        # self.input_p2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_p2')
        self.e1_start = tf.placeholder(tf.int32, shape=[None], name="e1_start")
        self.e2_start = tf.placeholder(tf.int32, shape=[None], name="e2_start")
        self.e1_end = tf.placeholder(tf.int32, shape=[None], name="e1_end")
        self.e2_end = tf.placeholder(tf.int32, shape=[None], name="e2_end")
        self.batch_size_len = tf.placeholder(tf.int32, name="batch_size_len")
        # self.e1_embeded = tf.placeholder(tf.float32, shape=[None, 768], name="e1_embedding")
        # self.e2_embeded = tf.placeholder(tf.float32, shape=[None, 768], name="e2_embedding")
        self.input_pos = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y') #[20 19]
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

        with  tf.variable_scope("pos-embedding"):
            self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size], initializer=initializer())
            self.pos_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_pos)
            # self.p2_embedded_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p2)
            self.pos_embedded_chars_expanded = tf.expand_dims(self.pos_embedded_chars, -1) #[800 90 50 1]
            # self.p2_embedded_chars_expanded = tf.expand_dims(self.p2_embedded_chars, -1)


        _embedding_size = 768

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            # self.embedded_chars_expanded = tf.concat([self.text_embedded_chars, self.pos_embedded_chars], 2)
            self.embedded_chars= tf.nn.dropout(self.text_embedded_chars, self.emb_dropout_keep_prob)

        dim_model = 768
        dim_ff = 3072
        num_stack = 1
        ##transformer
        for i in range(num_stack):
            with tf.variable_scope("block-{}".format(i)):
                # Multi-head Attention (self attention)
                with tf.variable_scope("multihead-attention"):
                    self.mh = multihead_attention(query=self.embedded_chars, key=self.embedded_chars, value=self.embedded_chars,
                                                       dim_model=dim_model, num_head=num_heads)
                    # Residual & Layer Normalization
                    self.mh = tf.contrib.layers.layer_norm(self.embedded_chars + self.mh)

                # Position-wise Feed Forward
                with tf.variable_scope("position-wise-feed-forward"):
                    self.ff = feedforward(self.mh, dim_model, dim_ff)
                    # Residual & Layer Normalization
                    self.enc = tf.contrib.layers.layer_norm(self.mh + self.ff)



        print(self.enc.shape[0].value)
        print(self.batch_size_len)
        m = tf.constant(value=32,dtype=tf.int32)
        print(m)
        print(tf.shape(self.enc)[0])
        print(FLAGS.batch_size_len)
        print("e1_start:",self.e1_start.get_shape())
        def true_func():
            e1_embedd = tf.expand_dims(tf.reduce_mean(self.enc[0][self.e1_start[0]:self.e1_end[0] + 1], axis=0), axis=0)
            e2_embedd = tf.expand_dims(tf.reduce_mean(self.enc[0][self.e2_start[0]:self.e2_end[0] + 1], axis=0), axis=0)
            print("e1 shape", e1_embedd.shape)
            for i in range(1, 32):
                e1 = tf.expand_dims(tf.reduce_mean(self.enc[i][self.e1_start[i]:self.e1_end[i] + 1], axis=0), axis=0)
                e2 = tf.expand_dims(tf.reduce_mean(self.enc[i][self.e2_start[i]:self.e2_end[i] + 1], axis=0), axis=0)
                e1_embedd =tf.concat([e1_embedd, e1],axis=0)
                e2_embedd = tf.concat([e2_embedd, e2],axis=0)
            print("embed shape:",e1_embedd.shape)
            # e1_embedd = tf.nn.relu(e1_embedd)
            # e2_embedd = tf.nn.relu(e2_embedd)
            print("embed shape:", e1_embedd.shape)
            return e1_embedd,e2_embedd

        def false_func():
            e1_embedd = tf.expand_dims(tf.reduce_mean(self.enc[0][self.e1_start[0]:self.e1_end[0] + 1], axis=0), axis=0)
            e2_embedd = tf.expand_dims(tf.reduce_mean(self.enc[0][self.e2_start[0]:self.e2_end[0] + 1], axis=0), axis=0)
            print("e1 shape", e1_embedd.shape)
            for i in range(1, 29):
                e1 = tf.expand_dims(tf.reduce_mean(self.enc[i][self.e1_start[i]:self.e1_end[i] + 1], axis=0), axis=0)
                # print(self.enc.shape[0].value)
                # print("e1 shape",e1.shape)
                # print("i:",FLAGS.batch_size_len)
                e2 = tf.expand_dims(tf.reduce_mean(self.enc[i][self.e2_start[i]:self.e2_end[i] + 1], axis=0), axis=0)
                e1_embedd = tf.concat([e1_embedd, e1], axis=0)
                e2_embedd = tf.concat([e2_embedd, e2], axis=0)
            print("embed shape 1:", e1_embedd.shape)
            # e1_embedd = tf.nn.relu(e1_embedd)
            # e2_embedd = tf.nn.relu(e2_embedd)
            print("embed shape 1:", e1_embedd.shape)
            return e1_embedd, e2_embedd
        e1_embedd_2, e2_embedd_2 = tf.cond(pred=tf.equal(m, self.batch_size_len), true_fn=true_func, false_fn=false_func)
        print("embed shape 2:", e1_embedd_2.shape)
        # def extract_entity(x, e):
        #     e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        #     return tf.gather_nd(x, e_idx)
        #
        # # self.enc_dense = tf.layers.dense(self.enc, units=128, activation=tf.nn.relu, kernel_initializer=initializer())
        # e1_end = extract_entity(self.enc, self.e1_end)  # (batch, hidden)
        # e1_start = extract_entity(self.enc, self.e1_start)
        # print("e1_h", e1_end.get_shape())
        # e2_end = extract_entity(self.enc, self.e2_end)  # (batch, hidden)
        # e2_start = extract_entity(self.enc, self.e2_start)
        # e1_h_add = (e1_start+e1_end)/2.0
        # e2_h_add = (e2_start+e2_end)/2.0
        # alpha = attention_4(self.enc , e1_h, e2_h)
        # self.enc_attention = tf.reduce_sum(tf.multiply(self.enc, tf.expand_dims(alpha, -1)), axis=1)
        # self.enc_attention_h = tf.expand_dims(self.enc_attention,1)
        # e1_h = tf.expand_dims(e1_h,1)
        # e2_h = tf.expand_dims(e2_h,1)

        # print(e1_h.get_shape())
        # e1_end_h = tf.expand_dims(e1_end, 1)
        # e2_end_h = tf.expand_dims(e2_end,1)
        # e1_start_h = tf.expand_dims(e1_start, 1)
        # e2_start_h = tf.expand_dims(e2_start, 1)
        # print(e1_end_h.get_shape())
        # e1_h =  (e1_start_h + e1_end_h)/2.0
        # e2_h =  (e2_start_h + e2_end_h)/2.0
        # e1_h = tf.reshape(tf.tile(e1_h_add, [1, sequence_length]), [-1, sequence_length, 128])  # (batch, seq_len, hidden_size)
        # e2_h = tf.reshape(tf.tile(e2_h_add, [1, sequence_length]), [-1, sequence_length, 128])  # (batch, seq_len, hidden_size)
        # e_h = (e1_h + e2_h)/2.0
        e1_h = tf.expand_dims(e1_embedd_2,axis=1)
        e2_h = tf.expand_dims(e2_embedd_2,axis=1)
        input_e1 = tf.concat([self.enc, e1_h, e2_h], axis=1)
        print(input_e1.get_shape())
        self.self_atten_change = tf.expand_dims(input_e1, -1)  # [800 90 300 1]
        # self.self_atten_change = tf.expand_dims(self.enc,-1)
        self.enc_dense = tf.layers.dense(self.enc, units=128, activation=tf.nn.relu)

        def true_func_2():
            e1_embedd_dense = tf.expand_dims(tf.reduce_mean(self.enc_dense[0][self.e1_start[0]:self.e1_end[0] + 1], axis=0), axis=0)
            e2_embedd_dense = tf.expand_dims(tf.reduce_mean(self.enc_dense[0][self.e2_start[0]:self.e2_end[0] + 1], axis=0), axis=0)
            print("e1 shape", e1_embedd_dense.shape)
            for i in range(1, 32):
                e1 = tf.expand_dims(tf.reduce_mean(self.enc_dense[i][self.e1_start[i]:self.e1_end[i] + 1], axis=0), axis=0)
                e2 = tf.expand_dims(tf.reduce_mean(self.enc_dense[i][self.e2_start[i]:self.e2_end[i] + 1], axis=0), axis=0)
                e1_embedd_dense = tf.concat([e1_embedd_dense, e1], axis=0)
                e2_embedd_dense = tf.concat([e2_embedd_dense, e2], axis=0)
            print("embed shape:", e1_embedd_dense.shape)
            return e1_embedd_dense, e2_embedd_dense

        def false_func_2():
            e1_embedd_dense = tf.expand_dims(
                tf.reduce_mean(self.enc_dense[0][self.e1_start[0]:self.e1_end[0] + 1], axis=0), axis=0)
            e2_embedd_dense = tf.expand_dims(
                tf.reduce_mean(self.enc_dense[0][self.e2_start[0]:self.e2_end[0] + 1], axis=0), axis=0)
            print("e1 shape", e1_embedd_dense.shape)
            for i in range(1, 29):
                e1 = tf.expand_dims(tf.reduce_mean(self.enc_dense[i][self.e1_start[i]:self.e1_end[i] + 1], axis=0),
                                    axis=0)
                e2 = tf.expand_dims(tf.reduce_mean(self.enc_dense[i][self.e2_start[i]:self.e2_end[i] + 1], axis=0),
                                    axis=0)
                e1_embedd_dense = tf.concat([e1_embedd_dense, e1], axis=0)
                e2_embedd_dense = tf.concat([e2_embedd_dense, e2], axis=0)
            print("embed shape:", e1_embedd_dense.shape)

            return e1_embedd_dense, e2_embedd_dense

        e1_embedd_dense_2, e2_embedd_dense_2 = tf.cond(pred=tf.equal(m, self.batch_size_len), true_fn=true_func_2,
                                           false_fn=false_func_2)
        print("dense:", e1_embedd_dense_2.get_shape())
        alpha = attention_7(self.enc_dense, e1_embedd_dense_2, e2_embedd_dense_2)
        self.enc_attention_1 = tf.reduce_sum(tf.multiply(self.enc_dense, tf.expand_dims(alpha, -1)), axis=1)

        # self.enc_attention_dense = tf.layers.dense(self.enc_attention_1, units=128,
        #                                            activation=tf.nn.relu)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = tf.layers.conv2d(self.self_atten_change, num_filters, [filter_size, _embedding_size],
                                        activation=tf.nn.relu, padding="SAME",
                                        strides=(1, _embedding_size),
                                        name="conv")  # num_filter=128,filter_size=2,3,4,5
                print(conv.get_shape())  # (?,89,1, 128);(?88,1,128)(?87,1,128)(?86 1 128)

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(conv, ksize=[1, sequence_length+2, 1, 1],
                                        strides=[1, sequence_length+2, 1, 1], padding='SAME', name="pool")
                print(pooled.get_shape())  # (?, 1, 1, 128)
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # print(pooled_outputs.get_shape())
        print(np.array(pooled_outputs).shape) #（4，）
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print(self.h_pool.get_shape()) #(?,1,1,512)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_flat_2 = tf.concat([self.h_pool_flat, self.enc_attention_1], axis=-1)
        print(self.h_pool_flat.get_shape())#(?,512)
        # print(self.h_pool_flat_2.get_shape())
        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat_2, self.dropout_keep_prob)
        print(self.h_drop.get_shape())
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