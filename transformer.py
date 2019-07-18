import tensorflow as tf

initializer = tf.keras.initializers.glorot_normal
def multihead_attention(query, key, value, dim_model, num_head, masked=False):
    attentions = []
    for i in range(num_head):
        dim_k = int(dim_model / num_head)
        dim_v = dim_k

        Q = tf.layers.dense(query, dim_k, activation=tf.nn.relu )
        K = tf.layers.dense(key, dim_k, activation=tf.nn.relu )
        V = tf.layers.dense(value, dim_v, activation=tf.nn.relu )

        # Scaled Dot Product Attention
        with tf.variable_scope("scaled-dot-product-attention"):
            QK_T = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            if masked:
                mask = tf.ones_like(QK_T)
                mask = tf.contrib.linalg.LinearOperatorTriL(mask, tf.float32).to_dense()
                # Tensorflow >= 1.5.0
                # mask = tf.linalg.LinearOperatorLowerTriangular(mask, tf.float32).to_dense()
                QK_T = tf.matmul(QK_T, mask)
            attention = tf.nn.softmax(QK_T * tf.sqrt(1 / dim_k))
            att_V = tf.matmul(attention, V)

        attentions.append(att_V)

    att_concat = tf.concat(attentions, axis=2)
    output = tf.layers.dense(att_concat, dim_model, activation=tf.nn.relu)

    return output


def feedforward(x, dim_model, dim_ff):
    # First Convolution
    output = tf.layers.conv1d(inputs=x, filters=dim_ff, kernel_size=1, activation=tf.nn.relu)
    # Second Convolution
    output = tf.layers.conv1d(inputs=output, filters=dim_model, kernel_size=1, activation=tf.nn.relu )

    return output


def  attention(x):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    n = x.shape[1].value
    size = x.shape[2].value
    print(n)
    # x_r = tf.reshape(x,[n,None,size])
    # print(x_r.get_shape())
    print(x.get_shape())
    b= tf.split(x,n,1)
    # print(b.get_shape())
    e1 = tf.squeeze(b[-2], axis=1)
    e2 = tf.squeeze(b[-1], axis=1)
    print("e1",e1.get_shape())
    print("e1", e2.get_shape())
    with tf.name_scope('input_attention'):
        A1 = tf.matmul(x, tf.expand_dims(e1, axis=-1))  # bz, n, 1
        print(A1.get_shape())
        A2 = tf.matmul(x, tf.expand_dims(e2, axis=-1))
        A1 = tf.squeeze(A1,axis=-1)
        A2 = tf.squeeze(A2,axis=-1)
        alpha1 = tf.nn.softmax(A1)  # bz, n
        alpha2 = tf.nn.softmax(A2)  # bz, n
        alpha = (alpha1 + alpha2) / 2
        print(alpha.get_shape())
        # output = x * tf.expand_dims(alpha, -1)
        return alpha

def  attention_2(x,e1,e2):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    n = x.shape[1].value
    size = x.shape[2].value
    print(n)
    def extract_entity(x, e):
        e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        print("e_idx_shape:", e_idx.get_shape())
        return tf.gather_nd(x, e_idx)  # (batch, hidden)

    e1_h = extract_entity(x, e1)  # (batch, hidden)
    print("e1_h",e1_h.get_shape())
    e2_h = extract_entity(x, e2)  # (batch, hidden)
    # x_r = tf.reshape(x,[n,None,size])
    # print(x_r.get_shape())
    # print(x.get_shape())
    # b= tf.split(x,n,1)
    # # print(b.get_shape())
    # e1 = tf.squeeze(b[-2], axis=1)
    # e2 = tf.squeeze(b[-1], axis=1)
    # print("e1",e1.get_shape())
    # print("e1", e2.get_shape())
    with tf.name_scope('input_attention'):
        # input_e1 = tf.add(x, e1_h)
        # input_e2 =  tf.add(x, e2_h)
        A1 = tf.matmul(x, tf.expand_dims(e1_h, axis=-1))  # bz, n, 1
        print(A1.get_shape())
        A2 = tf.matmul(x, tf.expand_dims(e2_h, axis=-1))
        A1 = tf.squeeze(A1,axis=-1)
        A2 = tf.squeeze(A2,axis=-1)
        alpha1 = tf.nn.softmax(A1)  # bz, n
        alpha2 = tf.nn.softmax(A2)  # bz, n
        alpha = (alpha1 + alpha2) / 2.0
        print(alpha.get_shape())
        # output = x * tf.expand_dims(alpha, -1)
        return alpha

def  attention_3(x,e1,e2):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    seq_len= x.shape[1].value
    size = x.shape[2].value

    # print(n)
    # x_r = tf.reshape(x,[n,None,size])
    # print(x_r.get_shape())
    # print(x.get_shape())
    # b= tf.split(x,n,1)
    # # print(b.get_shape())
    # e1 = tf.squeeze(b[-2], axis=1)
    # e2 = tf.squeeze(b[-1], axis=1)
    # print("e1",e1.get_shape())
    # print("e1", e2.get_shape())
    with tf.name_scope('input_attention'):
        def extract_entity(x, e):
            e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
            return tf.gather_nd(x, e_idx)  # (batch, hidden)

        e1 = extract_entity(x, e1)  # (batch, hidden)
        print("e1_h", e1.get_shape())
        e2 = extract_entity(x, e2)  # (batch, hidden)
        e1_h= tf.reshape(tf.tile(e1, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        e2_h = tf.reshape(tf.tile(e2, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        input_e1 = tf.concat([x, e1_h], axis=-1)
        input_e2 = tf.concat([x, e2_h], axis=-1)
        print(input_e1.get_shape())
        input_e1 = tf.layers.dense(input_e1, units=size,kernel_initializer=initializer())
        input_e2 = tf.layers.dense(input_e2, units=size,kernel_initializer=initializer())
        # A1 = tf.matmul(input_e1, tf.expand_dims(e1, axis=-1))  # bz, n, 1
        # print(A1.get_shape())
        # A2 = tf.matmul(input_e2, tf.expand_dims(e2, axis=-1))
        # A1 = tf.squeeze(A1,axis=-1)
        # A2 = tf.squeeze(A2,axis=-1)
        # alpha1 = tf.nn.softmax(A1)  # bz, n
        # alpha2 = tf.nn.softmax(A2)  # bz, n
        u1 = tf.get_variable("u1_var", [size], initializer=tf.keras.initializers.glorot_normal())
        u2 = tf.get_variable("u2_var", [size], initializer=tf.keras.initializers.glorot_normal())
        vu_1 = tf.tensordot(input_e1, u1, axes=1, name='vu_1')  # (B,T) shape
        print("vu_1",vu_1.get_shape())
        vu_2 = tf.tensordot(input_e2, u2, axes=1, name='vu_2')  # (B,T) shape
        # softmax
        alpha1 = tf.nn.softmax(vu_1, name='alphas_1')  # (B,T) shape
        alpha2 = tf.nn.softmax(vu_2, name='alphas_2')  # (B,T) shape
        alpha = (alpha1 + alpha2) / 2.0
        # print(alpha.get_shape())
        # output = tf.reduce_sum(x * tf.expand_dims(alpha, -1), 1)
        return alpha


def  attention_4(x,e1,e2):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    seq_len= x.shape[1].value
    size = x.shape[2].value

    # print(n)
    # x_r = tf.reshape(x,[n,None,size])
    # print(x_r.get_shape())
    # print(x.get_shape())
    # b= tf.split(x,n,1)
    # # print(b.get_shape())
    # e1 = tf.squeeze(b[-2], axis=1)
    # e2 = tf.squeeze(b[-1], axis=1)
    # print("e1",e1.get_shape())
    # print("e1", e2.get_shape())
    with tf.name_scope('input_attention'):
        # input_e1 = tf.add(x, e1_h)
        # input_e2 =  tf.add(x, e2_h)
        A1 = tf.matmul(x, tf.expand_dims(e1, axis=-1))  # bz, n, 1
        print(A1.get_shape())
        A2 = tf.matmul(x, tf.expand_dims(e2, axis=-1))
        A1 = tf.squeeze(A1,axis=-1)
        A2 = tf.squeeze(A2,axis=-1)
        alpha1 = tf.nn.softmax(A1)  # bz, n
        alpha2 = tf.nn.softmax(A2)  # bz, n
        alpha = (alpha1 + alpha2) / 2.0
        print(alpha.get_shape())
        # output = x * tf.expand_dims(alpha, -1)
        return alpha
def attention_5(x,e1,e2):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    seq_len = x.shape[1].value
    size = x.shape[2].value
    with tf.name_scope('input_attention'):

        e1_h= tf.reshape(tf.tile(e1, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        e2_h = tf.reshape(tf.tile(e2, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        input_e1 = tf.concat([x, e1_h], axis=-1)
        input_e2 = tf.concat([x, e2_h], axis=-1)
        print(input_e1.get_shape())
        input_e1 = tf.layers.dense(input_e1, units=size,kernel_initializer=initializer())
        input_e2 = tf.layers.dense(input_e2, units=size,kernel_initializer=initializer())
        u1 = tf.get_variable("u1_var", [size], initializer=tf.keras.initializers.glorot_normal())
        u2 = tf.get_variable("u2_var", [size], initializer=tf.keras.initializers.glorot_normal())
        vu_1 = tf.tensordot(input_e1, u1, axes=1, name='vu_1')  # (B,T) shape
        print("vu_1",vu_1.get_shape())
        vu_2 = tf.tensordot(input_e2, u2, axes=1, name='vu_2')  # (B,T) shape
        # softmax
        alpha1 = tf.nn.softmax(vu_1, name='alphas_1')  # (B,T) shape
        alpha2 = tf.nn.softmax(vu_2, name='alphas_2')  # (B,T) shape
        alpha = (alpha1 + alpha2) / 2.0
        # print(alpha.get_shape())
        # output = tf.reduce_sum(x * tf.expand_dims(alpha, -1), 1)
        return alpha
def  attention_6(x,e1_start,e1_end,e2_start,e2_end):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    n = x.shape[1].value
    size = x.shape[2].value
    print(n)
    def extract_entity(x, e):
        e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        print("e_idx_shape:", e_idx.get_shape())
        return tf.gather_nd(x, e_idx)  # (batch, hidden)

    e1_start = extract_entity(x, e1_start)  # (batch, hidden)
    print("e1_h",e1_start.get_shape())
    e1_end = extract_entity(x, e1_end)
    e2_start = extract_entity(x, e2_start)  # (batch, hidden)
    e2_end = extract_entity(x, e2_end)
    e1_h = (e1_start+e1_end)/2.0
    e2_h = (e2_start + e2_end)/2.0
    # x_r = tf.reshape(x,[n,None,size])
    # print(x_r.get_shape())
    # print(x.get_shape())
    # b= tf.split(x,n,1)
    # # print(b.get_shape())
    # e1 = tf.squeeze(b[-2], axis=1)
    # e2 = tf.squeeze(b[-1], axis=1)
    # print("e1",e1.get_shape())
    # print("e1", e2.get_shape())
    with tf.name_scope('input_attention'):
        # input_e1 = tf.add(x, e1_h)
        # input_e2 =  tf.add(x, e2_h)
        A1 = tf.matmul(x, tf.expand_dims(e1_h, axis=-1))  # bz, n, 1
        print(A1.get_shape())
        A2 = tf.matmul(x, tf.expand_dims(e2_h, axis=-1))
        A1 = tf.squeeze(A1,axis=-1)
        A2 = tf.squeeze(A2,axis=-1)
        alpha1 = tf.nn.softmax(A1)  # bz, n
        alpha2 = tf.nn.softmax(A2)  # bz, n
        alpha = (alpha1 + alpha2) / 2.0
        print(alpha.get_shape())
        # output = x * tf.expand_dims(alpha, -1)
        return alpha
def attention_7(x, e1_h, e2_h):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    seq_len= x.shape[1].value
    size = x.shape[2].value


    # def extract_entity(x, e):
    #     e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
    #     print("e_idx_shape:", e_idx.get_shape())
    #     return tf.gather_nd(x, e_idx)  # (batch, hidden)
    #
    # e1_start = extract_entity(x, e1_start)  # (batch, hidden)
    # print("e1_h", e1_start.get_shape())
    # e1_end = extract_entity(x, e1_end)
    # e2_start = extract_entity(x, e2_start)  # (batch, hidden)
    # e2_end = extract_entity(x, e2_end)
    # e1_h = (e1_start + e1_end) / 2.0
    # e2_h = (e2_start + e2_end) / 2.0
    with tf.name_scope('input_attention'):

        e1_h= tf.reshape(tf.tile(e1_h, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        print("e1_h", e1_h.get_shape())
        e2_h = tf.reshape(tf.tile(e2_h, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        input_e1 = tf.concat([x, e1_h], axis=-1)
        input_e2 = tf.concat([x, e2_h], axis=-1)
        # input_e1 = tf.add(x, e1_h)
        # input_e2 = tf.add(x, e2_h)
        print(input_e1.get_shape())
        input_e1 = tf.layers.dense(input_e1, units=size, activation=tf.nn.relu)
        input_e2 = tf.layers.dense(input_e2, units=size, activation=tf.nn.relu)
        u1 = tf.get_variable("u1_var", [size])
        u2 = tf.get_variable("u2_var", [size])
        vu_1 = tf.tensordot(input_e1, u1, axes=1, name='vu_1')  # (B,T) shape
        print("vu_1",vu_1.get_shape())
        vu_2 = tf.tensordot(input_e2, u2, axes=1, name='vu_2')  # (B,T) shape
        # softmax
        alpha1 = tf.nn.softmax(vu_1, name='alphas_1')  # (B,T) shape
        alpha2 = tf.nn.softmax(vu_2, name='alphas_2')  # (B,T) shape
        alpha = (alpha1 + alpha2) / 2.0
        # print(alpha.get_shape())
        # output = tf.reduce_sum(x * tf.expand_dims(alpha, -1), 1)
        return alpha
def attention_8(x, e1_start, e1_end,e2_start,e2_end):
    bz = tf.shape(x)[1]
    # print(bz.get_shape())
    seq_len= x.shape[1].value
    size = x.shape[2].value


    def extract_entity(x, e):
        e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        print("e_idx_shape:", e_idx.get_shape())
        return tf.gather_nd(x, e_idx)  # (batch, hidden)

    e1_start = extract_entity(x, e1_start)  # (batch, hidden)
    print("e1_h", e1_start.get_shape())
    e1_end = extract_entity(x, e1_end)
    e2_start = extract_entity(x, e2_start)  # (batch, hidden)
    e2_end = extract_entity(x, e2_end)
    e1_h = (e1_start + e1_end) / 2.0
    e2_h = (e2_start + e2_end) / 2.0
    with tf.name_scope('input_attention'):

        e1_h= tf.reshape(tf.tile(e1_h, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        print("e1_h", e1_h.get_shape())
        e2_h = tf.reshape(tf.tile(e2_h, [1, seq_len]), [-1, seq_len, size])  # (batch, seq_len, hidden_size)
        input_e1 = tf.concat([x, e1_h], axis=-1)
        input_e2 = tf.concat([x, e2_h], axis=-1)
        # input_e1 = tf.add(x, e1_h)
        # input_e2 = tf.add(x, e2_h)
        print(input_e1.get_shape())
        input_e1 = tf.layers.dense(input_e1, units=size, activation=tf.nn.relu, kernel_initializer=initializer())
        input_e2 = tf.layers.dense(input_e2, units=size, activation=tf.nn.relu, kernel_initializer=initializer())
        u1 = tf.get_variable("u1_var", [size], initializer=tf.keras.initializers.glorot_normal())
        u2 = tf.get_variable("u2_var", [size], initializer=tf.keras.initializers.glorot_normal())
        vu_1 = tf.tensordot(input_e1, u1, axes=1, name='vu_1')  # (B,T) shape
        print("vu_1",vu_1.get_shape())
        vu_2 = tf.tensordot(input_e2, u2, axes=1, name='vu_2')  # (B,T) shape
        # softmax
        alpha1 = tf.nn.softmax(vu_1, name='alphas_1')  # (B,T) shape
        alpha2 = tf.nn.softmax(vu_2, name='alphas_2')  # (B,T) shape
        alpha = (alpha1 + alpha2) / 2.0
        # print(alpha.get_shape())
        # output = tf.reduce_sum(x * tf.expand_dims(alpha, -1), 1)
        return alpha
def attention_pooling(R,relation_embedd,seq_len):

    with tf.name_scope('input_attention'):
        A1 = tf.matmul(R, tf.expand_dims(relation_embedd, axis=-1))  # bz, n, 1
        print(A1.get_shape())
        # A2 = tf.matmul(x, tf.expand_dims(e2, axis=-1))
        # A1 = tf.squeeze(A1,axis=-1)
        # A2 = tf.squeeze(A2,axis=-1)
        alpha1 = tf.nn.softmax(A1)  # bz, n
        print("alpha_shape",alpha1.get_shape())
        wo = tf.multiply(R, alpha1)
        print("wo:",wo.get_shape())
        wo = tf.nn.max_pool(tf.expand_dims(wo, -1),
                            ksize=[1,seq_len, 1, 1],
                            strides=[1,seq_len, 1, 1],
                            padding="SAME"
                            )  # (bz, dc, 1, 1)
        # alpha2 = tf.nn.softmax(A2)  # bz, n
        # alpha = (alpha1 + alpha2) / 2
        # print(alpha.get_shape())
        # output = x * tf.expand_dims(alpha, -1)
        print("wo:",wo.get_shape())
        # wo_out = tf.squeeze(wo,axis=-1)
        # wo_out = tf.squeeze(wo_out,axis=1)
        print("wo_out", wo.get_shape())
        return wo

def attention_pool(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    wo = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    # wo = tf.multiply(inputs, tf.expand_dims(alphas, -1))
    print("wo:", wo.get_shape())
    # wo = tf.nn.max_pool(tf.expand_dims(wo, -1),
    #                     ksize=[1, seq_len, 1, 1],
    #                     strides=[1, seq_len, 1, 1],
    #                     padding="SAME"
    #                     )  # (bz, dc, 1, 1)
    # pooled = tf.nn.max_pool(tf.expand_dims(wo, -1), ksize=[1, seq_len - filter_size + 1, 1, 1],
    #                         strides=[1, 1, 1, 1], padding='VALID', name="pool")
    # alpha2 = tf.nn.softmax(A2)  # bz, n
    # alpha = (alpha1 + alpha2) / 2
    # print(alpha.get_shape())
    # output = x * tf.expand_dims(alpha, -1)
    print("wo:", wo.get_shape())
    # wo_out = tf.squeeze(wo,axis=-1)
    # wo_out = tf.squeeze(wo_out,axis=1)
    print("wo_out", wo.get_shape())
    # wo = tf.tanh(wo)
    return wo
