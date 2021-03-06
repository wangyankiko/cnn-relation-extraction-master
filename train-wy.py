import tensorflow as tf
import numpy as np
import os,sys
import datetime
import time
import server_bert
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from text_cnn_wy import TextCNN
import data_helpers
import utils
from configure import FLAGS

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/device:GPU:0'):
        x_text, y, pos1, pos2, x_text_clean, sentence_len = data_helpers.load_data_and_labels(FLAGS.train_path)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    # print("text:",x_text)
    # text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # x = np.array(list(text_vocab_processor.fit_transform(x_text)))#token
    # pretrain_W = utils.load_word2vec(FLAGS.embedding_path, FLAGS.text_embedding_dim, text_vocab_processor)
    # print("pretrain_w:",pretrain_W)
    # print(pretrain_W.shape) #(19151,300)
    # print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))
    # print("vocabulary:", text_vocab_processor.vocabulary_._reverse_mapping)
    # with open("vocabulary.txt","w",encoding="utf-8") as f:
    #     f.write(str(x))
    # print("x = {0}".format(x.shape)) #（8000，90）
    # print("y = {0}".format(y.shape)) #（8000，19）
    # print("")

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    # pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # pos_vocab_processor.fit(pos1 + pos2) #fit
    # print("pos vocab position:", pos_vocab_processor)
    # p1 = np.array(list(pos_vocab_processor.transform(pos1))) #tokens
    # print("p1:", p1)
    # p2 = np.array(list(pos_vocab_processor.transform(pos2)))
    # print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    # with open("position.txt", "w", encoding="utf-8") as f:
    #         f.write(str(x))
    # print("position_1 = {0}".format(p1.shape)) #(8000,90)
    # print("position_2 = {0}".format(p2.shape)) #(8000,90)
    # print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    x_text_to_id = {}
    id_to_x_text = {}
    id = []
    for i, str1 in enumerate(x_text_clean):
        x_text_to_id[str1]=i
        id_to_x_text[i] = str1
        id.append(i)
    # print(x_text_to_id)
    # print(id_to_x_text)
    # print(id[0:3])
    print("id:",id)

    shuffle_indices = np.random.permutation(np.arange(len(y))) #len(y)=8000
    id_shuffled = np.array(id)[shuffle_indices]

    # # p1_shuffled = p1[shuffle_indices]
    # # p2_shuffled = p2[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # print(x_shuffled, p1_shuffled,p2_shuffled,y_shuffled)

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) #x_train=7200, x_dev =800
    id_train, id_dev = id_shuffled[:dev_sample_index], id_shuffled[dev_sample_index:]
    # p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    # p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    # x_train = [id_to_x_text[i] for i in id_train]
    # x_dev = [id_to_x_text[i] for i in id_dev]
    print("id_train:", id_train)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.max_sentence_length, #90
                num_classes=y_train.shape[1],#19
                text_embedding_size=FLAGS.text_embedding_dim,#300
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), #2,3,4,5
                num_filters=FLAGS.num_filters, #128
                l2_reg_lambda=FLAGS.l2_reg_lambda) #1e-5

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(cnn.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
            # pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # FLAGS._sess =sess

            # Pre-trained word2vec
            # if FLAGS.embedding_path:
            #     pretrain_W = utils.load_word2vec(FLAGS.embedding_path, FLAGS.text_embedding_dim, text_vocab_processor)
            #     sess.run(cnn.W_text.assign(pretrain_W))
            #     print("Success to load pre-trained word2vec model!\n")


            # Generate batches
            batches = data_helpers.batch_iter(list(zip(id_train, y_train)),
                                              FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            text_embedded_chars_dev = server_bert.load_vector("embedding.npy", list(id_dev))
            # print("id_dev:",id_dev)
            # print(text_embedded_chars_dev.shape) #(800 90 768)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # print(x_batch)
                # print(list(x_batch))
                # print(len(x_batch)) #20
                # print(len(y_batch)) #20

                # Train
                text_embedded_chars = server_bert.load_vector("embedding.npy", list(x_batch)) #[20 90 768]
                #print(text_embedded_chars.shape) #（20 90 768）
                feed_dict = {
                    cnn.text_embedded_chars: text_embedded_chars,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        cnn.text_embedded_chars: text_embedded_chars_dev,
                        cnn.input_y: y_dev,
                        cnn.dropout_keep_prob: 1.0
                    }

                    summaries, loss, accuracy, predictions,text_expand_shape,embedding_shape, text_shape= sess.run(
                        [dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.text_expand_shape, cnn.embedd_shape,cnn.text_shape], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))
                    # print("text_embedded_shape:", text_shape)
                    # print("text_embedd_extend:", text_expand_shape)
                    # # print("pos-embedd_extend:", pos_shape)
                    # print("embedding_size:", embedding_shape)
                    # print("embedding_size_shape", embedding_size_shape)

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
