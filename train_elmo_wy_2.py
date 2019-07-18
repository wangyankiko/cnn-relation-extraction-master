import tensorflow as tf
import numpy as np
import os,sys
import datetime
import time
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from text_cnn_elmo import TextCNN
import data_helpers
import utils
from configure import FLAGS
from logger import Logger
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/cpu:0'):
        train_text, train_y, train_pos1, train_pos2, train_x_text_clean, train_sentence_len = data_helpers.load_data_and_labels(FLAGS.train_path)
    with tf.device('/cpu:0'):
        test_text, test_y, test_pos1, test_pos2, test_x_text_clean, test_sentence_len = data_helpers.load_data_and_labels(FLAGS.test_path)
    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    # print("text:",x_text)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vocab_processor.fit(train_text + test_text)
    train_x = np.array(list(vocab_processor.transform(train_text)))
    test_x = np.array(list(vocab_processor.transform(test_text)))
    train_text = np.array(train_text)
    print("train_text",train_text[0:2])
    test_text = np.array(test_text)
    print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train_x = {0}".format(train_x.shape))  # (8000,90)
    print("train_y = {0}".format(train_y.shape))  # (8000,19)
    print("test_x = {0}".format(test_x.shape))  # (2717, 90)
    print("test_y = {0}".format(test_y.shape))  # (2717,19)

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    pos_vocab_processor.fit(train_pos1 + train_pos2 + test_pos1 + test_pos2)
    train_p1 = np.array(list(pos_vocab_processor.transform(train_pos1)))
    train_p2 = np.array(list(pos_vocab_processor.transform(train_pos2)))
    test_p1 = np.array(list(pos_vocab_processor.transform(test_pos1)))
    test_p2 = np.array(list(pos_vocab_processor.transform(test_pos2)))
    print("\nPosition Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("train_p1 = {0}".format(train_p1.shape))  # (8000, 90)
    print("test_p1 = {0}".format(test_p1.shape))  # (2717, 90)
    print("")

    # Randomly shuffle data to split into train and test(dev)
    # np.random.seed(10)
    #
    # shuffle_indices = np.random.permutation(np.arange(len(y))) #len(y)=8000
    # x_shuffled = x[shuffle_indices]
    # p1_shuffled = p1[shuffle_indices]
    # p2_shuffled = p2[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    # print(x_shuffled, p1_shuffled,p2_shuffled,y_shuffled)

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) #x_train=7200, x_dev =800
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    # p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    # print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    # print(x_train)
    # print(np.array(x_train))
    # print(x_dev)
    # print(np.array(x_dev))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.max_sentence_length, #90
                num_classes=train_y.shape[1],#19
                text_vocab_size=len(vocab_processor.vocabulary_), #19151
                text_embedding_size=FLAGS.text_embedding_size,#300
                pos_vocab_size=len(pos_vocab_processor.vocabulary_),#162
                pos_embedding_size=FLAGS.pos_embedding_dim,#50
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), #2,3,4,5
                num_filters=FLAGS.num_filters, #128
                l2_reg_lambda=FLAGS.l2_reg_lambda, #1e-5
                use_elmo = (FLAGS.embeddings == 'elmo'))

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(cnn.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("\nWriting to {}\n".format(out_dir))

            # Logger
            logger = Logger(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.embeddings == "word2vec":
                pretrain_W = utils.load_word2vec('resource/GoogleNews-vectors-negative300.bin', FLAGS.embedding_size,vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")
            elif FLAGS.embeddings == "glove100":
                pretrain_W = utils.load_glove('resource/glove.6B.100d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove100 model!\n")
            elif FLAGS.embeddings == "glove300":
                pretrain_W = utils.load_glove('resource/glove.840B.300d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove300 model!\n")

            # Generate batches
            train_batches = data_helpers.batch_iter(list(zip(train_x, train_y, train_text,
                                                             train_p1, train_p2)),
                                                    FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for train_batch in train_batches:
                train_bx, train_by, train_btxt,train_bp1, train_bp2 = zip(*train_batch)
                # print("train_bxt",list(train_btxt)[:2])
                # print(np.array(train_be1).shape) #(20, )
                # print(train_be1)
                feed_dict = {
                    cnn.input_text: train_bx,
                    cnn.input_y: train_by,
                    cnn.input_x_text: list(train_btxt),
                    cnn.input_p1: train_bp1,
                    cnn.input_p2: train_bp2,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    logger.logging_train(step, loss, accuracy)

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    test_batches = data_helpers.batch_iter(list(zip(test_x, test_y, test_text,
                                                                     test_p1, test_p2)),
                                                           FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    iter_cnt = 0
                    for test_batch in test_batches:
                        test_bx, test_by, test_btxt, test_bp1, test_bp2 = zip(*test_batch)
                        feed_dict = {
                            cnn.input_text: test_bx,
                            cnn.input_y: test_by,
                            cnn.input_x_text: list(test_btxt),
                            cnn.input_p1: test_bp1,
                            cnn.input_p2: test_bp2,
                            cnn.dropout_keep_prob: 1.0
                        }
                        loss, acc, pred = sess.run(
                            [cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                        losses += loss
                        accuracy += acc
                        predictions += pred.tolist()
                        iter_cnt += 1
                    losses /= iter_cnt
                    accuracy /= iter_cnt
                    predictions = np.array(predictions, dtype='int')

                    logger.logging_eval(step, loss, accuracy, predictions)

                    # Model checkpoint
                    if best_f1 < logger.best_f1:
                        best_f1 = logger.best_f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
