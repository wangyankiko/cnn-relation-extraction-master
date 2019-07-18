import tensorflow as tf
import numpy as np
import os,sys
import datetime
import time
import server_bert
from logger import Logger
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from text_cnn_wy_trans import TextCNN
import data_helpers
import utils
from configure import FLAGS

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def train():
    with tf.device('/cpu:0'):
        train_text, train_y, train_pos1, train_pos2, x_text_clean, train_sentence_len = data_helpers.load_data_and_labels(FLAGS.train_path)
    with tf.device('/cpu:0'):
        test_text, test_y, test_pos1, test_pos2, x_text_clean, test_sentence_len = data_helpers.load_data_and_labels(FLAGS.test_path)

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
    # vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # vocab_processor.fit(train_text + test_text)
    # train_x = np.array(list(vocab_processor.transform(train_text)))
    # test_x = np.array(list(vocab_processor.transform(test_text)))
    # # train_text = np.array(train_text)
    # test_text = np.array(test_text)
    # print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # print("train_x = {0}".format(train_x.shape))
    # print("train_y = {0}".format(train_y.shape))
    # print("test_x = {0}".format(test_x.shape))
    # print("test_y = {0}".format(test_y.shape))

    # pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # pos_vocab_processor.fit(train_pos1 + train_pos2 + test_pos1 + test_pos2)
    # train_p1 = np.array(list(pos_vocab_processor.transform(train_pos1)))
    # train_p2 = np.array(list(pos_vocab_processor.transform(train_pos2)))
    # test_p1 = np.array(list(pos_vocab_processor.transform(test_pos1)))
    # test_p2 = np.array(list(pos_vocab_processor.transform(test_pos2)))
    # print("\nPosition Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    # print("train_p1 = {0}".format(train_p1.shape))
    # print("test_p1 = {0}".format(test_p1.shape))
    # print("")

    x_text_to_id = {}
    id_to_x_text = {}
    id_train = []
    for i, str1 in enumerate(train_text):
        # print(str1)
        x_text_to_id[str1] = i
        id_to_x_text[i] = str1
        id_train.append(i)

    x_text_to_id = {}
    id_to_x_text = {}
    id_test = []
    for i, str1 in enumerate(test_text):
        x_text_to_id[str1] = i
        id_to_x_text[i] = str1
        id_test.append(i)

    # # Randomly shuffle data to split into train and test(dev)
    # np.random.seed(10)
    # x_text_to_id = {}
    # id_to_x_text = {}
    # id = []
    # for i, str1 in enumerate(x_text_clean):
    #     x_text_to_id[str1]=i
    #     id_to_x_text[i] = str1
    #     id.append(i)
    # # print(x_text_to_id)
    # # print(id_to_x_text)
    # # print(id[0:3])
    # print("id:",id)
    #
    # shuffle_indices = np.random.permutation(np.arange(len(y))) #len(y)=8000
    # id_shuffled = np.array(id)[shuffle_indices]
    #
    # # # p1_shuffled = p1[shuffle_indices]
    # # # p2_shuffled = p2[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    # # print(x_shuffled, p1_shuffled,p2_shuffled,y_shuffled)
    #
    # # Split train/test set
    # # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) #x_train=7200, x_dev =800
    # id_train, id_dev = id_shuffled[:dev_sample_index], id_shuffled[dev_sample_index:]
    # # p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    # # p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    # print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    # # x_train = [id_to_x_text[i] for i in id_train]
    # # x_dev = [id_to_x_text[i] for i in id_dev]
    # print("id_train:", id_train)
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
                text_embedding_size=FLAGS.text_embedding_size,#300
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), #2,3,4,5
                num_heads=FLAGS.num_heads,
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

            # Logger
            logger = Logger(out_dir)

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
            # print("id_train:", id_train.shape)
            # print("train_y", train_y.shape)
            id_train = np.array(id_train) #(8000,0)
            # print(id_train.shape)
            # print(id_train)
            # print(train_y.shape)
            # print(train_y)
            # print(list(zip(id_train, train_y)))
            # Generate batches
            batches = data_helpers.batch_iter(list(zip(id_train, train_y)),
                                              FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            # text_embedded_chars_dev = server_bert.load_clean_vector("embedding_unclean.npy", list(id_dev), sentence_len)
            # print("id_dev:",id_dev)
            # print(text_embedded_chars_dev.shape) #(800 90 768)
            for batch in batches:
                train_bx,  train_by = zip(*batch)
                # print(x_batch)
                # print(list(x_batch))
                # print(len(x_batch)) #20
                # print(len(y_batch)) #20

                # Train
                text_embedded_chars = server_bert.load_vector("embedding_unclean.npy", list(train_bx)) #[20 90 768]
                #print(text_embedded_chars.shape) #（20 90 768）
                feed_dict = {
                    cnn.text_embedded_chars: text_embedded_chars,
                    cnn.input_y: train_by,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                    # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    test_batches = data_helpers.batch_iter(list(zip(id_test, test_y)),
                                                           FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    iter_cnt = 0
                    for test_batch in test_batches:
                        test_bx, test_by = zip(*test_batch)
                        a = list(test_bx)
                        # print(a)
                        test_text_embedded_chars = server_bert.load_vector("embedding_unclean_test.npy", list(test_bx))  # [20 90 768)
                        feed_dict = {
                            cnn.text_embedded_chars: test_text_embedded_chars,
                            cnn.input_y:test_by,
                            cnn.emb_dropout_keep_prob: 1.0,
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
