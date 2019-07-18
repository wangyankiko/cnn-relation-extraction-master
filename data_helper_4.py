import numpy as np
import pandas as pd
import nltk
import re
import tokenization

import utils
from configure import FLAGS


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
def load_data_and_labels(path):
    data = []

    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    sentence_len = []
    sentence_tokens = []
    for idx in range(0, len(lines), 4):
        tokens_pices = []
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        sentence_tokens.append(tokens)
        # print(tokens)
        # print(sentence_tokens)
        # # print(tokens)
        # if max_sentence_length < len(tokens):
        #     max_sentence_length = len(tokens)
        # sentence_len.append(len(tokens))
        # e1 = tokens.index("e12") - 1
        # e2 = tokens.index("e22") - 1
        sentence_text = " ".join(tokens)
        # print(sentence_text)
        # sentence_clean = sentence.replace("e11", "").replace("e12", "").replace("e21", "").replace("e22", "").strip()
        tokens_pices.append("[CLS]")
        tokens_a = tokenizer.tokenize(sentence)
        for token in tokens_a:
            tokens_pices.append(token)
        tokens_pices.append("[SEP]")
        if max_sentence_length < len(tokens_pices):
            max_sentence_length = len(tokens_pices)
        sentence = " ".join(tokens_pices)
        # print(tokens)
        # print(sentence)
        e1 = tokens_pices.index("##12") -2
        e2 = tokens_pices.index("##22") -2
        # print(e1)
        pos_token = nltk.pos_tag(tokens_pices)
        pos = [x[1] for x in pos_token]
        # print(pos_token)
        # print(pos)
        sentence_pos = " ".join(pos)
        # print(sentence_pos)
        data.append([id, sentence, e1, e2, sentence_pos, relation, sentence_text])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2","sentence_pos", "relation", "sentence_text"])

    pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)

    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence_text'].tolist()
    x_text_sentence = df["sentence"].tolist()
    # print(x_text_sentence[:5])
    # print(x_text)
    # print(x_text[:3])
    e1 = df['e1'].tolist()
    e2 = df['e2'].tolist()
    # print(e1)
    # print(e2)
    pos_text = df['sentence_pos'].tolist()
    # print(pos_text[:5])
    # Label Data
    y = df['label']
    # print(y)
    labels_flat = y.values.ravel()
    # print(labels_flat)
    labels_count = np.unique(labels_flat).shape[0]
    # print(labels_count)

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    # print(labels[0:2])

    return x_text, labels, pos_text,e1, e2,pos1, pos2, sentence_len,labels_flat


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        # print(sentence)
        # tokens = nltk.word_tokenize(sentence)
        tokens = sentence.split(" ")
        e1 = df.iloc[df_idx]['e1']
        e2 = df.iloc[df_idx]['e2']

        p1 = ""
        p2 = ""
        # print(tokens)
        # print(e1, e2)
        for word_idx in range(len(tokens)):

            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "

            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        # print(p1)
        # print(p2)
        pos1.append(p1)
        pos2.append(p2)
        # print(pos1)
    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    load_data_and_labels(trainFile)
