import numpy as np
import pandas as pd
import nltk
import re
import os,sys
import tensorflow as tf
import codecs

import utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
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


def load_data_and_labels(path):
    data = []
    # f = codecs.open(path)
    lines = [line.strip() for line in open(path)]
    # print(lines)
    max_sentence_length = 0
    sentence_len = []
    for idx in range(0, len(lines), 4):
        # print(idx)
        id = lines[idx].split("\t")[0]
        # print(id)
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        # print(sentence)
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')
        # print(sentence)
        sentence = clean_str(sentence)
        # print(sentence)
        tokens = nltk.word_tokenize(sentence)
        # print(tokens)
        while len(tokens) < FLAGS.max_sentence_length:
            tokens.append("<PAD>")
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        sentence_len.append(len(tokens))
        e1 = tokens.index("e12") - 1
        # print(e1)
        e2 = tokens.index("e22") - 1
        # print(e2)
        sentence = " ".join(tokens)
        # print(sentence)
        sentence_clean = sentence.replace("e11","").replace("e12","").replace("e21", "").replace("e22","").strip()
        # print(sentence_clean)
        data.append([id, sentence,sentence_clean, e1, e2, relation])
        # print(data)

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence","sentence_clean", "e1", "e2", "relation"])
    # print(df)
    pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)

    df['label'] = [utils.class2label[r] for r in df['relation']]
    # print(df['label'])

    # Text Data
    x_text = df['sentence'].tolist()
    # print(x_text[:1])
    # print(len(x_text[:1]))
    x_text_clean = df["sentence_clean"].tolist()
    # print(x_text_clean)
    # print(len(x_text_clean))
    # print(x_text_clean[0])
    # print(sentence_len[0])
    # x_text_str = df["sentence_clean"].to_string()
    # print(x_text_str)
    # print(x_text_clean)
    # for i, str in enumerate(x_text_clean):
    #     print(i , "  ", str)
    # Label Data
    y = df['label']
    # print(y)
    labels_flat = y.values.ravel()
    print(labels_flat)
    labels_count = np.unique(labels_flat).shape[0]
    # print(labels_count)

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        # print(num_labels)
        index_offset = np.arange(num_labels) * num_classes
        # print(index_offset)
        labels_one_hot = np.zeros((num_labels, num_classes))
        # print(labels_one_hot)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        # print(labels_one_hot)
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    # print(labels)
    labels = labels.astype(np.uint8)
    # print(labels)#one-hot vector
    # print(labels.shape)#(8000,19)
    # f.close()
    # print(x_text)
    return x_text, labels, pos1, pos2, x_text_clean, sentence_len


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        # print(sentence)
        tokens = nltk.word_tokenize(sentence)
        # print(tokens)
        e1 = df.iloc[df_idx]['e1']
        # print(e1)
        e2 = df.iloc[df_idx]['e2']
        # print(e2)

        p1 = ""
        p2 = ""
        # print("max",max_sentence_length)
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
            # print(p1)
            # print(p2)
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    # print(len(data))
    # print(len(data[0])) #(7200,4)
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

def save_sentences(extreact_file, save_file):
    sentence,labels,pos1, pos2 = load_data_and_labels(extreact_file)
    with open(save_file, 'w', encoding='utf-8') as f:
            for i in range(len(sentence)):
                f.write(str(sentence[i]))
                f.write("\n")

def save_file_operation(file, file2):
    data_sentence = []
    data = []
    data_label = []
    # f1= codecs.open(file)
    # f2 = codecs.open(file2)
    lines = [line.strip() for line in open(file)]    # print(lines)
    max_sentence_length = 0
    sentence_len = []

    for idx in range(0, len(lines), 4):
        # print(idx)
        # id = lines[idx].split("\t")[0]
        # print(id)
        relation = lines[idx + 1]
        print(relation)

        sentence = lines[idx].split("\t")[1][1:-1]
        # print(sentence)
        sentence = sentence.replace('<e1>', ' e11 ')
        sentence = sentence.replace('</e1>', ' e12 ')
        sentence = sentence.replace('<e2>', ' e21 ')
        sentence = sentence.replace('</e2>', ' e22 ')
        # print(sentence)
        sentence = clean_str(sentence)
        data.append([relation, sentence])
    df = pd.DataFrame(data=data, columns=["label", "content"])
    df.to_csv(file2, index=False, sep='\t')
    # f1.close()
    # f2.close()


if __name__ == "__main__":
    trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    save_file = r"train_sentences.txt"
    save_file_tsv = r"train_bert.tsv"
    save_file_test_tsv = r"test_bert.tsv"
    # load_data_and_labels(trainFile)
    # save_sentences(trainFile,save_file)
    # save_file_operation(testFile, save_file_test_tsv)
    # data = pd.read_csv(r"/home/wangyan/bert-zzw/runs/2019-04-28T15:08:50.780556/test_results.tsv", sep="\t")
    # print(np.array(data).shape)
    #
    # predictions = np.argmax(np.array(data), axis=1)
    # # sess = tf.Session()
    # # predictions =  sess.run(predictions)
    # print(predictions[50:100])