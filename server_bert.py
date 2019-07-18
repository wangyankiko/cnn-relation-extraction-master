from bert_serving.client import BertClient
import tensorflow as tf
import os,sys
import data_helpers
import data_helper2
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import numpy
import tensorflow_hub as hub

#
bc = BertClient()
# m =bc.encode(['First do it', 'then do it right', 'then do it better'])
# print(bc.encode(['First do it', 'then do it right', 'then do it better']).shape)
# print(m)
# sess = tf.Session()
# print(sess.run(tf.shape(m[0])))
trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
def save_file(file,file1):

    # x_text,y,pos1, pos2, x_text_clean, sentence_list =data_helpers.load_data_and_labels(file)
    train_text, train_y, train_text_pos, train_e1, train_e2, train_pos1, train_pos2, train_sentence_len, train_sentence_tokens = data_helper2.load_data_and_labels(file)

    # print(train_sentence_tokens[0:1])
    #     # # #
    #     # text = [['e11', 'audits', 'e12']]
    #     # encode_sentence_embedding = bc.encode(text, is_tokenized=True,show_tokens=True)
    #     # print(encode_sentence_embedding[:,:6,])
    elmo_model = hub.Module("/tmp/tfhub_modules/67f40766542914aa2dd2ec6923c4f353f6d0663c", trainable=True)
    text_embedded_chars = elmo_model(train_text, signature="default", as_dict=True)["elmo"]
    numpy.save(file1,text_embedded_chars)
        # # print(sess.run(tf.shape(encode_sentence_embedding)))
        # with open(r"sentence_embedding.txt", "w", encoding="utf-8") as f:
        #     f.write(encode_sentence_embedding)

        # print(encode_sentence_embedding)

def load_vector_init(file):
    embedding = numpy.load(file)

    return numpy.array(embedding)
def load_vector(file,list1):
    embedding = numpy.load(file)
    # print(numpy.array(embedding).shape)
    # embedding= list(embedding)
    output_embedding = [embedding[i] for i in list1]
    # output_embedding[0][0]=[0]*768

    # print(output_embedding)
    # print(numpy.array(output_embedding).shape)
    return numpy.array(output_embedding)


def load_clean_vector(file, list1, sentence_list):
    embedding = numpy.load(file)
    output_embedding = [embedding[i] for i in list1]
    # print(numpy.array(output_embedding).shape)
    for i, j in enumerate(list1):
        # print(numpy.array(output_embedding[i]).shape)
        # output_embedding[i].pop(0)
        # output_embedding[i][0] = [0] * 768
        len = sentence_list[j]
        # print(len)
        if len+1<=89:
            output_embedding[i][len+1] = [0] * 768
        output_embedding[i] = numpy.delete(output_embedding[i],0, axis=0)
        output_embedding[i] = numpy.vstack((output_embedding[i], [0]*768))

    return numpy.array(output_embedding)

# def load_cls_vector(file, list1):
#     embedding = numpy.load(file)
#     output_embedding = [embedding[i] for i in list1]


def load_clean_vector_2(file, list1, sentence_list):
    embedding = numpy.load(file)
    output_embedding = [embedding[i] for i in list1]
    # print(numpy.array(output_embedding).shape)
    for i, j in enumerate(list1):
        # print(numpy.array(output_embedding[i]).shape)
        # output_embedding[i].pop(0)
        # output_embedding[i][0] = [0] * 768
        len = sentence_list[j]
        # print(len)
        if len+1<=89:
            output_embedding[i][len+1] = [0] * 1024
        output_embedding[i] = numpy.delete(output_embedding[i],0, axis=0)
        output_embedding[i] = numpy.vstack((output_embedding[i], [0]*1024))

    return numpy.array(output_embedding)

def get_sentence_embedding(x_text):
    bc = BertClient()
    emcode_sentence_embedding = bc.encode(x_text)

    return emcode_sentence_embedding

# save_file(trainFile)
file_embedding = "embedding.npy"
file_test_embedding = "embedding_test.npy"
file_text_unclean_embedding = "embedding_unclean.npy"
file_text_case_embedding_1_2_3_4 = "embedding_case.npy"
file_text_unclean_embedding_test = "embedding_unclean_test.npy"
file_test_case_embedding_1_2_3_4 = "embedding_case_test.npy"
file_text_case_embedding_2 = "embedding_case_2.npy"
file_test_case_embedding_2 = "embedding_case_test_2.npy"
file_text_case_embedding_2_clean = "embedding_case_2_clean.npy"
file_test_case_embedding_2_clean = "embedding_case_test_2_clean.npy"
file_emlo_test = "elmo_embedding.npy"
list1 = [0]
len_list = [20]
# embedding=load_vector("embedding_test.npy",list1)
# print(embedding.shape)
# save_file(testFile,file_emlo_test)
# a=load_vector(file_test_case_embedding_2_clean, list1)
# print(a.shape)
# # print(a)
# x = ["Roasted ants are a popular snack in Columbia","i am ugly", "Roasted ants are a popular snack in Columbia i am"]


# print("embedding_shape:",text_embedded_chars.get_shape())
# sess = tf.Session()
# # print(sess.run(text_embedded_chars[1]))
# print(text_embedded_chars[2].get_shape())