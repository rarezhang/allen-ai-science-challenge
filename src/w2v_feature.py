#!C:\Miniconda2\python.exe -u

"""
1. create word2vector
2. get w2v features

library: word2vec is on python2-32bit
do not call functions in this file from other file
"""

import word2vec
import numpy
from scipy.spatial.distance import cosine
from utils import *


# Part 1
# build w2v bin file (different corpus different bin file)
@time_it
def check_bin_file(general_bin_file_path, general_corpus_file_path, corpus_name):
    """

    :param bin_path:
    :param corpus_path:
    :return:
    """
    word2vec_bin_path = ''.join((general_bin_file_path, corpus_name, '.bin'))
    if not check_file_exist(word2vec_bin_path):
        w2v_bin(general_bin_file_path, general_corpus_file_path, corpus_name)


def w2v_bin(general_bin_file_path, general_corpus_file_path, corpus_name):
    """

    :param general_bin_file_path:
    :param general_corpus_file_path:
    :param corpus_name:
    :return:
    """
    # combine all files in one corpus
    text_file_path = ''.join((general_bin_file_path, corpus_name, '.text'))
    corpus_path = ''.join((general_corpus_file_path, corpus_name, '\\'))
    # create .text file for word2vec
    concatenate_files(corpus_path, text_file_path)

    # create word2vec .bin file
    word2vec_bin_path = ''.join((general_bin_file_path, corpus_name, '.bin'))
    word2vec.word2vec(text_file_path, word2vec_bin_path, size=200, verbose=True)  # size of word vectors


# Part 2
# load question/answer string
# get w2v score for each word
@load_or_make
def text_to_w2v(text, flag_ques, path=''):
    """

    :param text:
    :param flag_ques:
    True(questions): list of string [q1, q2, ...]
    False(answers): list of lists of string [[a1,a2,a3,a4],...]
    :param path:
    :return:
    """
    def word_to_vect(word):
        """
        handle KeyError
        :param word:
        :return:
        """
        try:
            return w2v_model[word]
        except KeyError:
            return numpy.ones(shape=(200,))  # 200: w2v size todo modify to global variable

    if flag_ques:  # questions
        questions = text
        return [[word_to_vect(w) for w in que.split()] for que in questions]
    else:  # answers
        answers = text
        return [[[word_to_vect(w) for w in a.split()] for a in answ] for answ in answers]


# Part 3
# calculate cosine similarity between answers vector and questions vector
@load_or_make
def ques_ans_cosine_sim(questions_w2v, answers_w2v, path=''):
    """

    :param questions_w2v:
    :param answers_w2v:
    :param path:
    :return:
    """
    return [[[cosine(q_w2v, a_w2v) for q_w2v, a_w2v in zip(ques, single_ans)] for single_ans in ans] for ques, ans in zip(questions_w2v, answers_w2v)]


# Part 4
# feature matrix
@load_or_make
def word2vec_score_feature(question_answer_similarity, feature_type, path=''):
    """

    :param question_answer_similarity: [[[a1],[a2],[a3],[a4]],[],...]
    :param feature_type:
    :param path:
    :return:
    """
    w2v_score_feature = [[map(lambda f: f(q_a_sim), feature_type) if len(q_a_sim) != 0 else [0]*len(feature_type) for q_a_sim in que_ans_sim] for que_ans_sim in question_answer_similarity]
    return sum(w2v_score_feature, [])


##################################################################################
# main
##################################################################################
# build w2v bin file: different corpus different bin file

# todo: modify here to index/make features for different corpus
#corpus_name = 'ck12'
corpus_name = 'study_cards'


general_bin_path = "..\\data\\w2vbin\\"
general_corpus_path = "..\\data\\corpus\\"

# won't build w2v bin file if already exists
check_bin_file(general_bin_path, general_corpus_path, corpus_name)

# global
w2v_bin_path = ''.join((general_bin_path, corpus_name, '.bin'))
w2v_model = word2vec.load(w2v_bin_path)


##################################################################################
# question / answer: string -> word2vec
training_path = '../data/training/training_set.tsv'
general_path = training_path

# just use nouns in each question
# load questions / answers: use results from question_answer_analysis.py
noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)

ques_w2v_path = noun_ques_path.replace('.pkl', '') + '_w2v'
ques_w2v = text_to_w2v(ques, flag_ques=True, path=ques_w2v_path)  # @load_or_make

ans_w2v_path = ans_path.replace('.pkl', '') + '_w2v'
ans_w2v = text_to_w2v(ans, flag_ques=False, path=ans_w2v_path)  # @load_or_make

##################################################################################
# calculate cosine similarity between answers-vector and question-vector
ques_ans_sim_path = general_path + '_ques_ans_sim'
ques_ans_sim = ques_ans_cosine_sim(ques_w2v, ans_w2v, path=ques_ans_sim_path)  # @load_or_make


##################################################################################
# feature matrix
# all word2vec features
fea_type = [max, min]  # feature type: list of functions
general_feature_path = '../data/feature/'

word2vec_features_path = ''.join((general_feature_path, corpus_name, '_noun_w2v_features_'))
word2vec_features = word2vec_score_feature(ques_ans_sim, fea_type, path=word2vec_features_path)  # @load_or_make


##################################################################################
# single word2vec feature
# if single feature does not exist, dump single feature
dump_feature(fea_type, word2vec_features_path, word2vec_features, flag_normalize_feature=False)
dump_feature(fea_type, word2vec_features_path, word2vec_features, flag_normalize_feature=True)


########################################################################################
# validation and test set
"""
import gc
########################################################################################
# validation set
general_feature_path = '../data/validation/feature/'
validation_path = '../data/validation/validation_set.tsv'
general_path = validation_path

# just use nouns in each question
# load questions / answers: use results from question_answer_analysis.py
noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)

ques_w2v_path = noun_ques_path.replace('.pkl', '') + '_w2v'
ques_w2v = text_to_w2v(ques, flag_ques=True, path=ques_w2v_path)  # @load_or_make

ans_w2v_path = ans_path.replace('.pkl', '') + '_w2v'
ans_w2v = text_to_w2v(ans, flag_ques=False, path=ans_w2v_path)  # @load_or_make
##################################################################################
# calculate cosine similarity between answers-vector and question-vector
ques_ans_sim_path = general_path + '_ques_ans_sim'
gc.disable()
ques_ans_sim = ques_ans_cosine_sim(ques_w2v, ans_w2v, path=ques_ans_sim_path)  # @load_or_make
gc.enable()
##################################################################################
# feature matrix
# all word2vec features
word2vec_features_path = ''.join((general_feature_path, corpus_name, '_noun_w2v_features_'))
gc.disable()
word2vec_features = word2vec_score_feature(ques_ans_sim, fea_type, path=word2vec_features_path)  # @load_or_make
gc.enable()
##################################################################################
# single word2vec feature
dump_feature(fea_type, word2vec_features_path, word2vec_features, flag_normalize_feature=True)


##################################################################################
# test set
general_feature_path = '../data/test/feature/'
test_path = '../data/test/test_set.tsv'
general_path = test_path

# just use nouns in each question
# load questions / answers: use results from question_answer_analysis.py
noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)

ques_w2v_path = noun_ques_path.replace('.pkl', '') + '_w2v'
ques_w2v = text_to_w2v(ques, flag_ques=True, path=ques_w2v_path)  # @load_or_make

ans_w2v_path = ans_path.replace('.pkl', '') + '_w2v'
ans_w2v = text_to_w2v(ans, flag_ques=False, path=ans_w2v_path)  # @load_or_make
##################################################################################
# calculate cosine similarity between answers-vector and question-vector
ques_ans_sim_path = general_path + '_ques_ans_sim'
gc.disable()
ques_ans_sim = ques_ans_cosine_sim(ques_w2v, ans_w2v, path=ques_ans_sim_path)  # @load_or_make
gc.enable()
##################################################################################
# feature matrix
# all word2vec features
word2vec_features_path = ''.join((general_feature_path, corpus_name, '_noun_w2v_features_'))
gc.disable()
word2vec_features = word2vec_score_feature(ques_ans_sim, fea_type, path=word2vec_features_path)  # @load_or_make
gc.enable()
##################################################################################
# single word2vec feature
dump_feature(fea_type, word2vec_features_path, word2vec_features, flag_normalize_feature=True)
"""