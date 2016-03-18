"""
utils
"""

import numpy as np
import nltk, os, pickle
import time
from nltk.stem import WordNetLemmatizer


######################################################
# decorator functions
######################################################


def timeit(f):
    """
    decorator function
    :param f: function needs time recording
    :return: higher order function -> f = timeit(f)
    """
    def timed(*args, **kw):
        begin_time = time.time()
        fun = f(*args, **kw)
        end_time = time.time()
        print(f, 'time used: ', begin_time-end_time)
        return fun
    return timed


def load_or_make(f):
    """
    decorator function
    :param f:
    :return:
    """
    def wrap_fun(*args, **kwargs):
        pickle_path = kwargs['path'] + '.pkl'
        if check_file_exist(pickle_path):
            data = load_pickle(pickle_path)
        else:
            data = f(*args, **kwargs)
            dump_pickle(pickle_path, data)
        return data
    return wrap_fun



######################################################


def dump_pickle(path, data):
    """
    save data as binary file
    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=2)  # protocol 3 is compatible with protocol 2, pickle_load can load protocol 2


def load_pickle(path):
    """

    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def check_file_exist(path):
    """
    check if ``file`` exists
    :param path:
    :return: T/F
    """
    return os.path.isfile(path)


def pos_tag_word(toks):
    """
    get the part of speech tag of each token
    :param toks: input: a list of token / or string
    :return:
    """
    if (type(toks) is str) or (type(toks) is unicode):
        return nltk.pos_tag(toks.split())
    elif type(toks) is list:
        return nltk.pos_tag(toks)
    else:
        print("can only process list of token / or string")
        exit(1)


def get_VNA(toks_pos, keepV, keepN, keepA):
    """
    keep verb, noun, adjective or/and adverb
    verb_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
    noun_tag = ['NN', 'NNP', 'NNPS', 'NNS']
    adj_tag = ['JJ', 'JJR', 'JJS']
    adv_tag = ['RB', 'RBR', 'RBS']

    :param toks_pos: [(tok, pos),...]
    :param keepV: keep verb or not
    :param keepN: keep noun or not
    :param keepA: keep adjective+adverb or not
    :return: [t_p[0] for t_p in toks_pos if t_p[1].startswith('NN')]
    """
    if keepV and keepN and keepA:
        tag = ('VB', 'NN', 'JJ', 'RB')
    elif keepV and keepN:
        tag = ('VB', 'NN')
    elif keepV and keepA:
        tag = ('VB', 'JJ', 'RB')
    elif keepN and keepA:
        tag = ('VB', 'JJ', 'RB')
    elif keepV:
        tag = 'VB'
    elif keepN:
        tag = 'NN'
    elif keepA:
        tag = ('JJ', 'RB')
    else:
        tag = ''  # if there is no N/V/A: use entire sentence
    results = [t_p[0] for t_p in toks_pos if t_p[1].startswith(tag)]
    return ' '.join(results)

######################################################

'''

def get_performance(pred_ans, correct_ans):
    """

    :param pred_ans:
    :param correct_ans:
    :return:
    """
    num_correct = 0
    for p, c in zip(pred_ans, correct_ans):
        if p == c: num_correct += 1
    total_ques = len(correct_ans)
    return num_correct / total_ques







def add_bigram_trigram(toks, addB=True, addT=True):
    """
    add bigrams and trigrams to the original list
    :param toks: list of tokens
    :param addB: add bigrams or not
    :param addT: add trigrams or not
    :return:
    """
    def find_ngrams(input_list, n):
        joiner = " ".join
        input_list = zip(*[input_list[i:] for i in range(n)])  # [(a,b),....]
        return [joiner(words) for words in input_list]

    if addB and addT: toks.extend(find_ngrams(toks, 2)), toks.extend(find_ngrams(toks, 3))
    elif addB: toks.extend(find_ngrams(toks, 2))
    elif addT: toks.extend(find_ngrams(toks, 3))
    return toks









def combine_features(fea_score1, fea_score2, lamb1=0.5, lamb2=0.5):
    """

    :param fea_score1:
    :param fea_score2:
    :param lamb1:
    :param lamb2:
    :return:
    """
    def add_list_elment(list1, list2, lam1, lam2):
        """

        :param list1: [1,2,3,4]
        :param list2:
        :return:
        """
        v1 = np.array(list1)
        v2 = np.array(list2)
        return (v1*lam1 + v2*lam2).tolist()

    feature_score = []
    for (fea1, fea2) in zip(fea_score1,fea_score2):
        fea = add_list_elment(fea1, fea2, lamb1, lamb2)
        feature_score.append(fea)
    return feature_score





def word_lemmatizer(word):
    """

    :param word:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    w = lemmatizer.lemmatize(word, pos='v')
    return w

'''




