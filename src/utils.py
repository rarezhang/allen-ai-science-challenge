"""
utils
"""

import numpy as np
import nltk, os, pickle, string, time, subprocess, fileinput
from nltk.stem import WordNetLemmatizer


######################################################
# decorator functions
######################################################
def time_it(f):
    """
    decorator function
    :param f: function needs time recording
    :return: higher order function -> f = timeit(f)
    """
    def timed(*args, **kw):
        begin_time = time.time()
        fun = f(*args, **kw)
        end_time = time.time()
        print(f, 'time used: ', end_time - begin_time)
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


######################################################
# files
# load and save pickle file
# check if file exist
# concatenate file in a directory
# dump feature: feature matrix -> single feature
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


def concatenate_files(input_directory, output_file):
    """

    :param input_directory:
    :param output_file:
    :return:
    """
    assert os.path.isdir(input_directory), 'input path should be a directory'

    if not input_directory.endswith('/'):
        input_directory = ''.join((input_directory, '/'))

    if not check_file_exist(output_file):
        file_names = os.listdir(input_directory)
        file_paths = [''.join((input_directory, f_n)) for f_n in file_names]
        with open(output_file, 'w', encoding='utf-8') as out_file:
            in_file = fileinput.input(files=file_paths, openhook=fileinput.hook_encoded('utf-8'))  # python 2.7.10, fileinput doest not have `__exit__` --> cannot use `with`
            for line in in_file:
                out_file.write(line)
            in_file.close()


def dump_feature(feature_type, feature_path, features):
    """
    if single feature does not exist, dump single feature
    :param feature_type:
    :param feature_path:
    :param features:
    :return:
    """
    for ind, fea in enumerate(feature_type):
        single_feature_path = ''.join((feature_path, fea.__name__, '.pkl'))
        if not check_file_exist(single_feature_path):
            single_feature = [r[ind] for r in features]
            dump_pickle(single_feature_path, single_feature)


######################################################


######################################################
# part of speech tag
######################################################
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


######################################################
# NLP tool
######################################################
def word_lemmatizer(word):
    """

    :param word:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    w = lemmatizer.lemmatize(word, pos='v')
    return w


def remove_punctuation(s):
    """

    :param s: string
    :return: string without punctuation
    """
    rm_punct_map = dict.fromkeys(map(ord, string.punctuation))
    return s.translate(rm_punct_map)
######################################################


######################################################
# Run command line from python
######################################################
def run_command(command):
    """
    Run command line from python
    :param command:
    :return:
    """
    """
    hint = '''Return binary callable_iterator. To print out the results:
    for line in run_command(command):
        print(line) \n'''
    print(hint)
    """
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')  # callable_iterator object


######################################################
# Performance related
######################################################
def correct_label_num2alpha(num_prediction):
    """
    read prediction results
    :param num_prediction
    :return:
    """
    result = []
    num_pre = np.array(num_prediction).reshape((-1, 4))
    for num_p in num_pre:
        num_p = num_p.tolist()

        max_num_p = max(num_p)
        ind = num_p.index(max_num_p)

        if ind == 0: result.append('A')  # todo: will get more A
        elif ind == 1: result.append('B')
        elif ind == 2: result.append('C')
        else: result.append('D')
    return result


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







'''




