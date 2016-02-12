"""
utils
"""
from __future__ import division
import numpy as np
import nltk, os, pickle
from nltk.stem import WordNetLemmatizer




verb_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
noun_tag = ['NN', 'NNP', 'NNPS', 'NNS']
ad_tag = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']  # adjective + adverb


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


def pos_tag_word(toks):
    """
    get the part of speech tag of each token
    :param toks: input: a list of token / or string
    :return:
    """
    if type(toks) is list:
        return nltk.pos_tag(toks)
    elif (type(toks) is str) or (type(toks) is unicode):
        return nltk.pos_tag(toks.split())
    else:
        print("can only process list of token / or string")
        exit(1)


def get_VNA(toks_pos, keepV=True, keepN=True, keepA=True):
    """
    keep all verb, noun, adjective
    :param toks_pos: [(tok, pos),...]
    :param keepV: keep verb or not
    :param keepN: keep noun or not
    :param keepA: keep adjective+adverb or not
    :return:
    """
    verbs, nouns, ad = [], [], []
    results = []
    for (toks, pos) in toks_pos:
        if pos in verb_tag: verbs.append(toks)
        elif pos in noun_tag: nouns.append(toks)
        elif pos in ad_tag: ad.append(toks)
    if keepV and keepN and keepA: results.extend(verbs), results.extend(nouns), results.extend(ad)
    elif keepV and keepN: results.extend(verbs), results.extend(nouns)
    elif keepV and keepA: results.extend(verbs), results.extend(ad)
    elif keepN and keepA: results.extend(nouns), results.extend(ad)
    elif keepV: results.extend(verbs)
    elif keepN: results.extend(nouns)
    elif keepA: results.extend(ad)
    return results


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


def dump_feature_score(path, feature_score):
    """
    store feature score with pickle
    :param path:
    :param feature_score:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(feature_score, f, protocol=2)


def load_feature_score(path):
    """

    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        feature_score = pickle.load(f)
    return feature_score


def check_score_exist(path):
    """
    check if certain feature score exists
    :param path:
    :return: T/F
    """
    return os.path.isfile(path)


def word_lemmatizer(word):
    """

    :param word:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    w = lemmatizer.lemmatize(word, pos='v')
    return w






