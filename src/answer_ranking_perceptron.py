"""

"""
import numpy as np
from question_analysis import read_training
from utils import load_feature_score, dump_feature_score, check_score_exist, get_performance


def combine_f(*features):
    """

    :param features:
    :return:
    """
    X = zip(*features)
    return np.array(X)


def correct_label_num2alpha(num_prediction):
    """

    :param num_prediction
    :return:
    """
    result = []
    num_pre = np.array(num_prediction).reshape((-1, 4))
    for num_p in num_pre:
        num_p = num_p.tolist()
        ind = num_p.index(max(num_p))  # index of the first instance of the largest valued element of list
        if ind == 0: result.append('A')  # todo: will get more A
        elif ind == 1: result.append('B')
        elif ind == 2: result.append('C')
        else: result.append('D')
    return result


def correct_label_alpha2num():
    """

    :return:
    """
    if not check_score_exist('../data/correct_label_num'):
        if check_score_exist('../data/correct_label'):
            cor = load_feature_score('../data/correct_label')
        else:
            _, cor, _ = read_training('../data/training/test.tsv')
        a = []
        for cor_a in cor:
            if cor_a == 'A':
                a.extend([1,-1,-1,-1])
            elif cor_a == 'B':
                a.extend([-1,1,-1,-1])
            elif cor_a == 'C':
                a.extend([-1,-1,1,-1])
            else:
                a.extend([-1,-1,-1,1])
        dump_feature_score('../data/correct_label_num', a)
    else:
        a = load_feature_score('../data/correct_label_num')
    return np.array(a)


def fit_perceptron(data, target, epoch=10, theta1=1.0, theta2=0.1):
    """

    :param data:
    :param target:
    :param epoch:
    :param theta1:
    :param theta2:
    :return:
    """
    num_observation = data.shape[0]
    num_feature = data.shape[1]

    data = np.split(data, num_observation/4)
    target = np.split(target, num_observation/4)
    # initialize
    w, b = np.ones((num_feature, 1.0)), np.zeros((4, 1.0))
    bb = 0
    for e in range(epoch):
        for d, t in zip(data, target):
            results = np.dot(d, w) + b  # 4 answers for each question
            ind_d = np.argmax(results)  # return the max value among 4 ans
            ind_t = np.argmax(t)
            if ind_d == ind_t:   # if current weight can get max value
                w += theta1*d[ind_d].reshape(num_feature, 1)   # update weights
                bb += theta1  # update bias
            else:
                w -= theta2*d[ind_d].reshape(num_feature, 1)   # update weights
                bb -= theta2  # update bias
    return (w, bb)


def pre_perceptron(wei, data):
    """

    :param weight:
    :param data:
    :return:
    """
    w, b = wei[0], wei[1]
    num_observation = data.shape[0]
    print '------------', num_observation
    predictions = []
    for o in range(num_observation):
        d = data[o]
        p = np.vdot(d, w) + b
        predictions.append(p)
    return predictions






