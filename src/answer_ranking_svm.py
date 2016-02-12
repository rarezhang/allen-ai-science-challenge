"""
answer ranking
1. svm pairwise ranking
https://gist.github.com/agramfort/2071994

too slow !!!
"""


import itertools
import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestClassifier
from question_analysis import read_training
from utils import load_feature_score, dump_feature_score, check_score_exist
from scipy.sparse import csr_matrix


def combine_f(*features):
    """

    :param features:
    :return:
    """
    X = zip(*features)
    return np.array(X)


def correct_label_alpha2num():
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


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """

    X_new = np.empty((0,2))
    y_new = np.empty((0,1))
    #X_new = []
    #y_new = []

    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue

        X_new = np.append(X_new, (X[i] - X[j]))
        y_new = np.append(y_new, (np.sign(y[i, 0] - y[j, 0])))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]

    print(X_new.reshape(-1,2).shape)
    return X_new.reshape(-1,2), y_new.ravel()


class RankSVM(RandomForestClassifier):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


# main

# read features
path_similarity_feature = '../data/similarity_feature'
rs_similarity = load_feature_score(path_similarity_feature)

path_retrieval_feature = '../data/retrieval_feature'
rs_retrieval = load_feature_score(path_retrieval_feature)

# make X
X = combine_f(rs_retrieval, rs_similarity)
# make y
y = correct_label_alpha2num()



num_samples = y.shape[0]
test_size = 0.5
"""
cv = cross_validation.KFold(num_samples, 2)
train, test = iter(cv).next()
#cv = cross_validation.StratifiedShuffleSplit(y, test_size=0.95)
#print(iter(cv))
#train, test = iter(cv).__next__()
"""
print(num_samples*test_size)
test = range(int(num_samples*test_size))
train = range(int(num_samples*test_size)+1,num_samples)
print test[:10], test[-10:]
print train[:10], train[-10:]
print(train)
print(X.shape)
X_train, y_train = X[train], y[train]
X_test, y_test = X[test], y[test]
'''
print('training..')
rank_svm = RankSVM().fit(X_train, y_train)
print('predicating..')
re = rank_svm.score(X_test, y_test)
print(re)
'''
