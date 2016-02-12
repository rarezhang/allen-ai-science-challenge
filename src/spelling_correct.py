"""
spelling correct
http://norvig.com/spell-correct.html
"""

import collections
from nltk.corpus import words, brown


def train(features):
    """

    :param features:
    :return:
    """
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

global NWORDS, alphabet
NWORDS = train(words.words())
alphabet = 'abcdefghijklmnopqrstuvwxyz'



def edits1(word):
    """

    :param word:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    """

    :param word:
    :return:
    """
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words):
    """

    :param words:
    :return:
    """
    return set(w for w in words if w in NWORDS)


def correct_word(word):
    """

    :param word:
    :return:
    """
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

