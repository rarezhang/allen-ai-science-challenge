"""
question analysis
1. kaggle data import
2.
"""

import io
from utils import *

#todo question: remove pountuations
#todo answer analysis:


@load_or_make
def read_kaggle_file(path='', training_set_flag=True, sep='\t'):
    """

    :param path: path to kaggle file
    :param training_set_flag: only training file has correct answer
    :param sep: separator
    :return:
    """
    qid, questions, correctAnswer, answers = [], [], [], []
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # ignore first line
        for line in f:
            line = line.strip().split(sep)
            # qid.append(line[0])  # question id
            questions.append(line[1].lower())  # question text   #lower():because lucene cannot parse <NOT>
            if training_set_flag:
                correctAnswer.append(line[2])  # correct answer A-D
                answers.append(line[3:])  # answers [[],[],...]
            else:
                answers.append(line[2:])
    if training_set_flag:
        return questions, correctAnswer, answers
    else:
        return questions, answers


@load_or_make
def que_concat_ans(questions, answers, path=''):
    """
    concatenate question string and each answer
    :param questions:
    :param answers:
    :param path: for @load_or_make
    :return: list of lists [[q1+A, q1+B,...],[q2+A, q2+B,...],...]
    """
    return [list(map(lambda x: " ".join((qq, x)), aa)) for qq, aa in zip(questions, answers)]


@load_or_make
def pos_questions(questions, path=''):
    """
    add pos tag to question string
    #import pos_tag_word from utils
    :param questions:
    :param path: for @load_or_make
    :return: list of lists [[(t1,pos),(t2,pos)],[],...]
    """
    return [pos_tag_word(q) for q in questions]


def slim_questions(questions_with_pos, V=True, N=True, A=True):
    """

    :param questions_with_pos: [[(t1,pos),(t2,pos)],[],...]
    :param V: keep verb
    :param N: keep noun
    :param A: keep adj / adv
    :return:
    """
    return [get_VNA(q_pos, keepV=V, keepN=N, keepA=A) for q_pos in questions_with_pos]





#####################################################33







