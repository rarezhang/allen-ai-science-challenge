"""
question analysis
1. kaggle data import
2.
"""

import io, re
from utils import *

# todo answer analysis: if word(s) appear in every answer -> remove ?


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
            questions.append(remove_punctuation(line[1].lower()))  # question text   #lower():because lucene cannot parse <NOT>
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


@load_or_make
def answer_preprocess(answers, path=''):
    """

    :param answers: # answers [[],[],...]
    :param path: for @load_or_make
    :return:
    """
    all_pat = re.compile('\s*all of the above\s*')  # all of the above
    none_pat = re.compile('\s*none of the above\s*')  # none of the above
    both_pat = re.compile('\s*both ([a-d]) and ([a-d])\s*')  # e.g., both A and B
    results = []
    for ans in answers:
        ans = [remove_punctuation(a.lower()) for a in ans]
        last_ans = ans[-1]  # the last one could be "all of the above" or "none of the above"

        if re.match(all_pat, last_ans):
            new_last_ans = ' '.join(ans[:3])
            ans = ans[:3]
            ans.append(new_last_ans)
        elif re.match(none_pat, last_ans):
            new_last_ans = ' '
            ans = ans[:3]
            ans.append(new_last_ans)
        else:  # expensive
            for ind, a in enumerate(ans):
                both_match = re.match(both_pat, a)
                if both_match:
                    both_left, both_right = both_match.groups()[0], both_match.groups()[1]
                    new_a = ' '.join((ans[ord(both_left) - ord('a')], ans[ord(both_right) - ord('a')]))
                    ans[ind] = new_a
        results.append(ans)
    return results
















