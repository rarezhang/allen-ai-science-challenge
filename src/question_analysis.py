"""
question analysis
1. training data import
2.
"""

import io


def read_training(path):
    """

    :param path:
    :return:
    """
    qid, question, correctAnswer, answers = [], [], [], []
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # ignore first line
        for line in f:
            line = line.strip().split('\t')
            # qid.append(line[0])  # question id
            question.append(line[1].lower())  # question text   #lower():because lucene cannot parse <NOT>
            #correctAnswer.append(line[2])  # correct answer A-D
            #answers.append(line[3:])  # answers [[],[],...]
            answers.append(line[2:])
    #return question, correctAnswer, answers
    return question, answers



