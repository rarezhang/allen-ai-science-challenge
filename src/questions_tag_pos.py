# -*- coding: utf-8 -*-

"""
use python3: much faster
tag pos for each word in the each question
store in a list -> file (use pickle)
so do not have to do this every time
"""

from question_analysis import read_training
from utils import pos_tag_word
import pickle, re

test_flag = False

def keep_alphanumeric(s):
    """
    keep alphanumeric char
    """
    return re.sub(r'\W+', ' ', s)


# read training dataset
if test_flag:
    training_path = '../data/training/test.tsv'
else:
    training_path = '../data/training/training_set.tsv'

que, _, _ = read_training(training_path)  # questions, correct answer, answers

# remove all non-ascii, get pos-tag for each word
all_pos_query_string = []
for query_string in que:
    query_string = keep_alphanumeric(query_string)
    query_string = pos_tag_word(query_string)  # get pos tag for each word
    all_pos_query_string.append(query_string)

#To write:
if test_flag:
    questions_pos_path = '../data/questions_pos_test'
else:
    questions_pos_path = '../data/questions_pos'

with open(questions_pos_path, 'wb') as f:
    pickle.dump(all_pos_query_string, f, protocol=2)

'''
# To read:
with open(questions_pos_path, 'rb') as f:
    my_list = pickle.load(f)

for i in my_list:
    print i
'''