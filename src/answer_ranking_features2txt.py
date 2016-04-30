#!C:\Miniconda2\python.exe -u
"""
write features to text file
for SVM-rank

use python 2 because pickle version issue

<line> .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
<target> .=. <float>
<qid> .=. <positive integer>
<feature> .=. <positive integer>
<value> .=. <float>
<info> .=. <string>
"""

from question_answer_analysis import *

# load data and features
training_path = '../data/training/training_set.tsv'
general_path = training_path
# read_kaggle_file() has @load_or_make
q_id, ques, correct_ans, ans = read_kaggle_file(path=general_path, training_set_flag=True)


# duplicate question id
question_id = [t for t in q_id for __ in range(4)]  # duplicate each q_id 4 times

# load target (correct answer)
correct_ans_dic = {'A': [2, 1, 1, 1], 'B': [1, 2, 1, 1], 'C': [1, 1, 2, 1], 'D': [1, 1, 1, 2]}
target = sum([correct_ans_dic[cor] for cor in correct_ans], [])

# load features
general_feature_path = '../data/feature/'
all_features = os.listdir(general_feature_path)

# do not include these features
# e.g., 'ck12_retrieval_features_.pkl' is temporary file
exclude_pat = '_.pkl'
all_features = [fea for fea in all_features if not fea.endswith(exclude_pat)]

# todo: change here to decide include what features
# only use these features

#include_pat = 'noun_class_sub'
#include_pat = 'class_sub'
#include_pat = 'network'
#include_pat = 'w2v'
#include_pat = 'retrieval'
#include_pat = 'study_cards'
#include_pat = 'simple_wiki'
#include_pat = 'ck12'
include_pat = ''  # all features
test_features = [fea for fea in all_features if include_pat in fea]

# n feature types
'''
include_pat_1 = 'retrieval'
include_pat_2 = 'network'
include_pat = '_'.join((include_pat_1, include_pat_2))
test_features = [fea for fea in all_features if include_pat_1 in fea or include_pat_2 in fea]
'''
# load test features
features = []
for f in test_features:
    single_feature_path = general_feature_path + f
    features.append(load_pickle(single_feature_path))

#######################################################################
# write features in to text file

# extract feature name --> write first line
features_name = [f.partition('.')[0] for f in test_features]   # remove extension # partition return: head, sep, tail
num_features = len(features_name)  # number of features

path = ''.join(('../data/svmrank/svm_rank_', include_pat, '.txt'))

if not check_file_exist(path):
    with open(path, 'a') as file:
        to_write = ' '.join(('# target qid', ' '.join(features_name), '\n'))
        file.write(to_write)  # write comments

        for ind, t in enumerate(target):
            q = question_id[ind]
            to_write = ' '.join((str(t), 'qid:' + str(q)))
            for f_n in range(num_features):
                fea = features[f_n][ind]
                to_write += ' ' + ''.join((str(f_n+1), ':', str(fea)))
            file.write(to_write + '\n')

