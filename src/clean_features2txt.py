"""
write features to text file
for SVM-rank

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


# load question id
question_id = [t for t in q_id for __ in range(4)]  # duplicate each q_id 4 times

# load target (correct answer)
correct_ans_dic = {'A': [2, 1, 1, 1], 'B': [1, 2, 1, 1], 'C': [1, 1, 2, 1], 'D': [1, 1, 1, 2]}
target = sum([correct_ans_dic[cor] for cor in correct_ans], [])

# load features
general_feature_path = '../data/feature/'
# do not include these feature
feature_exclude = ['svm_rank.txt', 'ck12_retrieval_features_.pkl', 'study_cards_retrieval_features_.pkl']

fea = set(os.listdir(general_feature_path)) - set(feature_exclude)
features_name = [f.partition('.')[0] for f in fea]   # remove extension # partition return: head, sep, tail
features = []

for f in fea:
    single_feature_path = general_feature_path + f
    features.append(load_pickle(single_feature_path))

#######################################################################3
# write features in to text file
num_features = len(features_name)

path = '../data/feature/svm_rank.txt'
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
