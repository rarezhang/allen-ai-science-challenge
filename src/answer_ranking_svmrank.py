"""
SVMrank
Support Vector Machine for Ranking
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
"""
import subprocess
import numpy as np
from utils import check_file_exist


def run_command(command):
    """
    Run command line from python
    :param command:
    :return:
    """
    hint = '''Return binary callable_iterator. To print out the results:
    for line in run_command(command):
        print(line) \n'''
    print(hint)

    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')  # callable_iterator object


####################################################
path_win = 'svm_rank_windows/'
path_learn = path_win + 'svm_rank_learn.exe'
path_classify = path_win + 'svm_rank_classify.exe'

path_training = path_win + 'svm_rank_training.txt'
path_testing = path_win + 'svm_rank_testing.txt'
path_model = path_win + 'model.dat'
path_prediction_score = path_win + 'prediction'


# training
# e.g., svm_rank_learn -c 20.0 train.dat model.dat
command_learn = ' '.join(('./' + path_learn, '-c 20.0', path_training, path_model))
print('command_learn:', command_learn)

if not check_file_exist(path_model):
    return_info = run_command(command_learn)
    for line in return_info:
        print(line)


# classifying
# e.g., svm_rank_classify ..\test.dat ..\model.dat ..\predictions
command_classify = ' '.join(('./' + path_classify, path_testing, path_model, path_prediction_score))
print('command_classify', command_classify)

if not check_file_exist(path_prediction_score):
    return_info = run_command(command_classify)
    for line in return_info:
        print(line)


###############################################################
# read prediction results
def correct_label_num2alpha(num_prediction):
    """

    :param num_prediction
    :return:
    """
    result = []
    num_pre = np.array(num_prediction).reshape((-1, 4))
    for num_p in num_pre:
        num_p = num_p.tolist()

        max_num_p = max(num_p)
        ind = num_p.index(max_num_p)

        if ind == 0: result.append('A')  # todo: will get more A
        elif ind == 1: result.append('B')
        elif ind == 2: result.append('C')
        else: result.append('D')
    return result

# prediction results
pre_score = [line.strip() for line in open(path_prediction_score, 'r')]
pre_results = correct_label_num2alpha(pre_score)

# correct results
cor_score = [line.strip()[0] for line in open(path_testing, 'r')][1:]  # skip first line, first line is #

cor_results = correct_label_num2alpha(cor_score)


########################################
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

print(get_performance(pre_results, cor_results))