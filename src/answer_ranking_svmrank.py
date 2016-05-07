"""
SVMrank
Support Vector Machine for Ranking
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
"""

from utils import *


# todo: need code refactoring


####################################################
# prepare training file and testing file
def split_file(general_svm_file_path, number_training):
    """

    :param general_svm_file_path:
    :param number_training:
    :return:
    """
    exclude_files = ['training', 'testing', 'test', 'validation']
    svm_files = [f for f in os.listdir(general_svm_file_path) if (f not in exclude_files) and (not f.endswith('.prediction'))]
    for svm_f in svm_files:
        svm_path = ''.join((general_svm_file_path, svm_f))
        training_path = ''.join((general_svm_file_path, 'training/', svm_f, '.training'))
        testing_path = ''.join((general_svm_file_path, 'testing/', svm_f, '.testing'))
        if not (check_file_exist(training_path) and check_file_exist(testing_path)):
            with open(svm_path, 'r') as in_file, open(training_path, 'w') as training_file, open(testing_path, 'w') as testing_file:
                line_number = 0
                for line in in_file:
                    if line_number < number_training:
                        training_file.write(line)
                    else:
                        testing_file.write(line)
                    line_number += 1
                print(svm_f, 'training|testing file split: how many line processed: ', line_number)


####################################################
# training & classifying

def svm_rank(general_svm_file_path):  # todo code refactoring
    """

    :param path_training:
    :param path_testing:

    :return:
    """
    # path to svm_rank tool
    path_win = 'svm_rank_windows/'
    path_learn = path_win + 'svm_rank_learn.exe'
    path_classify = path_win + 'svm_rank_classify.exe'

    # temporary file
    path_model = path_win + 'model.dat'

    # training & testing files
    exclude_files = ['training', 'testing', 'test', 'validation']
    svm_files = [f for f in os.listdir(general_svm_file_path) if (f not in exclude_files) and (not f.endswith('.prediction'))]

    for svm_f in svm_files:
        print(svm_f)
        svm_path = ''.join((general_svm_file_path, svm_f))
        training_path = ''.join((general_svm_file_path, 'training/', svm_f, '.training'))
        testing_path = ''.join((general_svm_file_path, 'testing/', svm_f, '.testing'))

        # path to prediction_score
        path_prediction_score = ''.join((svm_path, '.prediction'))

        # training
        # e.g., svm_rank_learn -c 20.0 train.dat model.dat
        command_learn = ' '.join(('./' + path_learn, '-c 20.0', training_path, path_model))
        #print('command_learn:', command_learn)
        return_info = run_command(command_learn)
        for line in return_info: line

        # classifying
        # e.g., svm_rank_classify ..\test.dat ..\model.dat ..\predictions
        command_classify = ' '.join(('./' + path_classify, testing_path, path_model, path_prediction_score))
        #print('command_classify', command_classify)
        if not check_file_exist(path_prediction_score):
            return_info = run_command(command_classify)
            for line in return_info: line

        # prediction results
        pre_score = [line.strip() for line in open(path_prediction_score, 'r')]
        pre_results = correct_label_num2alpha(pre_score)

        # correct results # todo don't need to read this file every time
        cor_score = [line.strip()[0] for line in open(testing_path, 'r')]
        cor_results = correct_label_num2alpha(cor_score)

        # print result
        print(svm_f, get_performance(pre_results, cor_results))


####################################################
# main
####################################################
# prepare training file and testing file
num_training_question = 1994
num_training = num_training_question * 4 + 1  # 1 -> first line is comment

general_svm_path = '../data/svmrank/'
split_file(general_svm_path, num_training)

####################################################
# training & classifying
# print out performance
svm_rank(general_svm_path)

