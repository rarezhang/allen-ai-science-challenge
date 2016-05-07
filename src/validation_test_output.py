#!C:\Miniconda3\python.exe -u

"""
validation & test output
"""

from utils import *


class SVMRank:
    """

    """

    def __init__(self, path_win='svm_rank_windows/', path_model='model.dat'):
        self.path_win = path_win
        self.path_learn = self.path_win + 'svm_rank_learn.exe'
        self.path_classify = self.path_win + 'svm_rank_classify.exe'
        self.path_model = path_model

    def learn(self, train_data_path):
        assert os.path.isfile(train_data_path), 'check train_data_path'
        path_learn = self.path_learn
        path_model = self.path_model
        # e.g., svm_rank_learn -c 20.0 train.dat model.dat
        command_learn = ' '.join(('./' + path_learn, '-c 20.0', train_data_path, path_model))  # todo: parameters '-c 20.0'
        return_info = run_command(command_learn)
        for line in return_info: print(line)

    def classify(self, test_data_path):
        assert os.path.isfile(test_data_path), 'check test_data_path'
        path_classify = self.path_classify
        path_prediction_score = test_data_path + '.prediction'
        path_model = self.path_model
        # e.g., svm_rank_classify ..\test.dat ..\model.dat ..\predictions

        command_classify = ' '.join(('./' + path_classify, test_data_path, path_model, path_prediction_score))
        return_info = run_command(command_classify)
        for line in return_info: print(line)

##########################################################################################
# main
##########################################################################################
svm_rank = SVMRank()

##########################################################################################
# validation set
train_data_path = '../data/svmrank/svm_rank_normalized_.txt'
svm_rank.learn(train_data_path)

test_data_path = '../data/svmrank/validation/svm_rank_validation_normalized_.txt'  # todo
test_data_path = '../data/svmrank/validation/svm_rank_validation_normalized_retrieval.txt'

svm_rank.classify(test_data_path)

# write label answer
prediction_score_path = test_data_path + '.prediction'
prediction_score = [line.strip() for line in open(prediction_score_path, 'r')]
pre_results = correct_label_num2alpha(prediction_score)
out_file_path = test_data_path + '.output'
out_file = open(out_file_path, 'w')
out_file.write("\n".join(pre_results))
##########################################################################################
# test set
train_data_path = '../data/svmrank/svm_rank_normalized_.txt'
svm_rank.learn(train_data_path)

test_data_path = '../data/svmrank/test/svm_rank_test_normalized_.txt'   # todo
test_data_path = '../data/svmrank/test/svm_rank_test_normalized_retrieval.txt'

svm_rank.classify(test_data_path)

# write label answer
prediction_score_path = test_data_path + '.prediction'
prediction_score = [line.strip() for line in open(prediction_score_path, 'r')]
pre_results = correct_label_num2alpha(prediction_score)
out_file_path = test_data_path + '.output'
out_file = open(out_file_path, 'w')
out_file.write("\n".join(pre_results))



