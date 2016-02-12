from corpus_indexing import check_lucene_index, lucene_retriever
from question_analysis import read_training
from feature_extraction import retrieval_score_feature,freq_dist, similarity_feature, w2v_feature, google_dis_feature
from utils import *
from answer_ranking_perceptron import *
import time, pickle, io


####################################################################################################################33
## test set
test_path = '../data/test/test_set.tsv'
test_path = '../data/test/dev_set.tsv'
que, ans = read_training(test_path)  # questions, correct answer, answers


for query_string in que:
    res = lucene_retriever(query_string, use_BM25=True)  # if don't wanna handle questions, start here
    corpus_names, DocIDs, scores, texts = [], [], [], []
    for r in res:
        corpus_names.append(r[0])
        DocIDs.append(r[1])
        scores.append(r[2])
        texts.append(r[3])


# retrieval feature
path_retrieval_feature = '../data/test_retrieval_feature'
if check_score_exist(path_retrieval_feature):
    dv_retrieval = load_feature_score(path_retrieval_feature)
else:
    dv_retrieval = retrieval_score_feature(texts, ans)
    dump_feature_score(path_retrieval_feature, dv_retrieval)


# use similarity feature: count # of words
path_similarity_feature = '../data/test_similarity_feature'
if check_score_exist(path_similarity_feature):
    dv_similarity = load_feature_score(path_similarity_feature)
else:
    od = freq_dist(texts)
    dv_similarity = similarity_feature(od, ans)
    dump_feature_score(path_similarity_feature, dv_similarity)

'''
# use w2v feature
path_w2v_feature = '../data/test_w2v_feature'
if check_score_exist(path_w2v_feature+'_max'):
    dv_w2v_min = load_feature_score(path_w2v_feature+'_min')
    dv_w2v_max = load_feature_score(path_w2v_feature+'_max')
    dv_w2v_avg = load_feature_score(path_w2v_feature+'_avg')
else:
    dv_w2v_min, dv_w2v_max, dv_w2v_avg = w2v_feature(texts, ans)
    dump_feature_score(path_w2v_feature+'_min', dv_w2v_min)
    dump_feature_score(path_w2v_feature+'_max', dv_w2v_max)
    dump_feature_score(path_w2v_feature+'_avg', dv_w2v_avg)


# use google distance feature
# not useful at all
path_google_dis_feature = '../data/google_dis_feature'
if check_score_exist(path_google_dis_feature+'_min'):
    rs_google_dis_min = load_feature_score(path_google_dis_feature+'_min')
    rs_google_dis_max = load_feature_score(path_google_dis_feature+'_max')
    rs_google_dis_avg = load_feature_score(path_google_dis_feature+'_avg')
else:
    print "google"
    rs_google_dis_min, rs_google_dis_max, rs_google_dis_avg = google_dis_feature(texts, ans)
    dump_feature_score(path_google_dis_feature+'_min', rs_google_dis_min)
    dump_feature_score(path_google_dis_feature+'_max', rs_google_dis_max)
    dump_feature_score(path_google_dis_feature+'_avg', rs_google_dis_avg)
'''



# main
# make X
#X = combine_f(dv_retrieval, dv_similarity, dv_w2v_min, dv_w2v_max)
X = combine_f(dv_retrieval, dv_similarity)


#weight = fit_perceptron(X_train, y_train, epoch=100)
weight = ([[1639031.09625254], [3164.16903221]], 40142.50000010997)
print 'weight from perceptron: {}'.format(weight)
pre_y = pre_perceptron(weight, X)
print len(pre_y)
print pre_y
y_hat = correct_label_num2alpha(pre_y)

print y_hat

test_output = test_path + '.output'
with io.open(test_output, 'a') as outfile:
    for y in y_hat:
        towrite = y+'\n'
        outfile.write(unicode(towrite))



