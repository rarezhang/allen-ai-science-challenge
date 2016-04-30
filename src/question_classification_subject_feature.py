#!C:\Miniconda2\python.exe -u

"""
using questions classification results
"""

import io
from utils import *

import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexReader
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector, BooleanQuery, BooleanClause
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity


def read_question_class(path, sep=' '):
    """

    :param path:
    :param sep: separator
    :return:
    """
    question_id, subject = [], []
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip().split(sep)
            question_id.append(line[0])
            subject.append(sub_dic.get(line[1]))
    return question_id, subject


def compare_2_sub_classification(ques_1, ques_2):
    """
    inter reliability
    compare 2 subjects classification results: (entire VS noun): 0.7976
    :param entire_ques:
    :param noun_ques:
    :return:
    """
    count = 0
    for e, n in zip(ques_1, ques_2):
        if e == n:
            count += 1
    print('Compare two subject classification results, {} of them are same'.format(count/2500.0))


def lucene_retrieval_multifield(q_string, q_class, feature_type, use_BM25=False):
    """
    multifield: different query string for different field
    not same word on different field
    :param q_string:
    :param feature_type:
    :param use_BM25:
    :return: retrieval_scores for each question-answer pair
    """
    index = set_lucene_index['ind']  # nonlocal variable index

    def retrieval_scores(hists):
        """
        return sorted document+score by score
        :param hists:
        """
        def doc_score(hists):
            """
            return doc_name & score
            :param hists:
            """
            for h in hists:
                # docID = h.doc
                # doc = searcher.doc(docID)
                # file_name = doc.get("corpus_name")
                # doc_name = doc.get("doc_name")
                # text = doc.get("text")
                score = h.score
                # yield (file_name, doc_name, score, text)
                yield score
        doc_score_list = list(doc_score(hists))
        return map(lambda f: f(doc_score_list), feature_type)  # feature_type is a list of function

    text_query = QueryParser(version, 'text', analyzer).parse(QueryParser.escape(q_string))
    subject_query = QueryParser(version, 'corpus_name', analyzer).parse(QueryParser.escape(q_class))
    query = BooleanQuery()

    # BooleanClause.Occur
    # MUST implies that the keyword must occur
    #  SHOULD implies that the keyword SHOULD occur
    query.add(text_query, BooleanClause.Occur.SHOULD)
    query.add(subject_query, BooleanClause.Occur.SHOULD)

    # search
    reader = IndexReader.open(index)
    searcher = IndexSearcher(reader)

    if use_BM25:
        searcher.setSimilarity(BM25Similarity(k1=1.5, b=0.75))  # todo: BM25 parameters

    collector = TopScoreDocCollector.create(hitsPerPage, True)
    searcher.search(query, collector)
    hs = collector.topDocs().scoreDocs  # hists

    results = retrieval_scores(hs)
    # reader.close()
    return results  # retrieval_scores for each question-answer pair


@load_or_make
def subject_class_features(que_ans_pairs, question_class, feature_type, path=''):
    """

    :param que_ans_pairs:
    :param question_class:
    :param feature_type:
    :param path: for @load_or_make
    :return:
    """
    def feature_scores(q_a_pairs, q_class):
        for q_a_p, q_cl in zip(q_a_pairs, q_class):
            yield [lucene_retrieval_multifield(q_a, q_cl, feature_type) for q_a in q_a_p]  # todo
    # sum(list, []) : [[[1],[2]], [[3],[4]], ...] -> [[1,2], [3,4]]
    return sum(feature_scores(que_ans_pairs, question_class), [])

########################################################
# main
########################################################
# global
flag_entire_ques = True
# question subjects
sub_dic = {'0': 'biology', '1': 'physics', '2': 'earth', '3': 'life', '4': 'chemistry', '5': 'physical'}


# initialize lucene
lucene.initVM()
version = Version.LUCENE_CURRENT  # set lucene version
analyzer = StandardAnalyzer()
hitsPerPage = 5  # keep top 5

# read question subjects classification results
general_question_classification_path = '../data/questionclass/'
subject_classification_path = general_question_classification_path + 'subject/'
if flag_entire_ques:
    ques_subject_path = subject_classification_path + 'training_subject.txt'
else:
    ques_subject_path = subject_classification_path + 'training_noun_subject.txt'

_, ques_sub_class = read_question_class(ques_subject_path)  # return q_id, q_sub

# compare 2 subjects classification results: (entire VS noun)
#_, entire_ques = read_question_class(entire_ques_subject_path)
#_, noun_ques = read_question_class(noun_ques_subject_path)
#compare_2_sub_classification(entire_ques, noun_ques)


##############################################################
# todo: change here to make features for different corpus
# for subject classification feature: only ck12 --> search on book title
corpus_name = 'ck12'


general_index_path = "../data/index/"
index_path = ''.join((general_index_path, corpus_name, '/'))

# python2 does not have ``nonlocal``
set_lucene_index = {'ind': SimpleFSDirectory(File(index_path))}  # use dic for nonlocal variable ``index``

# all retrieval features
# @load_or_make
fea_type = [max, sum]  # list of functions

# load question and answer
# load questions / answers: use results from question_answer_analysis.py
training_path = '../data/training/training_set.tsv'
general_path = training_path


general_feature_path = '../data/feature/'
if flag_entire_ques:
    ques_ans_path = general_path + '_entire_ques_ans.pkl'
    subject_class_features_path = ''.join((general_feature_path, corpus_name, '_class_sub_features_'))
else:
    ques_ans_path = general_path + '_noun_ques_ans.pkl'
    subject_class_features_path = ''.join((general_feature_path, corpus_name, '_noun_class_sub_features_'))

ques_ans = load_pickle(ques_ans_path)   # [[q1+A, ...],...]
retrieval_features = subject_class_features(ques_ans, ques_sub_class, fea_type, path=subject_class_features_path)

# single retrieval feature
# if single feature does not exist, dump single feature
dump_feature(fea_type, subject_class_features_path, retrieval_features)