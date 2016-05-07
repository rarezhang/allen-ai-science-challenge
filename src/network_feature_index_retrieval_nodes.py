#!C:\Miniconda2\python.exe -u

"""
index nodes
for network feature
"""

from utils import *

import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexReader
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity


def check_lucene_index(index_file_path, corpus_file_path, file_type=1):
    """

    :param index_file_path:
    :param corpus_file_path:
    :param file_type:
    :return:
    """
    if os.listdir(index_file_path) == []:
        lucene_index(corpus)


def lucene_index(texts):
    """

    :param corpus_file_path:
    :param f_type:
    :return:
    """
    index = set_lucene_index['ind']  # nonlocal variable index
    config = IndexWriterConfig(version, analyzer)
    writer = IndexWriter(index, config)

    for t in texts:
        addDoc(writer, t)
    writer.close()


def addDoc(w, text):
    """
    add single doc to the index
    :param w: writer
    :param doc_name:
    :param text:
    :param file_name:
    :return:
    """
    doc = Document()
    # TextField: sequence of terms: tokenized
    doc.add(TextField("text", text, Field.Store.YES))
    w.addDocument(doc)


def lucene_retrieval(q_string, use_BM25=False):
    """

    :param q_string:
    :param use_BM25:
    :return: retrieval_scores for each question-answer pair
    """
    index = set_lucene_index['ind']  # nonlocal variable index

    def doc_text(hists):
        """
        return doc_name & score
        :param hists:
        """
        text = '_NONE_'
        for h in hists:
            docID = h.doc
            doc = searcher.doc(docID)
            # file_name = doc.get("corpus_name")
            # doc_name = doc.get("doc_name")
            text = doc.get("text")
            #score = h.score
            # yield (file_name, doc_name, score, text)
        return text

    result = '_NONE_'

    # escape special characters via escape function
    if q_string and q_string.strip():   # when pre-process answers, `none of the above` -> '' cause error here
        #print(q_string)
        query = QueryParser(version, 'text', analyzer).parse(QueryParser.escape(q_string))

        # search
        reader = IndexReader.open(index)
        searcher = IndexSearcher(reader)

        if use_BM25:
            searcher.setSimilarity(BM25Similarity(k1=1.5, b=0.75))  # todo: BM25 parameters

        collector = TopScoreDocCollector.create(hitsPerPage, True)
        searcher.search(query, collector)
        hs = collector.topDocs().scoreDocs  # hists
        result = doc_text(hs)

        # reader.close()
    return result  # text: also nodes


@load_or_make
def question_answer_retrieval(questions, answers, path=''):
    """
    return text(nodes) number
    :param questions:
    :param answers:
    :param path: for @load_or_make
    :return:
    """
    num_ans = len(answers[0])

    # 1. loop through question and answers: for quest, answ in zip(questions, answers)
    # 2. get nodes number for each question, duplicate by 4: [nodes_dictionary.get(lucene_retrieval(q), -1)] * num_ans
    # 3. get nodes number for each answer: [nodes_dictionary.get(lucene_retrieval(a), -1) for a in answ]
    results = [zip([nodes_dictionary.get(lucene_retrieval(quest), -1)] * num_ans, [nodes_dictionary.get(lucene_retrieval(a), -1) for a in answ]) for quest, answ in zip(questions, answers)]
    return sum(results, [])


####################################################################################
# main
####################################################################################
# read network nodes
# nodes_dictionary: global
print('read network nodes...')

general_network_path = '../data/network/'
general_corpus_path = '../data/corpus/'

corpus_name = 'aristo_table'  # todo: modify here if want to build other networks
corpus_path = general_corpus_path + corpus_name
table_path = ''.join((general_network_path, corpus_name, '.text'))

nodes_dictionary, _ = load_pickle(table_path+'.pkl')

##############################################################
# index nodes content
# global variables
# initialize lucene
print('index nodes content...')
lucene.initVM()
version = Version.LUCENE_CURRENT  # set lucene version
analyzer = StandardAnalyzer()
hitsPerPage = 1  # just need 1 node

corpus = nodes_dictionary.keys()
index_path = general_network_path + 'nodes_index/'
# python2 does not have ``nonlocal``
set_lucene_index = {'ind': SimpleFSDirectory(File(index_path))}  # use dic for nonlocal variable ``index``
# won't build index if already exists
check_lucene_index(index_path, corpus)   # todo remove the index after testing


##############################################################

# load question and answer
# just use nouns in each question
# load questions / answers: use results from question_answer_analysis.py
print('load question and answer...')
training_path = '../data/training/training_set.tsv'
general_path = training_path

noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)


##############################################################
# questions and answers retrieval: return nodes number -> nodes_dic.value
# [(q1,a1), (q1,a2), ...]
question_answer_nodes_path = general_network_path + 'que_ans_nodes'
question_answer_nodes = question_answer_retrieval(ques, ans, path=question_answer_nodes_path)  # @load_or_make



########################################################################################
# validation and test set
"""
import gc
########################################################################################
# validation set
print('load question and answer...validation...')
validation_path = '../data/validation/validation_set.tsv'
general_path = validation_path

noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)

que_ans_nodes_path = general_network_path + 'validation_que_ans_nodes'

gc.disable()
question_answer_nodes = question_answer_retrieval(ques, ans, path=que_ans_nodes_path)  # @load_or_make
gc.enable()

########################################################################################
# test set
print('load question and answer...test...')
test_path = '../data/test/test_set.tsv'
general_path = test_path

noun_ques_path = general_path + '_noun_ques.pkl'
ques = load_pickle(noun_ques_path)
ans_path = general_path + '_ans.pkl'
ans = load_pickle(ans_path)

que_ans_nodes_path = general_network_path + 'test_que_ans_nodes'
gc.disable()
question_answer_nodes = question_answer_retrieval(ques, ans, path=que_ans_nodes_path)  # @load_or_make
gc.enable()
"""