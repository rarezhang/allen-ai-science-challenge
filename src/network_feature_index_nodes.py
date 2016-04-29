#!C:\Miniconda2\python.exe -u

"""
index nodes
for network feature
"""

from network_feature import read_aristo_file, table_path

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
        lucene_index(corpus_file_path, file_type)


def lucene_index(corpus_file_path, f_type):
    """

    :param corpus_file_path:
    :param f_type:
    :return:
    """
    index = set_lucene_index['ind']  # nonlocal variable index
    config = IndexWriterConfig(version, analyzer)
    writer = IndexWriter(index, config)
    for f_name in os.listdir(corpus_file_path):
        f_path = corpus_file_path + f_name
        document_names, texts = read_file(f_path, f_type)
        # add 1 corpus at a time

        # handle file name: 1. remove extension; 2. split by ``-``
        f_name, _, _ = f_name.partition('.')   # partition return: head, sep, tail
        f_name = " ".join(f_name.split('-'))
        for d, t in zip(document_names, texts):
            addDoc(writer, d, t, f_name)
    writer.close()


def addDoc(w, doc_name, text, file_name):
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
    # StringField: character strings with all punctuation, spacing, and case preserved.
    doc.add(TextField('doc_name', doc_name, Field.Store.YES))
    #doc.add(StringField('corpus_name', file_name, Field.Store.YES))

    doc.add(TextField('corpus_name', file_name, Field.Store.YES))
    w.addDocument(doc)



def lucene_retrieval(q_string, feature_type, use_BM25=False):
    """

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

    # escape special characters via escape function
    query = QueryParser(version, 'text', analyzer).parse(QueryParser.escape(q_string))

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


####################################################################################
# main
####################################################################################
# read network nodes
nodes_dictionary, _ = read_aristo_file(path=table_path)  # @load_or_make

# global variables
# initialize lucene
lucene.initVM()
version = Version.LUCENE_CURRENT  # set lucene version
analyzer = StandardAnalyzer()
hitsPerPage = 1  # just need 1 node

print(table_path)