"""
pylucene corpus indexing
"""

import lucene, io, os
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexReader
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity


def addDoc(w, doc_name, text, file_name):
    """
    add single doc to the index
    :param writer:
    :param doc_name:
    :param text:
    :return:
    """
    doc = Document()
    # TextField: sequence of terms: tokenized
    doc.add(TextField("text", text, Field.Store.YES))
    # StringField: character strings with all punctuation, spacing, and case preserved.
    doc.add(StringField('doc_name', doc_name, Field.Store.YES))
    doc.add(StringField('corpus_name', file_name, Field.Store.YES))
    w.addDocument(doc)


def lucene_index(corpus_main_path, version):
    """

    :param version:
    :return:
    """
    config = IndexWriterConfig(version, analyzer)
    writer = IndexWriter(index, config)
    for f_name in os.listdir(corpus_main_path):
        document_names, texts = read_file(f_name)
        # add 1 corpus at a time
        for d, t in zip(document_names, texts):
            addDoc(writer, d, t, f_name)
    writer.close()



# todo: add filter : file name
def lucene_retriever(q_string, version, use_BM25=False):
    """

    :param index:
    :param v:
    :return:
    """

    query = QueryParser(version, 'text', analyzer).parse(q_string)
    # search
    hitsPerPage = 10  # keep top 10
    reader = IndexReader.open(index)
    searcher = IndexSearcher(reader)

    if use_BM25:
        searcher.setSimilarity(BM25Similarity())
        print "Search query: \"{}\" Similarity: BM25".format(q_string)
    else:
        print "Search query: \"{}\" Similarity: Default".format(q_string)
    collector = TopScoreDocCollector.create(hitsPerPage, True)
    searcher.search(query, collector)
    hs = collector.topDocs().scoreDocs  # hists

    def sorted_doc(hists):
        """return sorted document+score by score"""
        def doc_score(hists):
            """return doc_name & score"""
            for h in hists:
                docID = h.doc
                doc = searcher.doc(docID)
                file_name = doc.get("corpus_name")
                doc_name = doc.get("doc_name")
                text = doc.get("text")
                score = h.score
                yield (file_name, doc_name, score, text)
        return sorted(doc_score(hists), key=lambda tup: tup[2], reverse=True)
    results = sorted_doc(hs)
    reader.close()
    return results


def read_file(f_name):
    """

    :param path:
    :return: python list: document_names, texts
    """
    doc_names, tes = [], []
    path = "..\\data\\corpus\\"+f_name
    with io.open(path, mode='r', encoding='utf-8') as f:
        if f_name in ["Barrons-Grade-4-Science-sentences.txt", "virginia_SOL20Study20Guide.filtered.noquestions.docids.txt"]:
            # format: tree.nodes [tab] text \n
            for line in f:
                line = line.split("\t")
                doc_names.append(line[0])
                tes.append(line[1].strip())
        elif f_name in ["virginia_SOL_flashcards-science5.filtered.txt"]:
            # format: text \n
            i = 0
            for line in f:
                dn = "doc"+str(i)
                doc_names.append(dn)
                tes.append(line.strip())
                i += 1
        elif f_name in ["simpleWiktionary-defs-apr30.txt_filteredStopVerbs"]:
            # format: word [tab] part of speech [tab] text
            i = 0
            for line in f:
                line = line.split("\t")
                dn = "doc" + str(i)
                doc_names.append(dn)
                tes.append(line[0] + " " + line[1] + " " + line[2].strip())
                i += 1
        elif f_name in ["Science_Dictionary_for_Kids_book_filtered.txt"]:
            # format: word
            #         [tab] explanation
            for line in f:
                if line[0] != "\t":
                    word = line.strip()
                    doc_names.append(word)
                else:
                    explanation = line.strip()
                    tes.append(word + " " + explanation)
    return doc_names, tes

import time
time_begin = time.clock()

# initialize lucene
lucene.initVM()
v = Version.LUCENE_CURRENT  # set lucene version
analyzer = StandardAnalyzer()
index_path = "..\\data\\index\\"
index = SimpleFSDirectory(File(index_path))  # todo

if os.listdir(index_path) == "":
    c_path = "..\\data\\corpus"
    lucene_index(c_path, v)
query_string = "simple"
res = lucene_retriever(query_string, v)

for r in res:
    print "corpus_name: \"{}\"   DocID: \"{}\"   Score: {}".format(r[0], r[1], r[2])
    print r[3]  # text
print '\n'



time_end = time.clock()
print time_end - time_begin

# tree type data
# http://stackoverflow.com/questions/9970193/how-to-store-tree-data-in-a-lucene-solr-elasticsearch-index-or-a-nosql-db
