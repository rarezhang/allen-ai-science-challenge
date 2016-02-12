"""
feature extraction
- similarity / distance
- translation
- correlation
"""

import nltk, numpy as np, word2vec as w2v, os, pickle
from nltk.probability import FreqDist
from utils import *
from scipy.spatial.distance import cosine

# global
hitsPerPageA = 20
lemmatize_flag = False
path_w2v = '../data/w2v/mergewikick12'
w2v_model = w2v.load(path_w2v+'.bin')
path_google_dic = '../data/google/google_distance_dic'
google_dic = load_feature_score(path_google_dic)


def freq_dist(document):
    """

    :param document:
    :return:
    """
    overall_token = []
    for line in document:
        tokens = nltk.word_tokenize(line.lower())
        if lemmatize_flag:
            for word in tokens:
                word = word_lemmatizer(word)
                overall_token.extend(word)
        else:
            overall_token.extend(tokens)
    overall_dist = FreqDist(overall_token)
    return overall_dist


def similarity_feature(overall_dist, answers):
    """

    :param overall_dist:
    :param answers:
    :return:[]
    """
    sims = []  # hold all similarities
    for al in answers:  
        sim = []
        for a in al:
            s = []
            for w in a:
                if lemmatize_flag:
                    w = word_lemmatizer(w)
                w_freq = overall_dist.freq(w)  # sample divided by the total number of sample outcomes
                s.append(w_freq)
            sim.append(sum(s))
        sims.extend(sim)
    return sims


def retrieval_score_feature(docs, answers):
    """

    :param doc:
    :param ans:
    :return:
    """
    retrieval_score = []
    import lucene
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field, TextField, StringField
    from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexReader
    from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
    from org.apache.lucene.util import Version
    from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector
    from org.apache.lucene.queryparser.classic import QueryParser
    from org.apache.lucene.search.similarities import BM25Similarity

    lucene.initVM()
    version = Version.LUCENE_CURRENT
    analyzer = StandardAnalyzer(version)
    index = RAMDirectory()  # ram store

    def lucene_index(text):
        """

        :param text:
        :return:
        """
        def addDoc(writer, text):
            """adds documents to the index"""
            docu = Document()
            # TextField: sequence of terms: tokenized
            docu.add(TextField("text", text, Field.Store.YES))
            writer.addDocument(docu)

        config = IndexWriterConfig(version, analyzer)
        writer = IndexWriter(index, config)
        for t in text:
            addDoc(writer, t)
        writer.close()
        return index

    def lucene_retriever(q_string, index, use_BM25=True):
        """

        :param q_string:
        :param index:
        :param use_BM25:
        :return:
        """
        query = QueryParser(version, 'text', analyzer).parse(QueryParser.escape(q_string))
        # search

        reader = DirectoryReader.open(index)
        searcher = IndexSearcher(reader)
        if use_BM25:
            searcher.setSimilarity(BM25Similarity())
        collector = TopScoreDocCollector.create(hitsPerPageA, True)
        searcher.search(query, collector)
        hists = collector.topDocs().scoreDocs
        return hists, searcher, reader

    def get_ans_retrieval_scores(hists, searcher, reader):
        """

        :param hists:
        :param searcher:
        :param reader:
        :return:
        """
        def retrieval_scores_sum(hists):
            """return sorted document+score by score"""
            def doc_score(hists):
                """score"""
                for his in hists:
                    #docID = his.doc
                    #doc = searcher.doc(docID)
                    #doc_name = doc.get("doc_name")
                    #text = doc.get("text")
                    score = his.score
                    yield score
            return sum(doc_score(hists))
        results = retrieval_scores_sum(hists)
        reader.close()
        return results

    ind = lucene_index(docs)
    for answ in answers:
        # answ: 4 answers for each question
        scors = []
        for a in answ:
            h, s, r = lucene_retriever(a, ind)
            sc = get_ans_retrieval_scores(h, s, r)
            scors.append(sc)
        retrieval_score.extend(scors)
    return retrieval_score


def w2v_feature(document, answers):
    """

    :return:
    """
    #check_w2v_bin()
    sims_min, sims_max, sims_avg = [], [], []  # for all 2500 questions
    for al in answers:
        sim_min, sim_max, sim_avg = [], [], []  # for 1 question (4 ans)
        for a in al:  # each ans
            s = [999]
            for w in a.split():  # each word in each answer
                try:
                    v_w = w2v_model[w]  # may get key error
                except:
                    continue
                for line in document:
                    for t in line.split():  # token in each line
                        try:
                            v_t = w2v_model[t]
                            cosine_similarity = cosine(v_w, v_t)
                            s.append(cosine_similarity)
                        except:
                            continue
            try: min_c = min(filter(None,s))
            except: min_c = 0
            try: max_c = max(filter(lambda a: a != 999, s))
            except: max_c = 999
            try: avg_c = sum(filter(lambda a: a != 999, s)) / len(s)
            except: avg_c = 0
            sim_min.append(min_c), sim_max.append(max_c), sim_avg.append(avg_c)
        sims_min.extend(sim_min), sims_max.extend(sim_max), sims_avg.extend(sim_avg)
    print len(sims_avg), len(sims_max), len(sims_min)
    return sims_min, sims_max, sims_avg


def check_w2v_bin():
    """

    :return:
    """
    path_w2v = '../data/w2v/mergewikick12'
    path_phrase = path_w2v + '-phrase'
    path_w2v_bin = path_w2v + '.bin'
    if not os.path.isfile(path_phrase):
        w2v.word2phrase(path_w2v, path_phrase, verbose=True)
    if not os.path.isfile(path_w2v_bin):
        w2v.word2vec(path_phrase, path_w2v_bin, size=200, verbose=True)  # size of word vectors


def google_dis_feature(document, answers):
    """

    :param document:
    :param answers:
    :return:
    """
    sims_min, sims_max, sims_avg = [], [], []  # for all 2500 questions
    for al in answers:
        sim_min, sim_max, sim_avg = [], [], []  # for 1 question (4 ans)
        for a in al:  # each ans
            s = [999]
            for w in a.split():  # each word in each answer
                for line in document:
                    for t in line.split():  # token in each line
                        try:
                            google_distance1 = google_dic[(w,t)]
                            try:
                                google_distance2 = google_dic[(t,w)]
                                s.append(min(filter(None,[google_distance1,google_distance2,999])))
                            except:
                                continue
                        except:
                            continue
            try: min_g = min(filter(None,s))
            except: min_g = 0
            try: max_g = max(filter(lambda a: a != 999, s))
            except: max_g = 999
            try: avg_g = sum(filter(lambda a: a != 999, s)) / len(s)*1.0
            except: avg_g = 0
            sim_min.append(min_g), sim_max.append(max_g), sim_avg.append(avg_g)
        sims_min.extend(sim_min), sims_max.extend(sim_max), sims_avg.extend(sim_avg)
    print len(sims_avg), len(sims_max), len(sims_min)
    print sims_max[:10]
    return sims_min, sims_max, sims_avg







