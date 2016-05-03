#!C:\Miniconda3\python.exe -u
"""
network feature
soft inference
1. build networks
2. shortest_path_feature, number_path_feature, random_walk_feature_max, random_walk_feature_sum
3. hyperparameter: length_of_inference -> nx.all_simple_paths cutoff
"""


import networkx as nx
import io
from utils import *


@load_or_make
def read_aristo_file(path='', sep='\t'):
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)   # ignore first line
        nodes_dic = dict()
        edges_list = []
        nodes_index = 1
        for line in f:
            line = line.strip().split(sep)
            nodes_in_line = []
            for node in line:
                if node in nodes_dic:
                    nodes_in_line.append(nodes_dic.get(node))
                else:
                    nodes_dic[node] = nodes_index
                    nodes_in_line.append(nodes_index)
                    nodes_index += 1
            edge_in_line = [(current_node, next_node) for current_node, next_node in zip(nodes_in_line[:-1], nodes_in_line[1:])]
            edges_list.extend(edge_in_line)
    return nodes_dic, edges_list


@load_or_make
def build_network(edges, path=''):
    """
    build undirected network
    :param edges:
    :param path: for @load_or_make
    :return:
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def shortest_path_feature(G, source, target):
    """

    :param G:
    :param source:
    :param target:
    :return:
    """
    if source != -1 and target != -1:  # nodes not in graph
        try:
            return 1.0 / len(nx.shortest_path(G, target=1, source=3))
        except:
            return -1  # no path
    return -1  # nodes not in graph


def random_probability_node(G, node):
    """

    :param G:
    :param node:
    :return:
    """
    # neighbors: dictionary -> {neighbor:cutoff, self:0}
    neighbors = nx.single_source_shortest_path_length(G, node, cutoff=1)  # cutoff=1: 1st degree neighbors
    num_neighbor = len(neighbors) - 1  # `-1` minus self
    probability = 1.0 / num_neighbor
    return probability


def random_probability_path(G, path):
    """

    :param G:
    :param path:
    :return:
    """
    # path[1:-1] -> exclude source and target nodes
    return sum([random_probability_node(G, node) for node in path[1:-1]])


def random_probability_all_paths(G, all_path):
    """

    :param G:
    :param all_path:
    :return:
    """
    return [random_probability_path(G, path) for path in all_path]


def get_all_paths(G, source, target):
    """
    return generator
    :param G:
    :param source:
    :param target:
    :return:
    """
    try:
        return nx.all_simple_paths(G, source, target, cutoff=length_of_inference)
    except:
        return []


def number_path_feature(G, source, target):
    """

    :param G:
    :param source:
    :param target:
    :return:
    """
    return len(list(get_all_paths(G, source, target)))


def random_walk_feature_max(G, source, target):
    """

    :param G:
    :param source:
    :param target:
    :return:
    """
    if source != -1 and target != -1:  # nodes not in graph
        try:
            all_path = get_all_paths(G, source, target)  # return generator
            return max(random_probability_all_paths(G, all_path))
        except:
            return -1  # no path
    return -1  # nodes not in graph


def random_walk_feature_sum(G, source, target):
    """

    :param G:
    :param source:
    :param target:
    :return:
    """
    if source != -1 and target != -1:  # nodes not in graph
        try:
            all_path = get_all_paths(G, source, target)  # return generator
            return sum(random_probability_all_paths(G, all_path))
        except:
            return -1  # no path
    return -1  # nodes not in graph


@load_or_make
def network_score_feature(question_answer_nodes, feature_type, G, path=''):
    """
    feature matrix
    :param question_answer_nodes:
    :param feature_type:
    :param path:
    :return:
    """

    count = 0
    features = []
    for source, target in question_answer_nodes:
        print(source, target)
        print(count)
        count += 1
        features.append(list(map(lambda f: f(G, source, target), feature_type)))
    return features

    #return [list(map(lambda f: f(G, source, target), feature_type)) for source, target in question_answer_nodes]

#################################################################
# main
#################################################################
# Part 1
# build networks based on aristo-table
# todo: these path should compatible with those in network_feature_index_retrieval_nodes.py
# combine tables
print('combine tables...')
general_network_path = '../data/network/'
general_corpus_path = '../data/corpus/'

corpus_name = 'aristo_table'  # todo: modify here if want to build other networks

corpus_path = general_corpus_path + corpus_name
table_path = ''.join((general_network_path, corpus_name, '.text'))

concatenate_files(corpus_path, table_path)

# read aristo table, create nodes and edges
print('create nodes & edges from table...')
nodes_dictionary, edges_list = read_aristo_file(path=table_path)  # @load_or_make

# create network
print('create network...')
network_path = table_path + '_network'
graph = build_network(edges_list, path=network_path)  # @load_or_make

# nodes index & answer / question retrieval
print('load (question<source>, ansewer<target>) nodes ...')
que_ans_nodes_path = general_network_path + 'que_ans_nodes.pkl'
if not check_file_exist(que_ans_nodes_path):
    print('run `network_feature_index_retrieval_nodes.py` now')
    exit(0)

que_ans_nodes = load_pickle(que_ans_nodes_path)


# network feature
# shortest path feature/ random walk feature
print('build network features matrix...')

fea_type = [shortest_path_feature, number_path_feature, random_walk_feature_max, random_walk_feature_sum]   # list of functions

general_feature_path = '../data/feature/'

length_of_inference = 2  # todo: hyperparameter -> nx.all_simple_paths cutoff

network_features_path = ''.join((general_feature_path, corpus_name, '_network_features_', str(length_of_inference), 'lp_'))
network_features = network_score_feature(que_ans_nodes, fea_type, graph, path=network_features_path)  # @load_or_make

# single network feature
print('single network feature...')
dump_feature(fea_type, network_features_path, network_features)
