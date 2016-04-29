#!C:\Miniconda3\python.exe -u
"""
network feature
soft inference
1. build networks
2.
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
    print(G)
    return G





#################################################################
# Part 1
# build networks based on aristo-table

# todo: delete big table, use all table
# combine tables
corpus_path = '../data/corpus/aristo_table/'
table_path = '../data/network/aristo_table.text'
concatenate_files(corpus_path, table_path)

# read aristo table, create nodes and edges
table_path = '../data/network/test.text'  # todo remove this line when finish test
nodes_dictionary, edges_list = read_aristo_file(path=table_path)  # @load_or_make

# create network
network_path = table_path + '_network'
graph = build_network(edges_list, path=network_path)  # @load_or_make

# aristo index
# answer / question retrieval
# Random walk on a graph
