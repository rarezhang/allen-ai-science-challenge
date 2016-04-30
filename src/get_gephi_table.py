
from utils import *

general_network_path = '../data/network/'
general_corpus_path = '../data/corpus/'

corpus_name = 'aristo_table'  # todo: change here if want to build other networks

corpus_path = general_corpus_path + corpus_name
table_path = ''.join((general_network_path, corpus_name, '.text.pkl'))
nodes_dic, edges_list = load_pickle(table_path)  # @load_or_make
count = 0

'''
# nodes table
print(','.join(('Id', 'Label')))
for node, node_value in nodes_dic.items():
    print(','.join((str(node_value), node)))
'''
'''
# edges table
print(','.join(('Id', 'Source', 'Target', 'Type')))
for t in set(edges_list):
    print(','.join((str(count), str(t[0]), str(t[1]), 'Undirected')))
    count+=1
'''