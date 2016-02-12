"""

"""

import io
from collections import defaultdict
from utils import *

path_google_dis = '../data/google/RTE3_RTE4_NGD.txt'
google_dic = defaultdict()
with io.open(path_google_dis, mode='r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line=line.split()
        google_distance = line[1]
        w1, w2 = line[0].split('_')
        google_dic[(w1,w2)] = google_distance
dump_feature_score('../data/google/google_distance_dic', google_dic)
