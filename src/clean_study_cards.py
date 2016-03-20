#!C:\Miniconda3\python.exe -u

"""
clean and reformat study cards corpus:
- quizlet_corpus.txt
- studystack_corpus.txt

use first notional words as document title
format:
[doc title] [\t] [text]
"""


import io, os, re


def read_study_cards(input_path, output_path):

    with io.open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        with io.open(output_path, mode='a', encoding='utf-8') as ff:
            stop_words = '^\d|first|third|lasts?|one|two|three|four|five|six|seven|eight|nine|etc|used?|with|so|that|this|other|either|closed?|open|line|steps?|from|list|most|in|if|on|some|only|of|types?|do|did|does|to|all|who|any|where|why|when|what|which|how|are|is|was|were|the|an?|our|it|give|define|example|you|can|each|really|there|name|main'
            stop_words_pat = re.compile(stop_words)

            for line in f:
                line = line.strip()

                if line.startswith(('<PAGE>', '<SECTION>')):  # empty line
                    continue
                else:
                    line_list = line.split()
                    for tok in line_list:
                        if stop_words_pat.match(tok):
                            continue
                        else:
                            doc_title = tok
                            break

                    to_write = ''.join((doc_title, '\t', line, '\n'))
                    ff.write(to_write)


################################################################
# main
path = '../data/corpus/study_cards/'
corpus = os.listdir(path)

for c in corpus:
    in_path = path + c
    out_path = in_path + '.clean'
    read_study_cards(in_path, out_path)

