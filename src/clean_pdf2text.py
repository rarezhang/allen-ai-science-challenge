# -*- coding: utf-8 -*-
"""
1. use pdfMiner commandline tools pdf2txt.py
commandline: python pdf2txt.py -o CK-12-Physical-Science-Concepts-For-Middle-School.txt -t text CK-12-Physical-Science-Concepts-For-Middle-School.pdf
pdf -> text file
2. manually remove all the content before chapter 1
text -> clean corpus
format: title [tab] text
3. manually change ﬁ to fi
ﬂ, ff, '- '
4. manually remove all ck12 text books from the corpus directory
"""

import re, io
from spelling_correct import *

punctuation_list = ['.', '!', '?', ':', ';', ',']
check_spelling = False  # very slow


def read_textbook(input_path, output_path):
    chapter_pat1 = '^Chapter \d+'
    chapter_pat2 = '^\d+. [A-Z]{1}[a-z]+'  # CK12_chemistry has different chapter title
    lesson_pat = '^[0-9]+\.[0-9]+ Lesson [0-9]+\.[0-9]+'  # lesson title pattern
    page_pat = '^[0-9]{1,4}'  # page number
    section_pat = '^[A-Z]{1}'
    section_content = ''
    new_section_title = ' '

    with io.open(input_path, 'r', encoding='utf-8', errors='ignore') as f:

        for line in f:
            new_section_flag = False
            line = line.strip()
            if len(line.split()) == 0:  # empty line
                continue
            elif re.match(chapter_pat1, line):  # chapter title
                print line, ' a'
                continue
            elif re.match(lesson_pat, line):  # lesson title
                print line, ' b'
                continue
            elif re.match(page_pat, line) or line == "www.ck12.org":  # / page / url: pattern should be after lesson pat
                continue
            elif re.match(section_pat, line) and len(line.split()) <= 10 and (not line[-1] in punctuation_list):
                line = line.lower()
                section_title = new_section_title
                new_section_title = line
                new_section_flag = True
            else:
                line = re.sub(r'- ', r'', line)
                if check_spelling:
                    for words in line.split():   # spelling check
                        for letter in words:
                            if ord(letter) >= 128:
                                correct_word(words)
                content = line

            if new_section_flag:
                if section_content != '':
                    to_write = unicode(section_title + '\t' + section_content + '\n')
                    with io.open(output_path, mode='a', encoding='utf-8') as ff:
                        ff.write(to_write)
                section_content = ''
            else:
                section_content += ' '
                section_content += content





in_path = '../data/corpus/CK12_Biology.txt'
in_path = '../data/corpus/CK12_Earth_Science.txt'
in_path = '../data/corpus/CK12_Life_Science.txt'
in_path = '../data/corpus/CK12_chemistry.txt'   # chapter pattern: use chapter_pat2

in_path = '../data/corpus/CK-12-Biology-Advanced-Concepts.txt'
in_path = '../data/corpus/CK-12-Biology-Concepts.txt'
in_path = '../data/corpus/CK-12-Biology-Concepts_b.txt'
in_path = '../data/corpus/CK-12-Chemistry-Concepts-Intermediate.txt'
in_path = '../data/corpus/CK-12-Earth-Science-Concepts-For-High-School.txt'
in_path = '../data/corpus/CK-12-Earth-Science-Concepts-For-Middle-School.txt'
in_path = '../data/corpus/CK-12-Life-Science-Concepts-For-Middle-School.txt'
in_path = '../data/corpus/CK-12-Physical-Science-Concepts-For-Middle-School.txt'
in_path = '../data/corpus/CK-12-Physics-Concepts-Intermediate.txt'

out_path = in_path + '.clean'
read_textbook(in_path, out_path)




