#!C:\Miniconda3\python.exe -u
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

import re, io, os
from spelling_correct import *

punctuation_list = ['.', '!', '?', ':', ';', ',', '-', '"', ')', '=']
check_spelling = False  # very slow


def read_textbook(input_path, output_path):
    chapter_pat = '^Chapter \d+|^\d+. [A-Z]{1}[a-z]+'  # chapter pattern (2 kinds)
    lesson_pat = '^[0-9]+\.[0-9]+ Lesson [0-9]+\.[0-9]+'  # lesson title pattern
    page_pat = '^[0-9]{1,4}'  # page number pattern
    section_pat = '^[A-Z]{1}'
    skip_pat = re.compile('explore\s*more|content|author|www.ck12.org|lesson\s*objectives|review|vocabulary|check\s*your\s*understanding|points\s*to\s*consider|further\s*reading|supplemental\s*links|figure|url|image\s*sources|media', re.IGNORECASE)
    section_content = ''
    new_section_title = ' '

    with io.open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        with io.open(output_path, mode='a', encoding='utf-8') as ff:

            for line in f:
                new_section_flag = False
                line = line.strip()

                skip_flag = re.match(page_pat, line) or skip_pat.match(line)

                if len(line.split()) == 0:  # empty line
                    continue
                elif re.match(chapter_pat, line):  # chapter title
                    print(line, ' a')
                    continue
                elif re.match(lesson_pat, line):  # lesson title
                    print(line, ' b')
                    continue
                elif skip_flag:  # should be after lesson pat
                    continue
                elif re.match(section_pat, line) and len(line.split()) <= 10 and (not line[-1] in punctuation_list):
                    line = ' '.join(re.findall('[A-Z][^A-Z]+', line))  # Split a string at uppercase letters
                    section_title = new_section_title
                    new_section_title = line
                    new_section_flag = True
                else:
                    if check_spelling:  # expensive
                        for w in line.split():   # spelling check
                            for letter in w:
                                if ord(letter) >= 128:
                                    correct_word(w)
                    content = line

                if new_section_flag:
                    if section_content != '':

                        section_title = re.sub('ﬁ', 'fi', section_title)
                        section_title = re.sub('ﬂ', 'fl', section_title)
                        section_title = re.sub('ﬀ', 'ff', section_title)

                        section_content = re.sub('ﬁ', 'fi', section_content)
                        section_content = re.sub('ﬂ', 'fl', section_content)
                        section_content = re.sub('ﬀ', 'ff', section_content)
                        section_content = re.sub('-\s+', '', section_content)  # e.g., "scien- tists" => "scientists"
                        section_content = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', section_content)

                        to_write = ' '.join((section_title, '\t', section_content, '\n'))
                        ff.write(to_write)

                    section_content = ''
                else:
                    section_content += ' '
                    section_content += content


################################################################
path = '../data/corpus/ck12/'
corpus = os.listdir(path)

for c in corpus:
    in_path = path + c
    out_path = in_path + '.clean'
    read_textbook(in_path, out_path)



