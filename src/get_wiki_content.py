"""
get wiki content based on certain ck12 keywords
"""


import wikipedia, io
from utils import pos_tag_word, get_VNA, add_bigram_trigram


path_keys = '../data/ck_keywords.txt'
path_wiki_content = '../data/corpus/wiki_content'
path_wiki_summary = '../data/corpus/wiki_summary'


B, T = True, True   # add bigram and trigram

with io.open(path_keys, mode='r', encoding='utf-8', errors='ignore') as key_f:
    content_f = io.open(path_wiki_content, mode='a', encoding='utf-8', errors='ignore')
    summary_f = io.open(path_wiki_summary, mode='a', encoding='utf-8', errors='ignore')

    all_wiki_urls = []

    for line in key_f:
        line = pos_tag_word(line)  # pos tag each word
        line = get_VNA(line, keepV=True, keepN=True, keepA=True)  # keep verb, noun and adj+adv

        if len(line) < 3: T = False
        elif len(line) < 2: B = False
        line = add_bigram_trigram(line, addB=B, addT=T)  # default add bigrams and trigrams

        print line
        for phrase in line:
            try:
                wiki_obj = wikipedia.page(phrase)
                wiki_url = wiki_obj.url
                if wiki_url not in all_wiki_urls:
                    all_wiki_urls.append(wiki_url)
                    wiki_content = wiki_obj.content
                    wiki_content = ' '.join(wiki_content.splitlines())
                    wiki_summary = wikipedia.summary(phrase)
                    wiki_summary = ' '.join(wiki_summary.splitlines())
                    content_f.write(phrase + '\t' + wiki_content + '\n')
                    summary_f.write(phrase + '\t' + wiki_summary + '\n')
            except:
                continue

content_f.close()
summary_f.close()









