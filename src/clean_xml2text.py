"""
xml to text
simple wiki -> text
"""


import xml.etree.cElementTree as ET
import io

path_xmlfile = '../data/simplewiki.xml'
path_xmlfile = '../data/Wikipedia-20160210171947.xml'
path_txt = path_xmlfile + '.txt'
"""
tree = ET.ElementTree(file=path_xmlfile)
root = tree.getroot()
"""


with io.open(path_txt, mode='a', encoding='utf-8', errors='ignore') as f:
    for event, element in ET.iterparse(path_xmlfile):

        if event == 'end':
            if element.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
                wiki_title = element.text
            elif element.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
                wiki_text = element.text
                try:
                    wiki_text = wiki_text.replace('\n', ' ')
                    to_write = unicode(wiki_title + '\t' + wiki_text + '\n')

                    if wiki_title and wiki_text:
                        f.write(to_write)
                        wiki_title, wiki_text = '', ''
                except:
                    continue





