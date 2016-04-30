"""
Question classification: question Subjects
"""

import sys
import re
import nltk
import os
import math
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np

stopset=set(stopwords.words('english'))

file_dir='/Users/xiaoliu/Desktop/CS665 Project/ck12'
biology=[]
physics=[]
earth=[]
life=[]
chemistry=[]
physical=[]
for subdir,dirs, files in os.walk(file_dir):
	for file in files:
		filepath= subdir+os.sep+file
		if 'Biology' in filepath:
			biology.append(filepath)
		if 'Physics' in filepath:
			physics.append(filepath)
		if 'Earth' in filepath:
			earth.append(filepath)
		if 'Life' in filepath:
			life.append(filepath)
		if 'Chemistry' in filepath:
			chemistry.append(filepath)
		if 'Physical' in filepath:
			physical.append(filepath)

def word_counts(files):
	corpus=[]
	for file in files:
		text=open(file)
		words=nltk.word_tokenize(text.read().decode('utf8', 'ignore').lower())
		corpus.append([w for w in words if not w in stopset])
	corpus_text=sum(corpus,[])
	fd=FreqDist(corpus_text)
	#fd.pprint()
	return fd

biology_freq=word_counts(biology)
physics_freq=word_counts(physics)
earth_freq=word_counts(earth)
life_freq=word_counts(life)
chemistry_freq=word_counts(chemistry)
physical_freq=word_counts(physical)

file= open('/Users/xiaoliu/Desktop/CS665 Project/training_set.tsv')
questions=file.readlines()
for line in questions:
	elements=re.split('\t',line.decode('utf8', 'ignore'))
	qid=elements[0]
	qtext=elements[1]
	tokens=nltk.word_tokenize(qtext.lower())
	#tagged=nltk.pos_tag(tokens)
	#nouns= [word for (word,pos) in tagged if 'NN' in pos]
	nouns= [word for word in tokens if not word in stopset]
	score=[0.0]*6
	for noun in nouns:
		score[0]+= math.log(biology_freq.freq(noun)+0.000001)
		score[1]+= math.log(physics_freq.freq(noun)+0.000001)
		score[2]+= math.log(earth_freq.freq(noun)+0.000001)
		score[3]+= math.log(life_freq.freq(noun)+0.000001)
		score[4]+= math.log(chemistry_freq.freq(noun)+0.000001)
		score[5]+= math.log(physical_freq.freq(noun)+0.000001)
	index=np.argmax(score)
	print str(qid), str(index)







