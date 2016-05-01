# allen-ai-science-challenge  
The Allen Institute for Artificial Intelligence (AI2) is working to improve humanity through fundamental advances in artificial intelligence.   
One critical but challenging problem in AI is to demonstrate the ability to consistently understand and correctly answer general questions about the world. 
Is your model smarter than an 8th grader? [Read More] (https://www.kaggle.com/c/the-allen-ai-science-challenge)  


## Question and Answer Pre-process

### Question pre-process
- Remove punctuation 
- Convert to lowercase
- Part of speech tagging:  
Only use (nouns): [NN\*]  
Only use (noun, verb, adj/adv): [NN\* | VB\* | JJ\* | RB\*]  
- Concatenate question and each answer  

### Answer pre-process  
Replace:  
- `all of the above`: 16 in (2500 * 4 answers)  
(answer A + answer B + answer C)  
- `none of the above`: 4 in (2500 * 4 answers)  
(empty string)  
- `both A and B` & `both A and C`: 4 in (2500 * 4 answers)    
(answer A + answer B | answer C)  
[question_answer_preprocess.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/question_answer_preprocess.py)  

## Knowledge Source
### Data collection
- CK12: 36 books & 6 subjects
- Study Cards: quizlet & studystack
- Simple wiki: simplewiki-20150702-pages-articles-multistream.xml [get_wiki_content.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/get_wiki_content.py)
- Aristo table: Nov 2015, Snapshot
- SuperSenseTagger: to do: hyponymy & hypernymy query expansion
- Google ngram: to do: words distance [get_google_dic.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/get_google_dic.py)

### Data cleaning
- CK12: [book title] -> [section title] -> [text] [clean_ck12.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/clean_ck12.py)
- Study Cards: [first notional word] -> [text] [clean_study_cards.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/clean_study_cards.py)
- Simple wiki: xml to text [clean_xml2text.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/clean_xml2text.py)
- Aristo table: to do: data cleaning !

## Ranking Algorithm
Support Vector Machine for Ranking: [SVMrank] (https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)  
- Windows (32-bit)
- Use default setting to do: optimize parameters  
svm_rank_learn -c 20.0 train.dat model.dat  
svm_rank_classify ..\test.dat ..\model.dat ..\predictions  
- Prepare input data: [answer_ranking_features2txt.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/answer_ranking_features2txt.py)  
- Run SVMrank from Python: [answer_ranking_svmrank.py] (https://github.com/rarezhang/allen-ai-science-challenge/blob/master/src/answer_ranking_svmrank.py)

## Features 
- Retrieval Features
- Word2vec Features
- Network Features: soft inference 
- Question Classification Features: soft inference 

### Retrieval Features
- Index  
Index corpuses separately: CK12 | Study Cards | Simple Wiki  
- 3 fields:  
  - Data source (book title) -> classification features
  - Document name (section title | first notional word) -> classification features
  - Content -> retrieval features
- Search: to do: optimize parameters  
StandardAnalyzer | hitsPerPage = 5 | DefaultSimilarity  
- 18 retrieval features   
![18 retrieval feature](https://cloud.githubusercontent.com/assets/5633774/14943834/95d85408-0f98-11e6-9d2b-7f010da47393.png "18 retrieval feature")

### Word2vec Features
- Training Word2Vec Model   
Train corpuses separately: CK12 | Study Cards
- Cosine similarity  
Each token in question V.S each token in each answer
- Only use noun 
- 4 word2vec features 
![4 word2vec features](https://cloud.githubusercontent.com/assets/5633774/14943861/374a4666-0f99-11e6-8bdc-dd7528c55a86.png "4 word2vec features")

### Network Features: soft inference 
- Based on [Random walk inference and learning in a large scale knowledge base] (https://www.cs.cmu.edu/~tom/pubs/lao-emnlp11.pdf)
- Modify and Simplify 
![difference](https://cloud.githubusercontent.com/assets/5633774/14943886/98794f86-0f99-11e6-872b-7d0de552f891.png "difference")
- Random walk probability  
![Random walk probability](https://cloud.githubusercontent.com/assets/5633774/14943903/105ab724-0f9a-11e6-9dc8-471a496cd69a.png "Random walk probability")
  - Path 1: Q -> 1 -> A
    - Degree(node1) = 4
    - ProbRandomWalkQ-A = 0.25
  - Path 2: Q -> 2 -> 3 -> A
    - Degree(node2) = 3  and  Degree(node3) = 3  
    - ProbRandomWalkQ-A = 0.11  
- Buid network (Based on Aristo table)
to do: 1. Edges with attributes (e.g., `absorb` -> edge attribute)  2. Undirected to directed graph 
  - plants -> absorb -> minerals
  - plants -> absorb -> nutrients  
![Buid network](https://cloud.githubusercontent.com/assets/5633774/14943919/9ab9feb6-0f9a-11e6-9382-fe87efc3152b.png "Buid network")
- Index
  - Nodes: text
  - Search: Each question V.S each answer
    - StandardAnalyzer | hitsPerPage = 1 | DefaultSimilarity
			to do: optimize parameters
- 13 network features  
![13 network features](https://cloud.githubusercontent.com/assets/5633774/14943962/896869ee-0f9b-11e6-970c-08b2a864cd9c.png "13 network features")

### Question Classification Features: soft inference
#### Classification Features - Subjects
- add xxxxxxxxxxxxxxxxxxxx
- Index (3 fields)
  - Data source (book title) -> subjects classification 
  - Document name (section title) -> question type classification
  - Content
- Search  
text_query = QueryParser(version, 'text', analyzer).parse(QueryParser.escape(q_string))  
subject_query = QueryParser(version, 'corpus_name', analyzer).parse(QueryParser.escape(q_class))  
query = BooleanQuery()  
query.add(text_query, BooleanClause.Occur.SHOULD) #  the keyword SHOULD occur  
query.add(subject_query, BooleanClause.Occur.MUST) # the keyword MUST occur   
- 4 subjects classification features  
![4 subjects classification features](https://cloud.githubusercontent.com/assets/5633774/14943985/23679092-0f9c-11e6-894d-11b45f11c196.png " 4 subjects classification features")



#### Classification Features – Question type


## Performance



Performance:  
all features:  
svm_rank_.txt 0.5335968379446641  w2v + retrieval + network + question classification (subjects)  
svm_rank_.txt 0.5118577075098815  w2v + retrieval + network  
svm_rank_.txt 0.5316205533596838  w2v + retrieval  

all word2vec:  
svm_rank_w2v.txt 0.20355731225296442  

all retrieval:  
svm_rank_retrieval.txt 0.5335968379446641  

all network:
svm_rank_w2v.txt 0.13256547892154826  
 
question classification (subjects):  
svm_rank_class_sub.txt 0.43478260869565216 (subject_query, BooleanClause.Occur.MUST)   
svm_rank_class_sub.txt 0.4426877470355731 (subject_query, BooleanClause.Occur.SHOULD)   
svm_rank_noun_class_sub.txt 0.4268774703557312 (subject_query, BooleanClause.Occur.SHOULD + noun)   

corpus:  
svm_rank_ck12.txt 0.4683794466403162  
svm_rank_simple_wiki.txt 0.391304347826087  
svm_rank_study_cards.txt 0.5158102766798419  


