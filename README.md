# allen-ai-science-challenge  
The Allen Institute for Artificial Intelligence (AI2) is working to improve humanity through fundamental advances in artificial intelligence. One critical but challenging problem in AI is to demonstrate the ability to consistently understand and correctly answer general questions about the world. https://www.kaggle.com/c/the-allen-ai-science-challenge  

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


