# allen-ai-science-challenge  
The Allen Institute for Artificial Intelligence (AI2) is working to improve humanity through fundamental advances in artificial intelligence. One critical but challenging problem in AI is to demonstrate the ability to consistently understand and correctly answer general questions about the world. https://www.kaggle.com/c/the-allen-ai-science-challenge  

## main  
allen_challenge.py  

## question understanding
allen_challenge.py -> slim_flag = True  # if True: only use Noun, Verb, Adj+Adv

## features  
feature_extraction.py  
1. retrieval feature: BM25    
2. similarity feature: count number of words  
3. w2v feature (min, max, avg)  
4. google distance feature ( min, max, avg)  # not useful  
   co-occurrence of the word pairs using Normalized google distance
   

## ranking 
1. answer_ranking_perceptron.py   # use this one  
2. answer_ranking_svm.py  # too slow !!

## results
1. retrieval feature: 0.268  
2. similarity feature: 0.266  
3. w2v feature (min=0.268, max=0.258, avg=0.23)  
4. google distance feature ( min, max, avg)  # not useful  
   
combine: retrieval + similarity + w2v feature (min, max) = 0.268  
combine: retrieval + similarity = 0.28
