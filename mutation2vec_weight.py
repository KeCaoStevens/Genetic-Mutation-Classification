import gensim
import pandas as pd
import re
import string
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
model = gensim.models.Word2Vec.load('C:/Users/DC20693/Desktop/kaggle/200features_pubmed')
logging.info('word2vec loaded')
train = pd.read_pickle('C:/Users/dongg/Desktop/kaggle/kaggletext.pkl')
logging.info('dataframe loaded')
Paragraph = []
for i in range(0,8989):
    temp = ''
    for sen in train["Reduced"].iloc[i]:
        for word in sen:
            temp+=word + " "
    Paragraph.append(temp)
logging.info('dataframe cleaned')
vectorizer = TfidfVectorizer(min_df=20)
X = vectorizer.fit_transform(Paragraph)
logging.info('idf calculated')
def makeFeatureVec(words, model, num_features,counter,X=X,vectorizer=vectorizer):
    if (counter+1) in [5,10,20,50,100,200,500,1000,1500,2000,2500,3000,5000,6000,7000]:
        message = str(counter+1) + ' files processed'
        logging.info(message)        
    featureVecSen = np.zeros((num_features,),dtype="float32")
    nwords = 0
  
    index2word_set = set(model.wv.index2word)
    
    for sen in words:
        featureVec = np.zeros((num_features,),dtype="float32")
        for word in sen:
            if word in index2word_set: 
                nwords = nwords + 1
                try:
                    weight = X[counter, vectorizer.vocabulary_[word]]
                except KeyError:
                    weight = 0
                featureVec = np.add(featureVec,weight*model[word])
        featureVec = np.divide(featureVec,nwords)
        featureVecSen = np.add(featureVecSen,featureVec)
    if len(words)>0:
        featureVecOut = np.divide(featureVecSen,len(words))
    else:
        featureVecOut = featureVecSen
    return featureVecOut


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features,counter)
       counter = counter + 1
    return reviewFeatureVecs

trainDataVecs = getAvgFeatureVecs(train["Reduced"], model, 200 )
logging.info('vector calculated')
vector = pd.DataFrame(trainDataVecs)
vector.to_csv('C:/Users/dongg/Desktop/kaggle/vecOut.csv',index=False) 


