import gensim
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
import numpy as np
wordnet_lemmatizer = WordNetLemmatizer()
model = gensim.models.Word2Vec.load('C:/Users/DC20693/Documents/Hantao/Word2Vec/trainedModel/300features_function')
print('word2vec loaded')

train = pd.read_csv("C:/Users/DC20693/Desktop/targetOut.csv",encoding = "ISO-8859-1")

print('load done')


def review_to_wordlist(review):
    translator = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^`_{|}~')
    review=review.replace("GOF","gain of function")
    review=review.replace("LOF","loss of function")
    review=review.replace("gainof function","gain of function")
    review=review.replace("lossof function","loss of function")
    review=review.replace("gainof-function","gain of function")
    review=review.replace("lossof-function","loss of function")
    #remove numbers
    review = review.translate(translator)
    words = review.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    return(words)

def makeFeatureVec(words, model, num_features):
    
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
  
    index2word_set = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter = counter + 1
    return reviewFeatureVecs

clean_train_reviews = []
for review in train["Text"]:
    if type(review)==float:
        clean_train_reviews.append(review_to_wordlist('mutation'))
    else:
        clean_train_reviews.append(review_to_wordlist(review))
print("cleaning done")
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, 300 )

vector = pd.DataFrame(trainDataVecs)
vector.to_csv('C:/Users/DC20693/Desktop/vecOut.csv',index=False) 


