import gensim
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import string
#nltk.download() 
text = pd.read_csv("C:/Users/DC20693/Desktop/text2train.csv")
print('load done')
wordnet_lemmatizer = WordNetLemmatizer()

def review_to_wordlist(review):
    words = review.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    words = [wordnet_lemmatizer.lemmatize(w) for w in words]
    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    translator = str.maketrans('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~',' '*31)
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        raw_sentence = raw_sentence.translate(translator)
        " ".join(raw_sentence.split())
        raw_sentence=raw_sentence.replace("GOF","gain_of_function")
        raw_sentence=raw_sentence.replace("LOF","loss_of_function")
        raw_sentence=raw_sentence.replace("gainof function","gain_of_function")
        raw_sentence=raw_sentence.replace("lossof function","loss_of_function")
        raw_sentence=raw_sentence.replace("gainof","gain_of_function")
        raw_sentence=raw_sentence.replace("lossof","loss_of_function")
        raw_sentence=raw_sentence.replace("gain of function","gain_of_function")
        raw_sentence=raw_sentence.replace("loss of function","loss_of_function")
        #raw_sentence=raw_sentence.replace("gain function","gain_of_function")
        #raw_sentence=raw_sentence.replace("loss function","loss_of_function")
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences

sentences = []  

for t in text["x"]:
    sentences += review_to_sentences(t, tokenizer)
print('tokenize done')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
num_features = 300    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3

model = gensim.models.word2vec.Word2Vec(sentences, workers=num_workers, 
            size=num_features, min_count = min_word_count,iter=10, 
            window = context, sample = downsampling)

model_name = "300features_8989_obs"
model.save(model_name)
